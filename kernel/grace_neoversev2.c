/**
 * @file grace_neoversev2.c
 * @author Enoch Jung
 * @author Seunghoon Lee
 * @brief dgemm for
          - cores : 72
          - A     : RowMajor
          - B     : RowMajor
          - C     : RowMajor
          - k     : even number
          - alpha : 1.0
          - beta  : 1.0
 * @date 2025-07-xx
 */

#define _GNU_SOURCE

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

#include <arm_neon.h>
#include <armpl.h>
#include <numa.h>
#include <pthread.h>

#include "common.h"

#ifdef OC
#include "grace_neoversev2.oc.inc"
#else
#include "grace_neoversev2.mc.inc"
#endif

#define MR 4
#define NR 8

#define _A_SIZE (sizeof(double) * (MB + MR) * KB)
#define _B_SIZE (sizeof(double) * KB * (NB + NR))

#if defined(USE_LDP)
#define VLD1_Q2(r1, r2, source) \
    " ldp q" #r1 ", q" #r2 ", [%[" #source "]] \t\n"
#define VLD1_Q2_F(r1, r2, source) \
    " ldp q" #r1 ", q" #r2 ", [%[" #source "]], #32 \t\n"
#define VLD1_Q4(r1, r2, r3, r4, source)              \
    " ldp q" #r1 ", q" #r2 ", [%[" #source "]] \t\n" \
    " ldp q" #r3 ", q" #r4 ", [%[" #source "], #32] \t\n"
#define VST1_Q2(r1, r2, source) \
    " stp q" #r1 ", q" #r2 ", [%[" #source "]] \t\n"
#define VST1_Q4(r1, r2, r3, r4, source)              \
    " stp q" #r1 ", q" #r2 ", [%[" #source "]] \t\n" \
    " stp q" #r3 ", q" #r4 ", [%[" #source "], #32] \t\n"
#else
#define VLD1_Q2(r1, r2, source)            \
    " ldr q" #r1 ", [%[" #source "]] \t\n" \
    " ldr q" #r2 ", [%[" #source "], #16] \t\n"
#define VLD1_Q2_F(r1, r2, source)               \
    " ldr q" #r1 ", [%[" #source "]], #16 \t\n" \
    " ldr q" #r2 ", [%[" #source "]], #16 \t\n"
#define VLD1_Q4(r1, r2, r3, r4, source)         \
    " ldr q" #r1 ", [%[" #source "]] \t\n"      \
    " ldr q" #r2 ", [%[" #source "], #16] \t\n" \
    " ldr q" #r3 ", [%[" #source "], #32] \t\n" \
    " ldr q" #r4 ", [%[" #source "], #48] \t\n"
#define VST1_Q2(r1, r2, source)            \
    " str q" #r1 ", [%[" #source "]] \t\n" \
    " str q" #r2 ", [%[" #source "], #16] \t\n"
#define VST1_Q4(r1, r2, r3, r4, source)         \
    " str q" #r1 ", [%[" #source "]] \t\n"      \
    " str q" #r2 ", [%[" #source "], #16] \t\n" \
    " str q" #r3 ", [%[" #source "], #32] \t\n" \
    " str q" #r4 ", [%[" #source "], #48] \t\n"
#endif

__forceinline void micro_kernel(
    uint64_t kk,
    const double *restrict _A,
    const double *restrict _B,
    double *restrict C,
    const uint64_t ldc)
{
    register float64x2_t v0 asm("v0");
    register float64x2_t v1 asm("v1");
    register float64x2_t v2 asm("v2");
    register float64x2_t v3 asm("v3");
    register float64x2_t v4 asm("v4");
    register float64x2_t v5 asm("v5");
    register float64x2_t v6 asm("v6");
    register float64x2_t v7 asm("v7");
    register float64x2_t v8 asm("v8");
    register float64x2_t v9 asm("v9");
    register float64x2_t v10 asm("v10");
    register float64x2_t v11 asm("v11");
    register float64x2_t v12 asm("v12");
    register float64x2_t v13 asm("v13");
    register float64x2_t v14 asm("v14");
    register float64x2_t v15 asm("v15");
    register float64x2_t v16 asm("v16");
    register float64x2_t v17 asm("v17");
    register float64x2_t v18 asm("v18");
    register float64x2_t v19 asm("v19");
    register float64x2_t v20 asm("v20");
    register float64x2_t v21 asm("v21");
    register float64x2_t v22 asm("v22");
    register float64x2_t v23 asm("v23");
    register float64x2_t v24 asm("v24");
    register float64x2_t v25 asm("v25");
    register float64x2_t v26 asm("v26");
    register float64x2_t v27 asm("v27");
    register float64x2_t v28 asm("v28");
    register float64x2_t v29 asm("v29");
    register float64x2_t v30 asm("v30");
    register float64x2_t v31 asm("v31");

#pragma unroll(MK_PREFETCH_C_DEPTH)
    for (uint8_t i = 0; i < MK_PREFETCH_C_DEPTH; ++i)
    {
        __builtin_prefetch(C + i * ldc, READ, MK_PREFETCH_C_LOCALITY);
    }

    register double *C0 = C + 0 * ldc;
    register double *C1 = C + 1 * ldc;
    register double *C2 = C + 2 * ldc;
    register double *C3 = C + 3 * ldc;

    asm volatile(
        VLD1_Q2_F(24, 25, A) //
        " movi   v0.2d, #0  \t\n"
        " movi   v1.2d, #0  \t\n"
        " movi   v2.2d, #0  \t\n"
        " movi   v3.2d, #0  \t\n"  //
        VLD1_Q4(16, 17, 18, 19, B) //
        " add     %[B], %[B], #64  \t\n"
        " movi   v4.2d, #0  \t\n"
        " movi   v5.2d, #0  \t\n"
        " movi   v6.2d, #0  \t\n"
        " movi   v7.2d, #0  \t\n"
        " movi   v8.2d, #0  \t\n"
        " movi   v9.2d, #0  \t\n"
        " movi  v10.2d, #0  \t\n"
        " movi  v11.2d, #0  \t\n"
        " movi  v12.2d, #0  \t\n"
        " movi  v13.2d, #0  \t\n"
        " movi  v14.2d, #0  \t\n"
        " movi  v15.2d, #0  \t\n"

        : [A] "+r"(_A), [B] "+r"(_B),
          [v0] "=w"(v0), [v1] "=w"(v1), [v2] "=w"(v2), [v3] "=w"(v3), [v4] "=w"(v4), [v5] "=w"(v5), [v6] "=w"(v6), [v7] "=w"(v7), [v8] "=w"(v8), [v9] "=w"(v9),
          [v10] "=w"(v10), [v11] "=w"(v11), [v12] "=w"(v12), [v13] "=w"(v13), [v14] "=w"(v14), [v15] "=w"(v15), [v16] "=w"(v16), [v17] "=w"(v17), [v18] "=w"(v18), [v19] "=w"(v19),
          [v24] "=w"(v24), [v25] "=w"(v25));

    kk >>= 1;
#pragma unroll(MK_UNROLL_DEPTH)
    for (uint64_t i = 0; i < kk; ++i)
    {
#if MK_PREFETCH_A_DISTANCE != 0
        __builtin_prefetch(_A + MK_PREFETCH_A_DISTANCE, READ, MK_PREFETCH_A_LOCALITY);
#endif
#pragma unroll(MK_PREFETCH_B_DEPTH)
        for (uint8_t i = 0; i < MK_PREFETCH_B_DEPTH; ++i)
        {
            __builtin_prefetch(_B + MK_PREFETCH_B_DISTANCE + 8 * i, READ, MK_PREFETCH_B_LOCALITY);
        }

        asm volatile(
            VLD1_Q4(20, 21, 22, 23, B) //
            " fmla   v0.2d, v16.2d, v24.d[0] \t\n"
            " fmla   v1.2d, v17.2d, v24.d[0] \t\n"
            " fmla   v2.2d, v18.2d, v24.d[0] \t\n"
            " fmla   v3.2d, v19.2d, v24.d[0] \t\n"

            " ldr    q26, [%[A]]             \t\n"
            " fmla   v4.2d, v16.2d, v24.d[1] \t\n"
            " fmla   v5.2d, v17.2d, v24.d[1] \t\n"
            " fmla   v6.2d, v18.2d, v24.d[1] \t\n"
            " fmla   v7.2d, v19.2d, v24.d[1] \t\n"

            " ldr    q27, [%[A], #16]        \t\n"
            " fmla   v8.2d, v16.2d, v25.d[0] \t\n"
            " fmla  v12.2d, v16.2d, v25.d[1] \t\n"
            " ldr    q16, [%[B], #64]    \t\n"
            " fmla   v9.2d, v17.2d, v25.d[0] \t\n"
            " fmla  v13.2d, v17.2d, v25.d[1] \t\n"
            " ldr    q17, [%[B], #80]    \t\n"
            " fmla  v10.2d, v18.2d, v25.d[0] \t\n"
            " fmla  v14.2d, v18.2d, v25.d[1] \t\n"
            " ldr    q18, [%[B], #96]    \t\n"
            " fmla  v11.2d, v19.2d, v25.d[0] \t\n"
            " fmla  v15.2d, v19.2d, v25.d[1] \t\n"
            " ldr    q19, [%[B], #112]   \t\n"

            " add   %[B], %[B], #128         \t\n"
            " fmla   v0.2d, v20.2d, v26.d[0] \t\n"
            " fmla   v1.2d, v21.2d, v26.d[0] \t\n"
            " fmla   v2.2d, v22.2d, v26.d[0] \t\n"
            " fmla   v3.2d, v23.2d, v26.d[0] \t\n"

            " ldr    q24, [%[A], #32]        \t\n"
            " fmla   v4.2d, v20.2d, v26.d[1] \t\n"
            " fmla   v5.2d, v21.2d, v26.d[1] \t\n"
            " fmla   v6.2d, v22.2d, v26.d[1] \t\n"
            " fmla   v7.2d, v23.2d, v26.d[1] \t\n"

            " ldr    q25, [%[A], #48]        \t\n"
            " add   %[A], %[A], #64          \t\n"
            " fmla   v8.2d, v20.2d, v27.d[0] \t\n"
            " fmla   v9.2d, v21.2d, v27.d[0] \t\n"
            " fmla  v10.2d, v22.2d, v27.d[0] \t\n"
            " fmla  v11.2d, v23.2d, v27.d[0] \t\n"
            " fmla  v12.2d, v20.2d, v27.d[1] \t\n"
            " fmla  v13.2d, v21.2d, v27.d[1] \t\n"
            " fmla  v14.2d, v22.2d, v27.d[1] \t\n"
            " fmla  v15.2d, v23.2d, v27.d[1] \t\n"

            : [A] "+r"(_A), [B] "+r"(_B),
              [v0] "+w"(v0), [v1] "+w"(v1), [v2] "+w"(v2), [v3] "+w"(v3), [v4] "+w"(v4), [v5] "+w"(v5), [v6] "+w"(v6), [v7] "+w"(v7), [v8] "+w"(v8), [v9] "+w"(v9),
              [v10] "+w"(v10), [v11] "+w"(v11), [v12] "+w"(v12), [v13] "+w"(v13), [v14] "+w"(v14), [v15] "+w"(v15), [v16] "+w"(v16), [v17] "+w"(v17), [v18] "+w"(v18), [v19] "+w"(v19),
              [v20] "+w"(v20), [v21] "+w"(v21), [v22] "+w"(v22), [v23] "+w"(v23), [v24] "+w"(v24), [v25] "+w"(v25), [v26] "+w"(v26), [v27] "+w"(v27));
    }

    asm volatile(
        VLD1_Q4(16, 17, 18, 19, C0) //
        VLD1_Q4(20, 21, 22, 23, C1) //
        VLD1_Q4(24, 25, 26, 27, C2) //
        VLD1_Q4(28, 29, 30, 31, C3) //

        " fadd  v16.2d, v16.2d, v0.2d    \t\n"
        " fadd  v17.2d, v17.2d, v1.2d    \t\n"
        " fadd  v18.2d, v18.2d, v2.2d    \t\n"
        " fadd  v19.2d, v19.2d, v3.2d    \t\n" //
        VST1_Q4(16, 17, 18, 19, C0)            //

        " fadd  v20.2d, v20.2d, v4.2d    \t\n"
        " fadd  v21.2d, v21.2d, v5.2d    \t\n"
        " fadd  v22.2d, v22.2d, v6.2d    \t\n"
        " fadd  v23.2d, v23.2d, v7.2d    \t\n" //
        VST1_Q4(20, 21, 22, 23, C1)            //

        " fadd  v24.2d, v24.2d,  v8.2d   \t\n"
        " fadd  v25.2d, v25.2d,  v9.2d   \t\n"
        " fadd  v26.2d, v26.2d, v10.2d   \t\n"
        " fadd  v27.2d, v27.2d, v11.2d   \t\n" //
        VST1_Q4(24, 25, 26, 27, C2)            //

        " fadd  v28.2d, v28.2d, v12.2d   \t\n"
        " fadd  v29.2d, v29.2d, v13.2d   \t\n"
        " fadd  v30.2d, v30.2d, v14.2d   \t\n"
        " fadd  v31.2d, v31.2d, v15.2d   \t\n" //
        VST1_Q4(28, 29, 30, 31, C3)            //

        : [v0] "+w"(v0), [v1] "+w"(v1), [v2] "+w"(v2), [v3] "+w"(v3), [v4] "+w"(v4), [v5] "+w"(v5), [v6] "+w"(v6), [v7] "+w"(v7), [v8] "+w"(v8), [v9] "+w"(v9),
          [v10] "+w"(v10), [v11] "+w"(v11), [v12] "+w"(v12), [v13] "+w"(v13), [v14] "+w"(v14), [v15] "+w"(v15), [v16] "+w"(v16), [v17] "+w"(v17), [v18] "+w"(v18), [v19] "+w"(v19),
          [v20] "+w"(v20), [v21] "+w"(v21), [v22] "+w"(v22), [v23] "+w"(v23), [v24] "+w"(v24), [v25] "+w"(v25), [v26] "+w"(v26), [v27] "+w"(v27), [v28] "+w"(v28), [v29] "+w"(v29),
          [v30] "+w"(v30), [v31] "+w"(v31)
        : [C0] "r"(C0),
          [C1] "r"(C1),
          [C2] "r"(C2),
          [C3] "r"(C3));
}

void micro_dxpy(
    const uint64_t m,
    const uint64_t n,
    double *restrict C,
    const uint64_t ldc,
    const double *restrict _C)
{
    for (uint64_t i = 0; i < m; ++i)
    {
        for (uint64_t j = 0; j < n; ++j)
        {
            C[j] += _C[j];
        }
        C += ldc;
        _C += NR;
    }
}

void inner_kernel(
    const uint64_t mm,
    const uint64_t nn,
    const uint64_t kk,
    const double *restrict _A,
    const double *restrict B,
    double *restrict C,
    const uint64_t ldc)
{
    const uint64_t mmc = ROUND_UP(mm, MR);
    const uint64_t mmr = mm % MR;
    const uint64_t nnc = ROUND_UP(nn, NR);
    const uint64_t nnr = nn % NR;

    const double *A;

    for (uint64_t nni = 0; nni < nnc; ++nni)
    {
        const uint64_t nnn = (nni != nnc - 1 || nnr == 0) ? NR : nnr;

        A = _A;
        for (uint64_t mmi = 0; mmi < mmc; ++mmi)
        {
            const uint64_t mmm = (mmi != mmc - 1 || mmr == 0) ? MR : mmr;

            if (LIKELY(mmm == MR && nnn == NR))
            {
                micro_kernel(kk, A, B, C + mmi * MR * ldc + nni * NR, ldc);
            }
            else
            {
                double _C[MR * NR] __attribute__((aligned(CACHE_LINE)));
                micro_kernel(kk, A, B, _C, NR);
                micro_dxpy(mmm, nnn, C + mmi * MR * ldc + nni * NR, ldc, _C);
            }

            A += MR * kk;
        }

        B += NR * kk;
    }
}

void pack_arc(
    const uint64_t mm,
    const uint64_t kk,
    const double *restrict A,
    const uint64_t lda,
    double *restrict B)
{
    const uint64_t mmc = ROUND_UP(mm, MR);
    uint64_t mmr = MR;
    const uint64_t mmr_ = mm % MR;
    if (mmr_ != 0)
    {
        mmr = mmr_;
    }

    const uint64_t kkc = ROUND_UP(kk, CACHE_ELEM);
    uint64_t kkr = CACHE_ELEM;
    const uint64_t kkr_ = kk % CACHE_ELEM;
    if (kkr_ != 0)
    {
        kkr = kkr_;
    }

    const double *A_acc;
    double *B_acc;
    double *inner_acc;

    for (uint64_t mmi = 0; LIKELY(mmi < mmc); ++mmi)
    {
        const uint64_t mmm = mmi != mmc - 1 ? MR : mmr;

#pragma unroll(ARC_PREFETCH_DEPTH)
        for (uint8_t i = 0; i < ARC_PREFETCH_DEPTH; ++i)
        {
            __builtin_prefetch(A + lda * 0 + CACHE_ELEM * i, READ, ARC_PREFETCH_LOCALITY);
            __builtin_prefetch(A + lda * 1 + CACHE_ELEM * i, READ, ARC_PREFETCH_LOCALITY);
            __builtin_prefetch(A + lda * 2 + CACHE_ELEM * i, READ, ARC_PREFETCH_LOCALITY);
            __builtin_prefetch(A + lda * 3 + CACHE_ELEM * i, READ, ARC_PREFETCH_LOCALITY);
        }

        A_acc = A;
        B_acc = B;
        for (uint64_t kki = 0; UNLIKELY(kki < kkc); ++kki)
        {
            const uint64_t kkk = kki != kkc - 1 ? CACHE_ELEM : kkr;

            if (LIKELY(mmm == MR && kkk == CACHE_ELEM))
            {
                const double *A0 = A_acc + lda * 0;
                const double *A1 = A_acc + lda * 1;
                const double *A2 = A_acc + lda * 2;
                const double *A3 = A_acc + lda * 3;

                inner_acc = B_acc;
                asm volatile(
                    VLD1_Q4(0, 4, 8, 12, A0)  //
                    VLD1_Q4(1, 5, 9, 13, A1)  //
                    VLD1_Q4(2, 6, 10, 14, A2) //
                    VLD1_Q4(3, 7, 11, 15, A3) //

                    " st4   {v0.2d-v3.2d},   [%[B]], #64  \t\n"
                    " st4   {v4.2d-v7.2d},   [%[B]], #64  \t\n"
                    " st4   {v8.2d-v11.2d},  [%[B]], #64  \t\n"
                    " st4   {v12.2d-v15.2d}, [%[B]]  \t\n"

                    : [A0] "+r"(A0),
                      [A1] "+r"(A1),
                      [A2] "+r"(A2),
                      [A3] "+r"(A3),
                      [B] "+r"(inner_acc));
            }
            else
            {
                for (uint64_t mmmi = 0; mmmi < mmm; ++mmmi)
                {
                    for (uint64_t kkki = 0; kkki < kkk; ++kkki)
                    {
                        B_acc[mmmi + kkki * MR] = A_acc[mmmi * lda + kkki];
                    }
                }
            }

            A_acc += CACHE_ELEM;
            B_acc += MR * CACHE_ELEM;
        }

        A += MR * lda;
        B += MR * kk;
    }
}

void pack_brr(
    const uint64_t kk,
    const uint64_t nn,
    const double *restrict B,
    const uint64_t ldb,
    double *restrict _B)
{
    const uint64_t nnc = ROUND_UP(nn, NR);
    for (uint64_t j = 0; j < kk; ++j)
    {
        register const double *B_acc = B;
        register double *_B_acc = _B;
#pragma unroll(NB / NR)
        for (uint64_t nni = 0; LIKELY(nni < nnc); ++nni)
        {
            register float64x2x4_t vec;
            vec = vld1q_f64_x4(B_acc);
            vst1q_f64_x4(_B_acc, vec);

            B_acc += NR;
            _B_acc += kk * NR;
        }
        B += ldb;
        _B += NR;
    }
}

struct thread_info
{
    pthread_t tid;
    uint64_t pid;
    uint64_t m;
    uint64_t n;
    uint64_t k;
    const double *A;
    uint64_t lda;
    const double *B;
    uint64_t ldb;
    double *restrict C;
    uint64_t ldc;
    double *_A;
    double *_B;
};

#ifdef OC
static double *_A = NULL;
static double *_B = NULL;
#else
static double *_A[TOTAL_CORE] = {
    NULL,
};
static double *_B[TOTAL_CORE] = {
    NULL,
};
#endif

__forceinline void middle_kernel(const uint64_t m, const uint64_t n, const uint64_t k, const double *A,
                                 const uint64_t lda, const double *B, const uint64_t ldb, double *C, const uint64_t ldc, double *_A, double *_B)
{
    const uint64_t mc = ROUND_UP(m, MB);
    const uint64_t mr = m % MB;
    const uint64_t nc = ROUND_UP(n, NB);
    const uint64_t nr = n % NB;
    const uint64_t kc = ROUND_UP(k, KB);
    const uint64_t kr = k % KB;

    for (uint64_t ni = 0; ni < nc; ++ni)
    {
        const uint64_t nn = (ni != nc - 1 || nr == 0) ? NB : nr;

        for (uint64_t ki = 0; ki < kc; ++ki)
        {
            const uint64_t kk = (ki != kc - 1 || kr == 0) ? KB : kr;

            pack_brr(kk, nn, B + ki * KB * ldb + ni * NB, ldb, _B);

            for (uint64_t mi = 0; mi < mc; ++mi)
            {
                const uint64_t mm = (mi != mc - 1 || mr == 0) ? MB : mr;

                pack_arc(mm, kk, A + mi * MB * lda + ki * KB, lda, _A);

                inner_kernel(mm, nn, kk, _A, _B, C + mi * MB * ldc + ni * NB, ldc);
            }
        }
    }
}

#ifndef OC
static void *thread_routine(void *arg)
{
    struct thread_info *tinfo = arg;
    const uint64_t pid = tinfo->pid;
    const uint64_t m = tinfo->m;
    const uint64_t n = tinfo->n;
    const uint64_t k = tinfo->k;
    const double *A = tinfo->A;
    const uint64_t lda = tinfo->lda;
    const double *B = tinfo->B;
    const uint64_t ldb = tinfo->ldb;
    double *C = tinfo->C;
    const uint64_t ldc = tinfo->ldc;

    middle_kernel(m, n, k, A, lda, B, ldb, C, ldc, _A[pid], _B[pid]);

    return NULL;
}
#endif

__attribute__((constructor)) void alloc_buffers()
{
#ifdef OC
    if (_A == NULL)
    {
        _A = numa_alloc(_A_SIZE);
        _B = numa_alloc(_B_SIZE);
    }
#else
    if (_A[0] == NULL)
    {
        for (uint64_t i = 0; i < TOTAL_CORE; ++i)
        {
            _A[i] = numa_alloc(_A_SIZE);
            _B[i] = numa_alloc(_B_SIZE);
        }
    }
#endif
}

__attribute__((destructor)) void free_buffers()
{
#ifdef OC
    if (_A != NULL)
    {
        numa_free(_A, _A_SIZE);
        numa_free(_B, _B_SIZE);
        _A = NULL;
    }
#else
    if (_A[0] != NULL)
    {
        for (uint64_t i = 0; i < TOTAL_CORE; ++i)
        {
            numa_free(_A[i], _A_SIZE);
            numa_free(_B[i], _B_SIZE);
        }
        _A[0] = NULL;
    }
#endif
}

__forceinline uint64_t min(const uint64_t a, const uint64_t b)
{
    return a < b ? a : b;
}

void call_dgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                CBLAS_TRANSPOSE TransB, int64_t m, int64_t n, int64_t k,
                double alpha, const double *A, int64_t lda, const double *B,
                int64_t ldb, double beta, double *C, int64_t ldc)
{
    const bool is_C_row = (layout == CblasRowMajor ? 1 : 0);
    const bool is_A_row = (TransA == CblasTrans ? !is_C_row : is_C_row);
    const bool is_B_row = (TransB == CblasTrans ? !is_C_row : is_C_row);

    assert(is_A_row == 1);
    assert(is_B_row == 1);
    assert(is_C_row == 1);
    assert(k % 2 == 0);
    assert(alpha == 1.0);
    assert(beta == 1.0);

    alloc_buffers();

#ifdef OC
    middle_kernel(m, n, k, A, lda, B, ldb, C, ldc, _A, _B);
#else
    const uint64_t total_m_jobs = ROUND_UP(m, MR);
    const uint64_t min_each_m_jobs = total_m_jobs / CM;
    const uint64_t rest_m_jobs = total_m_jobs % CM;

    const uint64_t total_n_jobs = ROUND_UP(n, NR);
    const uint64_t min_each_n_jobs = total_n_jobs / CN;
    const uint64_t rest_n_jobs = total_n_jobs % CN;

    struct thread_info tinfo[TOTAL_CORE];
    cpu_set_t mask;

    pthread_attr_t attr;
    pthread_attr_init(&attr);

    for (uint64_t pid = 0; pid < TOTAL_CORE; ++pid)
    {
        const uint64_t m_pid = pid % CM;
        const uint64_t n_pid = pid / CM;

        const uint64_t my_m_idx_start = m_pid * min_each_m_jobs + min(m_pid, rest_m_jobs);
        const uint64_t my_m_idx_end = (m_pid + 1) * min_each_m_jobs + min(m_pid + 1, rest_m_jobs);
        const uint64_t my_m_start = min(my_m_idx_start * MR, m);
        const uint64_t my_m_end = min(my_m_idx_end * MR, m);
        const uint64_t my_m_size = my_m_end - my_m_start;

        const uint64_t my_n_idx_start = n_pid * min_each_n_jobs + min(n_pid, rest_n_jobs);
        const uint64_t my_n_idx_end = (n_pid + 1) * min_each_n_jobs + min(n_pid + 1, rest_n_jobs);
        const uint64_t my_n_start = min(my_n_idx_start * NR, n);
        const uint64_t my_n_end = min(my_n_idx_end * NR, n);
        const uint64_t my_n_size = my_n_end - my_n_start;

        const double *A_start = A + my_m_start * lda;
        const double *B_start = B + my_n_start * 1;
        double *C_start = C + my_m_start * ldc + my_n_start * 1;

        tinfo[pid].pid = pid;
        tinfo[pid].m = my_m_size;
        tinfo[pid].n = my_n_size;
        tinfo[pid].k = k;
        tinfo[pid].A = A_start;
        tinfo[pid].lda = lda;
        tinfo[pid].B = B_start;
        tinfo[pid].ldb = ldb;
        tinfo[pid].C = C_start;
        tinfo[pid].ldc = ldc;
        tinfo[pid]._A = _A[pid];
        tinfo[pid]._B = _B[pid];

        CPU_ZERO(&mask);
        CPU_SET(pid, &mask);

        pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &mask);
        pthread_create(&tinfo[pid].tid, &attr, &thread_routine, &tinfo[pid]);
    }

    pthread_attr_destroy(&attr);

    for (uint64_t pid = 0; pid < TOTAL_CORE; ++pid)
    {
        void *res;
        pthread_join(tinfo[pid].tid, &res);
    }
#endif
}
