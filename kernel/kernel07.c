/**
 * @file kernel07
 * @author Enoch Jung
 * @author Seunghoon Lee
 * @brief dgemm for
          - core  : 64 cores,
          - A     : RowMajor
          - B     : RowMajor
          - C     : RowMajor
          - k     : even number
          - alpha : 1.0
          - beta  : 1.0
 * @date 2023-10-16
 * @date 2025-01-22
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

#define PTHREAD_CALL(expr)  \
    do                      \
    {                       \
        int err = (expr);   \
        if (err == 0)       \
        {                   \
            break;          \
        }                   \
        perror(#expr);      \
        exit(EXIT_FAILURE); \
    } while (0)

#define UNLIKELY(expr) __builtin_expect(!!(expr), 0)
#define LIKELY(expr) __builtin_expect(!!(expr), 1)

#define ROUND_UP(a, b) ((a + b - 1) / b)

#define READ 0
#define WRITE 1

#define LOCALITY_NONE 0
#define LOCALITY_LOW 1
#define LOCALITY_MODERATE 2
#define LOCALITY_HIGH 3

#define PAGE_SIZE 4096
#define CACHE_LINE 64
#define CACHE_ELEM (CACHE_LINE / 8)

#ifdef OC
#define TOTAL_CORE 1
#define NUMA_NODE 1
#else
#define TOTAL_CORE 64
#define NUMA_NODE 2
#endif

#ifdef OC
#define CM 1
#else
#ifndef CM
#define CM 8
#endif
#endif

#define CN (TOTAL_CORE / CM)

#define MR 4
#define NR 8

// tunable parameters
#ifndef MB
#define MB (MR * 9)
#endif

#ifndef NB
#define NB (NR * 43)
#endif

#ifndef KB
#define KB 280
#endif

// #define MK_PREFETCH_C

// #define USE_LDP

#ifndef MK_PREFETCH_A_DISTANCE
#define MK_PREFETCH_A_DISTANCE ((MR * 2) * 5)
#endif

#ifndef MK_PREFETCH_B_DISTANCE
#define MK_PREFETCH_B_DISTANCE ((NR * 2) * 5)
#endif

// 0 ~ 4
#ifndef MK_PREFETCH_B_DEPTH
#define MK_PREFETCH_B_DEPTH 2
#endif

#ifndef ARC_PREFETCH_DEPTH
#define ARC_PREFETCH_DEPTH 4
#endif

#ifndef ARC_PREFETCH_LOCALITY
#define ARC_PREFETCH_LOCALITY LOCALITY_NONE
#endif
// tunable parameters

#if defined(USE_LDP)
#define VLD2(r1, r2, source) \
    " ldp q" #r1 ", q" #r2 ", [%[" #source "]] \t\n"
#define VLD2F(r1, r2, source) \
    " ldp q" #r1 ", q" #r2 ", [%[" #source "]], #32 \t\n"
#define VLD4(r1, r2, r3, r4, source)                 \
    " ldp q" #r1 ", q" #r2 ", [%[" #source "]] \t\n" \
    " ldp q" #r3 ", q" #r4 ", [%[" #source "], #32] \t\n"
#define VST2(r1, r2, source) \
    " stp q" #r1 ", q" #r2 ", [%[" #source "]] \t\n"
#define VST4(r1, r2, r3, r4, source)                 \
    " stp q" #r1 ", q" #r2 ", [%[" #source "]] \t\n" \
    " stp q" #r3 ", q" #r4 ", [%[" #source "], #32] \t\n"
#else
#define VLD2(r1, r2, source)               \
    " ldr q" #r1 ", [%[" #source "]] \t\n" \
    " ldr q" #r2 ", [%[" #source "], #16] \t\n"
#define VLD2F(r1, r2, source)                   \
    " ldr q" #r1 ", [%[" #source "]], #16 \t\n" \
    " ldr q" #r2 ", [%[" #source "]], #16 \t\n"
#define VLD4(r1, r2, r3, r4, source)            \
    " ldr q" #r1 ", [%[" #source "]] \t\n"      \
    " ldr q" #r2 ", [%[" #source "], #16] \t\n" \
    " ldr q" #r3 ", [%[" #source "], #32] \t\n" \
    " ldr q" #r4 ", [%[" #source "], #48] \t\n"
#define VST2(r1, r2, source)               \
    " str q" #r1 ", [%[" #source "]] \t\n" \
    " str q" #r2 ", [%[" #source "], #16] \t\n"
#define VST4(r1, r2, r3, r4, source)            \
    " str q" #r1 ", [%[" #source "]] \t\n"      \
    " str q" #r2 ", [%[" #source "], #16] \t\n" \
    " str q" #r3 ", [%[" #source "], #32] \t\n" \
    " str q" #r4 ", [%[" #source "], #48] \t\n"
#endif

void micro_kernel(
    const uint64_t kk,
    const double *restrict _A,
    const double *restrict _B,
    double *restrict C,
    const uint64_t ldc)
{
    register double *C0 = C + 0 * ldc;
    register double *C1 = C + 1 * ldc;
    register double *C2 = C + 2 * ldc;
    register double *C3 = C + 3 * ldc;

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

#ifdef MK_PREFETCH_C
    __builtin_prefetch(C0, READ, LOCALITY_HIGH);
    __builtin_prefetch(C1, READ, LOCALITY_HIGH);
    __builtin_prefetch(C2, READ, LOCALITY_HIGH);
    __builtin_prefetch(C3, READ, LOCALITY_HIGH);
#endif

    asm volatile(
        VLD2F(24, 25, A) //
        " movi   v0.2d, #0  \t\n"
        " movi   v1.2d, #0  \t\n"
        " movi   v2.2d, #0  \t\n"
        " movi   v3.2d, #0  \t\n" VLD4(16, 17, 18, 19, B) //
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

#pragma unroll(2)
    for (uint64_t i = 0; i < kk >> 1; ++i)
    {
#if MK_PREFETCH_A_DISTANCE != 0
        __builtin_prefetch(_A + MK_PREFETCH_A_DISTANCE, READ, LOCALITY_NONE);
#endif
#pragma unroll(MK_PREFETCH_B_DEPTH)
        for (uint8_t i = 0; i < MK_PREFETCH_B_DEPTH; ++i)
        {
            __builtin_prefetch(_B + MK_PREFETCH_B_DISTANCE + 8 * i, READ, LOCALITY_HIGH);
        }

        asm volatile(
            VLD4(20, 21, 22, 23, B) //
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
            " fmla   v9.2d, v17.2d, v25.d[0] \t\n"
            " fmla  v10.2d, v18.2d, v25.d[0] \t\n"
            " fmla  v11.2d, v19.2d, v25.d[0] \t\n"
            " fmla  v12.2d, v16.2d, v25.d[1] \t\n"
            " fmla  v13.2d, v17.2d, v25.d[1] \t\n"
            " fmla  v14.2d, v18.2d, v25.d[1] \t\n"
            " fmla  v15.2d, v19.2d, v25.d[1] \t\n"

#ifdef USE_LDP
            " ldp    q16, q17, [%[B], #64]   \t\n"
            " ldp    q18, q19, [%[B], #96]   \t\n"
#else
            " ldr    q16, [%[B], #64]    \t\n"
            " ldr    q17, [%[B], #80]    \t\n"
            " ldr    q18, [%[B], #96]    \t\n"
            " ldr    q19, [%[B], #112]   \t\n"
#endif
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

    //    __builtin_prefetch(A_next + 0x00, 0, 3);
    //    __builtin_prefetch(B_next + 0x00, 0, 3);

    asm volatile(
        VLD4(16, 17, 18, 19, C0) //
        VLD4(20, 21, 22, 23, C1) //
        VLD4(24, 25, 26, 27, C2) //
        VLD4(28, 29, 30, 31, C3) //

        " fadd  v16.2d, v16.2d, v0.2d    \t\n"
        " fadd  v17.2d, v17.2d, v1.2d    \t\n"
        " fadd  v18.2d, v18.2d, v2.2d    \t\n"
        " fadd  v19.2d, v19.2d, v3.2d    \t\n" //
        VST4(16, 17, 18, 19, C0)               //

        " fadd  v20.2d, v20.2d, v4.2d    \t\n"
        " fadd  v21.2d, v21.2d, v5.2d    \t\n"
        " fadd  v22.2d, v22.2d, v6.2d    \t\n"
        " fadd  v23.2d, v23.2d, v7.2d    \t\n" //
        VST4(20, 21, 22, 23, C1)               //

        " fadd  v24.2d, v24.2d,  v8.2d   \t\n"
        " fadd  v25.2d, v25.2d,  v9.2d   \t\n"
        " fadd  v26.2d, v26.2d, v10.2d   \t\n"
        " fadd  v27.2d, v27.2d, v11.2d   \t\n" //
        VST4(24, 25, 26, 27, C2)               //

        " fadd  v28.2d, v28.2d, v12.2d   \t\n"
        " fadd  v29.2d, v29.2d, v13.2d   \t\n"
        " fadd  v30.2d, v30.2d, v14.2d   \t\n"
        " fadd  v31.2d, v31.2d, v15.2d   \t\n" //
        VST4(28, 29, 30, 31, C3)               //

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
#pragma unroll(2)
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
    uint64_t mmr = MR;
    const uint64_t mmr_ = mm % MR;
    if (mmr_ != 0)
    {
        mmr = mmr_;
    }

    const uint64_t nnc = ROUND_UP(nn, NR);
    uint64_t nnr = NR;
    const uint64_t nnr_ = nn % NR;
    if (nnr_ != 0)
    {
        nnr = nnr_;
    }

    const double *A;

    for (uint64_t nni = 0; nni < nnc; ++nni)
    {
        const uint64_t nnn = LIKELY(nni != nnc - 1) ? NR : nnr;

        A = _A;
        for (uint64_t mmi = 0; mmi < mmc; ++mmi)
        {
            const uint64_t mmm = LIKELY(mmi != mmc - 1) ? MR : mmr;

            if (LIKELY(mmm == MR && nnn == NR))
            {
                micro_kernel(kk, A, B, C + mmi * MR * ldc + nni * NR, ldc);
            }
            else
            {
                double _C[MR * NR] __attribute__((aligned(CACHE_LINE))) = {};
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
        const register uint64_t mmm = LIKELY(mmi != mmc - 1) ? MR : mmr;

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
            const uint64_t kkk = LIKELY(kki != kkc - 1) ? CACHE_ELEM : kkr;

            if (LIKELY(mmm == MR && kkk == CACHE_ELEM))
            {
                const double *A0 = A_acc + lda * 0;
                const double *A1 = A_acc + lda * 1;
                const double *A2 = A_acc + lda * 2;
                const double *A3 = A_acc + lda * 3;

                inner_acc = B_acc;
                asm volatile(
                    VLD4(0, 4, 8, 12, A0)  //
                    VLD4(1, 5, 9, 13, A1)  //
                    VLD4(2, 6, 10, 14, A2) //
                    VLD4(3, 7, 11, 15, A3) //

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
        const register double *Bm = B;
        register double *_Bm = _B;
#pragma unroll(NB / NR)
        for (uint64_t nni = 0; LIKELY(nni < nnc); ++nni)
        {
            register float64x2x4_t vec;
            vec = vld1q_f64_x4(Bm);
            vst1q_f64_x4(_Bm, vec);

            Bm += NR;
            _Bm += kk * NR;
        }
        B += ldb;
        _B += NR;
    }
}

struct thread_info
{
    pthread_t thread_id;
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

static void *middle_kernel(
    void *arg)
{
    struct thread_info *tinfo = arg;
    const uint64_t m = tinfo->m;
    const uint64_t n = tinfo->n;
    const uint64_t k = tinfo->k;
    const double *A = tinfo->A;
    const uint64_t lda = tinfo->lda;
    const double *B = tinfo->B;
    const uint64_t ldb = tinfo->ldb;
    double *C = tinfo->C;
    const uint64_t ldc = tinfo->ldc;
    double *_A = tinfo->_A;
    double *_B = tinfo->_B;

    const uint64_t mc = ROUND_UP(m, MB);
    uint64_t mr = MB;
    const uint64_t mr_ = m % MB;
    if (mr_ != 0)
    {
        mr = mr_;
    }
    const uint64_t nc = ROUND_UP(n, NB);
    uint64_t nr = NB;
    const uint64_t nr_ = n % NB;
    if (nr_ != 0)
    {
        nr = nr_;
    }
    const uint64_t kc = ROUND_UP(k, KB);
    uint64_t kr = KB;
    const uint64_t kr_ = k % KB;
    if (kr_ != 0)
    {
        kr = kr_;
    }

    for (uint64_t ni = 0; ni < nc; ++ni)
    {
        const uint64_t nn = ni != nc - 1 ? NB : nr;

        for (uint64_t ki = 0; ki < kc; ++ki)
        {
            const register uint64_t kk = ki != kc - 1 ? KB : kr;

            pack_brr(kk, nn, B + ki * KB * ldb + ni * NB, ldb, _B);

            for (uint64_t mi = 0; mi < mc; ++mi)
            {
                const register uint64_t mm = mi != mc - 1 ? MB : mr;

                pack_arc(mm, kk, A + mi * MB * lda + ki * KB, lda, _A);

                inner_kernel(mm, nn, kk, _A, _B, C + mi * MB * ldc + ni * NB, ldc);
            }
        }
    }

    return NULL;
}

#define MIN(a, b) ((a < b) ? (a) : (b))

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
    assert(alpha == 1.0);
    assert(beta == 1.0);

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
        const uint64_t nid = pid / (CM * CN / NUMA_NODE);
        const uint64_t m_pid = pid % CM;
        const uint64_t n_pid = pid / CM;

        const uint64_t my_m_idx_start = m_pid * min_each_m_jobs + MIN(m_pid, rest_m_jobs);
        const uint64_t my_m_idx_end = (m_pid + 1) * min_each_m_jobs + MIN(m_pid + 1, rest_m_jobs);
        const uint64_t my_m_start = MIN(my_m_idx_start * MR, m);
        const uint64_t my_m_end = MIN(my_m_idx_end * MR, m);
        const uint64_t my_m_size = my_m_end - my_m_start;

        const uint64_t my_n_idx_start = n_pid * min_each_n_jobs + MIN(n_pid, rest_n_jobs);
        const uint64_t my_n_idx_end = (n_pid + 1) * min_each_n_jobs + MIN(n_pid + 1, rest_n_jobs);
        const uint64_t my_n_start = MIN(my_n_idx_start * NR, n);
        const uint64_t my_n_end = MIN(my_n_idx_end * NR, n);
        const uint64_t my_n_size = my_n_end - my_n_start;

        const double *A_start = A + my_m_start * lda;
        const double *B_start = B + my_n_start * 1;
        double *C_start = C + my_m_start * ldc + my_n_start * 1;

        static double *_A[TOTAL_CORE] = {
            NULL,
        };
        static double *_B[TOTAL_CORE] = {
            NULL,
        };

        if (_A[pid] == NULL)
        {
            _A[pid] = numa_alloc_onnode(sizeof(double) * (MB + MR) * KB, nid);
            _B[pid] = numa_alloc_onnode(sizeof(double) * KB * (NB + NR), nid);
        }

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

        PTHREAD_CALL(pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &mask));
        PTHREAD_CALL(pthread_create(&tinfo[pid].thread_id, &attr, &middle_kernel, &tinfo[pid]));
    }

    PTHREAD_CALL(pthread_attr_destroy(&attr));

    for (uint64_t pid = 0; pid < TOTAL_CORE; ++pid)
    {
        void *res;
        PTHREAD_CALL(pthread_join(tinfo[pid].thread_id, &res));
    }
}
