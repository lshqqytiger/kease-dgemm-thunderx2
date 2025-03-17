/**
 * @file kernel.06.mc.c
 * @author Enoch Jung
 * @brief dgemm for
          - core  : 64 cores,
          - A     : RowMajor
          - B     : RowMajor
          - C     : RowMajor
          - k     : even number
          - alpha : 1.0
          - beta  : 1.0
 * @date 2023-10-16
 */

#define _GNU_SOURCE

#include <errno.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#include <arm_neon.h>
#include <armpl.h>
#include <numa.h>
#include <pthread.h>
#include <omp.h>
#include <sched.h>

#define HANDLE_ERROR(expr) \
        do { \
            int err = (expr); \
            if (err == 0) { break; } \
            perror(#expr); \
            exit(EXIT_FAILURE); \
        } while (0)

#define PAGE_SIZE 4096
#define CACHE_LINE 64
#define CACHE_ELEM (CACHE_LINE / 8)

#define UNLIKELY(expr) __builtin_expect(!!(expr), 0)
#define LIKELY(expr)   __builtin_expect(!!(expr), 1)

#define TOTAL_CORE 64

#define NUMA_NODE 2

#ifndef CM
#define CM 8
#endif

#define CN (TOTAL_CORE / CM)

#define MR 4
#define NR 8

#ifndef MB
#define MB (MR * 9)
#endif

#ifndef NB
#define NB (NR * 42)
#endif

#ifndef KB
#define KB 280
#endif

void micro_kernel (
    const uint64_t kk,
    const double *restrict _A,
    const double *restrict _B,
    double *restrict C,
    const uint64_t ldc,
    const double *restrict A_next,
    const double *restrict B_next
) {
    register double* C0 = C + 0 * ldc;
    register double* C1 = C + 1 * ldc;
    register double* C2 = C + 2 * ldc;
    register double* C3 = C + 3 * ldc;

    register float64x2_t  v0 asm( "v0");
    register float64x2_t  v1 asm( "v1");
    register float64x2_t  v2 asm( "v2");
    register float64x2_t  v3 asm( "v3");
    register float64x2_t  v4 asm( "v4");
    register float64x2_t  v5 asm( "v5");
    register float64x2_t  v6 asm( "v6");
    register float64x2_t  v7 asm( "v7");
    register float64x2_t  v8 asm( "v8");
    register float64x2_t  v9 asm( "v9");
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

    __builtin_prefetch(C0, 0, 3);
    __builtin_prefetch(C1, 0, 3);
    __builtin_prefetch(C2, 0, 3);
    __builtin_prefetch(C3, 0, 3);

    asm volatile(
    " ldr      q16, [%[B]]      \t\n"
    " ldr      q17, [%[B], #16] \t\n"
    " ldr      q18, [%[B], #32] \t\n"
    " ldr      q19, [%[B], #48] \t\n"
    " add     %[B],  %[B], #64  \t\n"
    " movi    v0.2d,  #0        \t\n"
    " movi    v1.2d,  #0        \t\n"
    " movi    v2.2d,  #0        \t\n"
    " movi    v3.2d,  #0        \t\n"
    " movi    v4.2d,  #0        \t\n"
    " movi    v5.2d,  #0        \t\n"
    " ldr      q24, [%[A]]      \t\n"
    " movi    v6.2d,  #0        \t\n"
    " movi    v7.2d,  #0        \t\n"
    " movi    v8.2d,  #0        \t\n"
    " movi    v9.2d,  #0        \t\n"
    " movi   v10.2d,  #0        \t\n"
    " movi   v11.2d,  #0        \t\n"
    " ldr      q25, [%[A], #16] \t\n"
    " add     %[A],  %[A], #32  \t\n"
    " movi   v12.2d,  #0        \t\n"
    " movi   v13.2d,  #0        \t\n"
    " movi   v14.2d,  #0        \t\n"
    " movi   v15.2d,  #0        \t\n"
    : [A]"+r"(_A), [B]"+r"(_B),
      [ v0]"=w"( v0), [ v1]"=w"( v1), [ v2]"=w"( v2), [ v3]"=w"( v3), [ v4]"=w"( v4), [ v5]"=w"( v5), [ v6]"=w"( v6), [ v7]"=w"( v7), [ v8]"=w"( v8), [ v9]"=w"( v9), 
      [v10]"=w"(v10), [v11]"=w"(v11), [v12]"=w"(v12), [v13]"=w"(v13), [v14]"=w"(v14), [v15]"=w"(v15), [v16]"=w"(v16), [v17]"=w"(v17), [v18]"=w"(v18), [v19]"=w"(v19), 
                                                                      [v24]"=w"(v24), [v25]"=w"(v25)
    );

    uint64_t iter = kk >> 1;
#pragma unroll(2)
    for (uint64_t i = 0; LIKELY(i < iter); ++i) {
        __builtin_prefetch(_A + (MR * 2) * 5, 0, 0);
        __builtin_prefetch(_B + (NR * 2) * 5, 0, 3);
        __builtin_prefetch(_B + (NR * 2) * 5 + 8, 0, 3);

        asm volatile(
        " ldr    q20, [%[B]] \t\n"
        " ldr    q21, [%[B], #16] \t\n"
        " ldr    q22, [%[B], #32] \t\n"
        " ldr    q23, [%[B], #48] \t\n"
        " add    %[B], %[B], #64  \t\n"
        " fmla   v0.2d, v16.2d, v24.d[0] \t\n"
        " fmla   v1.2d, v17.2d, v24.d[0] \t\n"
        " fmla   v2.2d, v18.2d, v24.d[0] \t\n"
        " fmla   v3.2d, v19.2d, v24.d[0] \t\n"

        " ldr    q26, [%[A]] \t\n"
        " fmla   v4.2d, v16.2d, v24.d[1] \t\n"
        " fmla   v5.2d, v17.2d, v24.d[1] \t\n"
        " fmla   v6.2d, v18.2d, v24.d[1] \t\n"
        " fmla   v7.2d, v19.2d, v24.d[1] \t\n"

        " ldr    q27, [%[A], #16] \t\n"
        " fmla   v8.2d, v16.2d, v25.d[0] \t\n"
        " fmla   v9.2d, v17.2d, v25.d[0] \t\n"
        " fmla  v10.2d, v18.2d, v25.d[0] \t\n"
        " fmla  v11.2d, v19.2d, v25.d[0] \t\n"
        " fmla  v12.2d, v16.2d, v25.d[1] \t\n"
        " fmla  v13.2d, v17.2d, v25.d[1] \t\n"
        " fmla  v14.2d, v18.2d, v25.d[1] \t\n"
        " fmla  v15.2d, v19.2d, v25.d[1] \t\n"

        " ldr    q16, [%[B]] \t\n"
        " ldr    q17, [%[B], #16] \t\n"
        " ldr    q18, [%[B], #32] \t\n"
        " ldr    q19, [%[B], #48] \t\n"
        " add    %[B], %[B], #64  \t\n"
        " fmla   v0.2d, v20.2d, v26.d[0] \t\n"
        " fmla   v1.2d, v21.2d, v26.d[0] \t\n"
        " fmla   v2.2d, v22.2d, v26.d[0] \t\n"
        " fmla   v3.2d, v23.2d, v26.d[0] \t\n"

        " ldr    q24, [%[A], #32] \t\n"
        " fmla   v4.2d, v20.2d, v26.d[1] \t\n"
        " fmla   v5.2d, v21.2d, v26.d[1] \t\n"
        " fmla   v6.2d, v22.2d, v26.d[1] \t\n"
        " fmla   v7.2d, v23.2d, v26.d[1] \t\n"

        " ldr    q25, [%[A], #48] \t\n"
        " add    %[A], %[A], #64  \t\n"
        " fmla   v8.2d, v20.2d, v27.d[0] \t\n"
        " fmla   v9.2d, v21.2d, v27.d[0] \t\n"
        " fmla  v10.2d, v22.2d, v27.d[0] \t\n"
        " fmla  v11.2d, v23.2d, v27.d[0] \t\n"
        " fmla  v12.2d, v20.2d, v27.d[1] \t\n"
        " fmla  v13.2d, v21.2d, v27.d[1] \t\n"
        " fmla  v14.2d, v22.2d, v27.d[1] \t\n"
        " fmla  v15.2d, v23.2d, v27.d[1] \t\n"

        : [A]"+r"(_A), [B]"+r"(_B),
          [ v0]"+w"( v0), [ v1]"+w"( v1), [ v2]"+w"( v2), [ v3]"+w"( v3), [ v4]"+w"( v4), [ v5]"+w"( v5), [ v6]"+w"( v6), [ v7]"+w"( v7), [ v8]"+w"( v8), [ v9]"+w"( v9), 
          [v10]"+w"(v10), [v11]"+w"(v11), [v12]"+w"(v12), [v13]"+w"(v13), [v14]"+w"(v14), [v15]"+w"(v15), [v16]"+w"(v16), [v17]"+w"(v17), [v18]"+w"(v18), [v19]"+w"(v19), 
          [v20]"+w"(v20), [v21]"+w"(v21), [v22]"+w"(v22), [v23]"+w"(v23), [v24]"+w"(v24), [v25]"+w"(v25), [v26]"+w"(v26), [v27]"+w"(v27)
        );
    }

//    __builtin_prefetch(A_next + 0x00, 0, 3);
//    __builtin_prefetch(B_next + 0x00, 0, 3);

    asm volatile(
    " ldr      q16, [%[C0]]          \t\n"
    " ldr      q17, [%[C0],#16]      \t\n"
    " ldr      q18, [%[C0],#32]      \t\n"
    " ldr      q19, [%[C0],#48]      \t\n"
    " ldr      q20, [%[C1]]          \t\n"
    " ldr      q21, [%[C1],#16]      \t\n"
    " ldr      q22, [%[C1],#32]      \t\n"
    " ldr      q23, [%[C1],#48]      \t\n"
    " ldr      q24, [%[C2]]          \t\n"
    " ldr      q25, [%[C2],#16]      \t\n"
    " ldr      q26, [%[C2],#32]      \t\n"
    " ldr      q27, [%[C2],#48]      \t\n"
    " ldr      q28, [%[C3]]          \t\n"
    " ldr      q29, [%[C3],#16]      \t\n"
    " ldr      q30, [%[C3],#32]      \t\n"
    " ldr      q31, [%[C3],#48]      \t\n"

    " fadd  v16.2d, v16.2d,  v0.2d   \t\n"
    " fadd  v17.2d, v17.2d,  v1.2d   \t\n"
    " fadd  v18.2d, v18.2d,  v2.2d   \t\n"
    " fadd  v19.2d, v19.2d,  v3.2d   \t\n"
    " str   q16, [%[C0]] \t\n"
    " str   q17, [%[C0], #16] \t\n"
    " str   q18, [%[C0], #32] \t\n"
    " str   q19, [%[C0], #48] \t\n"

    " fadd  v20.2d, v20.2d,  v4.2d   \t\n"
    " fadd  v21.2d, v21.2d,  v5.2d   \t\n"
    " fadd  v22.2d, v22.2d,  v6.2d   \t\n"
    " fadd  v23.2d, v23.2d,  v7.2d   \t\n"
    " str   q20, [%[C1]] \t\n"
    " str   q21, [%[C1], #16] \t\n"
    " str   q22, [%[C1], #32] \t\n"
    " str   q23, [%[C1], #48] \t\n"

    " fadd  v24.2d, v24.2d,  v8.2d   \t\n"
    " fadd  v25.2d, v25.2d,  v9.2d   \t\n"
    " fadd  v26.2d, v26.2d, v10.2d   \t\n"
    " fadd  v27.2d, v27.2d, v11.2d   \t\n"
    " str   q24, [%[C2]] \t\n"
    " str   q25, [%[C2], #16] \t\n"
    " str   q26, [%[C2], #32] \t\n"
    " str   q27, [%[C2], #48] \t\n"

    " fadd  v28.2d, v28.2d, v12.2d   \t\n"
    " fadd  v29.2d, v29.2d, v13.2d   \t\n"
    " fadd  v30.2d, v30.2d, v14.2d   \t\n"
    " fadd  v31.2d, v31.2d, v15.2d   \t\n"
    " str   q28, [%[C3]] \t\n"
    " str   q29, [%[C3], #16] \t\n"
    " str   q30, [%[C3], #32] \t\n"
    " str   q31, [%[C3], #48] \t\n"
    ""
    : [ v0]"+w"( v0), [ v1]"+w"( v1), [ v2]"+w"( v2), [ v3]"+w"( v3), [ v4]"+w"( v4), [ v5]"+w"( v5), [ v6]"+w"( v6), [ v7]"+w"( v7), [ v8]"+w"( v8), [ v9]"+w"( v9), 
      [v10]"+w"(v10), [v11]"+w"(v11), [v12]"+w"(v12), [v13]"+w"(v13), [v14]"+w"(v14), [v15]"+w"(v15), [v16]"+w"(v16), [v17]"+w"(v17), [v18]"+w"(v18), [v19]"+w"(v19), 
      [v20]"+w"(v20), [v21]"+w"(v21), [v22]"+w"(v22), [v23]"+w"(v23), [v24]"+w"(v24), [v25]"+w"(v25), [v26]"+w"(v26), [v27]"+w"(v27), [v28]"+w"(v28), [v29]"+w"(v29),
      [v30]"+w"(v30), [v31]"+w"(v31)
    : [C0]"r"(C0),
      [C1]"r"(C1),
      [C2]"r"(C2),
      [C3]"r"(C3)
    );
}

void micro_dxpy(
    const uint64_t m,
    const uint64_t n,
    double *restrict C,
    const uint64_t ldc,
    const double *restrict _C
) {
    for (uint64_t i = 0; i < m; ++i) {
        for (uint64_t j = 0; j < n; ++j) {
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
    const double *restrict _B,
    double *restrict C,
    const uint64_t ldc
) {
    const register uint64_t mmc = (mm + MR - 1) / MR;
    register uint64_t mmr = mm % MR;
    if (mmr == 0) {
        mmr = MR;
    }

    const register uint64_t nnc = (nn + NR - 1) / NR;
    register uint64_t nnr = nn % NR;
    if (nnr == 0) {
        nnr = NR;
    }

    const register double *A_now;
    const register double *B_now;
    const double *A_next;
    const double *B_next;

    A_next = _A;
    B_next = _B;
    for (uint64_t nni = 0; LIKELY(nni < nnc); ++nni) {
        const register uint64_t nnn = nni != nnc - 1 ? NR : nnr;

        B_now = B_next;
        B_next = nni != nnc - 1 ? B_next + NR * kk : _B;

        for (uint64_t mmi = 0; LIKELY(mmi < mmc); ++mmi) {
            const register uint64_t mmm = mmi != mmc - 1 ? MR : mmr;

            A_now = A_next;
            A_next = mmi != mmc - 1 ? A_next + MR * kk : _A;

            if (LIKELY(mmm == MR && nnn == NR)) {
                micro_kernel(kk, A_now, B_now, C + mmi * MR * ldc + nni * NR, ldc, A_next, B_next);
            } else {
                double _C[MR * NR] __attribute__((aligned(CACHE_LINE))) = {};
                micro_kernel(kk, A_now, B_now, _C, NR, A_next, B_next);
                micro_dxpy(mmr, nnr, C + mmi * MR * ldc + nni * NR, ldc, _C);
            }
        }
    }
}

void pack_arc(
    const uint64_t mm,
    const uint64_t kk,
    const double *restrict A,
    const uint64_t lda,
    double *restrict _A
) {
    const register uint64_t mmc = (mm + MR - 1) / MR;
    register uint64_t mmr = mm % MR;
    if (mmr == 0) {
        mmr = MR;
    }

    const register uint64_t kkc = (kk + CACHE_ELEM - 1) / CACHE_ELEM;
    register uint64_t kkr = kk % CACHE_ELEM;
    if (kkr == 0) {
        kkr = CACHE_ELEM;
    }

    const register double* A_now = A;
    const double* A_m_next = A;

    for (uint64_t mmi = 0; LIKELY(mmi < mmc); ++mmi) {
        const register uint64_t mmm = mmi != mmc - 1 ? MR : mmr;

        A_now = A_m_next;
        A_m_next = A_m_next + MR * lda;

        __builtin_prefetch(A_m_next + lda * 0 + CACHE_ELEM * 0, 0, 0);
        __builtin_prefetch(A_m_next + lda * 1 + CACHE_ELEM * 0, 0, 0);
        __builtin_prefetch(A_m_next + lda * 2 + CACHE_ELEM * 0, 0, 0);
        __builtin_prefetch(A_m_next + lda * 3 + CACHE_ELEM * 0, 0, 0);
        __builtin_prefetch(A_m_next + lda * 0 + CACHE_ELEM * 1, 0, 0);
        __builtin_prefetch(A_m_next + lda * 1 + CACHE_ELEM * 1, 0, 0);
        __builtin_prefetch(A_m_next + lda * 2 + CACHE_ELEM * 1, 0, 0);
        __builtin_prefetch(A_m_next + lda * 3 + CACHE_ELEM * 1, 0, 0);
        __builtin_prefetch(A_m_next + lda * 0 + CACHE_ELEM * 2, 0, 0);
        __builtin_prefetch(A_m_next + lda * 1 + CACHE_ELEM * 2, 0, 0);
        __builtin_prefetch(A_m_next + lda * 2 + CACHE_ELEM * 2, 0, 0);
        __builtin_prefetch(A_m_next + lda * 3 + CACHE_ELEM * 2, 0, 0);
        __builtin_prefetch(A_m_next + lda * 0 + CACHE_ELEM * 3, 0, 0);
        __builtin_prefetch(A_m_next + lda * 1 + CACHE_ELEM * 3, 0, 0);
        __builtin_prefetch(A_m_next + lda * 2 + CACHE_ELEM * 3, 0, 0);
        __builtin_prefetch(A_m_next + lda * 3 + CACHE_ELEM * 3, 0, 0);

        for (uint64_t kki = 0; LIKELY(kki < kkc); ++kki) {
            const register uint64_t kkk = kki != kkc - 1 ? CACHE_ELEM : kkr;

            double* _A_now = _A + mmi * MR * kk + kki * MR * CACHE_ELEM;

            if (mmm == MR && kkk == CACHE_ELEM) {
                const register double* A0 = A_now + lda * 0;
                const register double* A1 = A_now + lda * 1;
                const register double* A2 = A_now + lda * 2;
                const register double* A3 = A_now + lda * 3;
                asm volatile(
                " ld1   {v0.2d},         [%[A0]], #16  \t\n"
                " ld1   {v4.2d},         [%[A0]], #16  \t\n"
                " ld1   {v8.2d},         [%[A0]], #16  \t\n"
                " ld1   {v12.2d},        [%[A0]], #16  \t\n"
                " ld1   {v1.2d},         [%[A1]], #16  \t\n"
                " ld1   {v5.2d},         [%[A1]], #16  \t\n"
                " ld1   {v9.2d},         [%[A1]], #16  \t\n"
                " ld1   {v13.2d},        [%[A1]], #16  \t\n"
                " ld1   {v2.2d},         [%[A2]], #16  \t\n"
                " ld1   {v6.2d},         [%[A2]], #16  \t\n"
                " ld1   {v10.2d},        [%[A2]], #16  \t\n"
                " ld1   {v14.2d},        [%[A2]], #16  \t\n"
                " ld1   {v3.2d},         [%[A3]], #16  \t\n"
                " ld1   {v7.2d},         [%[A3]], #16  \t\n"
                " ld1   {v11.2d},        [%[A3]], #16  \t\n"
                " ld1   {v15.2d},        [%[A3]], #16  \t\n"

                " st4   {v0.2d-v3.2d},   [%[_A]], #64  \t\n"
                " st4   {v4.2d-v7.2d},   [%[_A]], #64  \t\n"
                " st4   {v8.2d-v11.2d},  [%[_A]], #64  \t\n"
                " st4   {v12.2d-v15.2d}, [%[_A]]  \t\n"

                : [A0]"+r"(A0),
                  [A1]"+r"(A1),
                  [A2]"+r"(A2),
                  [A3]"+r"(A3),
                  [_A]"+r"(_A_now)
                );
            }
            else {
                for (uint64_t mmmi = 0; mmmi < mmm; ++mmmi) {
                    for (uint64_t kkki = 0; kkki < kkk; ++kkki) {
                        _A_now[mmmi + kkki * MR] = A_now[mmmi * lda + kkki];
                    }
                }
            }

            A_now += CACHE_ELEM;
        }
    }
}



void pack_brr(
    const uint64_t kk,
    const uint64_t nn,
    const double *restrict B,
    const uint64_t ldb,
    double *restrict _B
) {
    const register double* B_base = B;
    register double* _B_base = _B;
    const uint64_t _B_inc = kk * NR;

    const uint64_t nnc = (nn + NR - 1) / NR;
    if (nnc == NB / NR) {
        for (uint64_t j = 0; j < kk; ++j) {
            B = B_base;
            _B = _B_base;
#pragma unroll(NB / NR)
            for (uint64_t nni = 0; nni < NB / NR; ++nni) {
                register float64x2x4_t vec;
                vec = vld1q_f64_x4(B);
                vst1q_f64_x4(_B, vec);

                B += NR;
                _B += _B_inc;
            }
            B_base += ldb;
            _B_base += NR;
        }
    }
    else {
        for (uint64_t j = 0; j < kk; ++j) {
            B = B_base;
            _B = _B_base;
#pragma unroll(NB / NR)
            for (uint64_t nni = 0; nni < nnc; ++nni) {
                register float64x2x4_t vec;
                vec = vld1q_f64_x4(B);
                vst1q_f64_x4(_B, vec);

                B += NR;
                _B += _B_inc;
            }
            B_base += ldb;
            _B_base += NR;
        }
    }
}

struct thread_info {
    pthread_t thread_id;
    uint64_t nid;
    uint64_t m;
    uint64_t n;
    uint64_t k;
    const double* A;
    uint64_t lda;
    const double* B;
    uint64_t ldb;
    double *restrict C;
    uint64_t ldc;
};

static void* middle_kernel(
    void *arg
) {
    struct thread_info *tinfo = arg;
    const uint64_t nid = tinfo->nid;
    const uint64_t m = tinfo->m;
    const uint64_t n = tinfo->n;
    const uint64_t k = tinfo->k;
    const double* A = tinfo->A;
    const uint64_t lda = tinfo->lda;
    const double* B = tinfo->B;
    const uint64_t ldb = tinfo->ldb;
    double* C = tinfo->C;
    const uint64_t ldc = tinfo->ldc;

    double* _A;
    double* _B;

    //const double start_time = omp_get_wtime();

    _A = numa_alloc_onnode(sizeof(double) * (MB + MR) * KB, nid); //posix_memalign((void**)&_A, PAGE_SIZE, sizeof(double) * (MB + MR) * KB);
    _B = numa_alloc_onnode(sizeof(double) * KB * (NB + NR), nid); //posix_memalign((void**)&_B, PAGE_SIZE, sizeof(double) * KB * (NB + NR));

    const uint64_t mc = (m + MB - 1) / MB;
    const uint64_t mr = m % MB;
    const uint64_t nc = (n + NB - 1) / NB;
    const uint64_t nr = n % NB;
    const uint64_t kc = (k + KB - 1) / KB;
    const uint64_t kr = k % KB;

    /*
    pack_brr(KB, NB, B, ldb, _B);
    pack_arc(MB, KB, A, lda, _A);
    for (uint64_t ni = 0; ni < nc; ++ni) {
        const uint64_t nn = (ni != nc - 1 || nr == 0) ? NB : nr;

        for (uint64_t ki = 0; ki < kc; ++ki) {
            const uint64_t kk = (ki != kc - 1 || kr == 0) ? KB : kr;

            for (uint64_t mi = 0; mi < mc; ++mi) {
                const uint64_t mm = (mi != mc - 1 || mr == 0) ? MB : mr;

                inner_kernel(mm, nn, kk, _A, _B, C + mi * MB * ldc + ni * NB, ldc);
            }
        }
    }
    */
    for (uint64_t ni = 0; ni < nc; ++ni) {
        const uint64_t nn = (ni != nc - 1 || nr == 0) ? NB : nr;

        for (uint64_t ki = 0; ki < kc; ++ki) {
            const uint64_t kk = (ki != kc - 1 || kr == 0) ? KB : kr;

            pack_brr(kk, nn, B + ki * KB * ldb + ni * NB, ldb, _B);

            for (uint64_t mi = 0; mi < mc; ++mi) {
                const uint64_t mm = (mi != mc - 1 || mr == 0) ? MB : mr;

                pack_arc(mm, kk, A + mi * MB * lda + ki * KB, lda, _A);

                inner_kernel(mm, nn, kk, _A, _B, C + mi * MB * ldc + ni * NB, ldc);
            }
        }
    }

    numa_free(_A, nid); // free(_A);
    numa_free(_B, nid); // free(_B);

    return NULL;

    /*const double end_time = omp_get_wtime();
    double *mem;
    mem = malloc(sizeof(double));
    *mem = (end_time - start_time);

    return (void*)mem;*/
}

#define MIN(a, b) ((a < b) ? (a) : (b))

void call_dgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                CBLAS_TRANSPOSE TransB, int64_t m, int64_t n, int64_t k,
                double alpha, const double *A, int64_t lda, const double *B,
                int64_t ldb, double beta, double *C, int64_t ldc) {
    const bool is_C_row = (layout == CblasRowMajor ? 1         : 0       );
    const bool is_A_row = (TransA == CblasTrans    ? !is_C_row : is_C_row);
    const bool is_B_row = (TransB == CblasTrans    ? !is_C_row : is_C_row);

    assert(is_A_row == 1);
    assert(is_B_row == 1);
    assert(is_C_row == 1);
    assert(alpha == 1.0);
    assert(beta == 1.0);

    const uint64_t total_m_jobs = (m + MR - 1) / MR;
    const uint64_t min_each_m_jobs = total_m_jobs / CM;
    const uint64_t rest_m_jobs = total_m_jobs % CM;

    const uint64_t total_n_jobs = (n + NR - 1) / NR;
    const uint64_t min_each_n_jobs = total_n_jobs / CN;
    const uint64_t rest_n_jobs = total_n_jobs % CN;

    struct thread_info tinfo[TOTAL_CORE];
    cpu_set_t mask;

    pthread_attr_t attr;
    pthread_attr_init(&attr);

    for (uint64_t pid = 0; pid < TOTAL_CORE; ++pid) {
        const uint64_t nid = pid / (CM * CN / NUMA_NODE);
        const uint64_t m_pid = pid % CM;
        const uint64_t n_pid = pid / CM;

        const uint64_t my_m_idx_start = (m_pid    ) * min_each_m_jobs + MIN(m_pid    , rest_m_jobs);
        const uint64_t my_m_idx_end   = (m_pid + 1) * min_each_m_jobs + MIN(m_pid + 1, rest_m_jobs);
        const uint64_t my_m_start     = MIN(my_m_idx_start * MR, m);
        const uint64_t my_m_end       = MIN(my_m_idx_end   * MR, m);
        const uint64_t my_m_size      = my_m_end - my_m_start;

        const uint64_t my_n_idx_start = (n_pid    ) * min_each_n_jobs + MIN(n_pid    , rest_n_jobs);
        const uint64_t my_n_idx_end   = (n_pid + 1) * min_each_n_jobs + MIN(n_pid + 1, rest_n_jobs);
        const uint64_t my_n_start     = MIN(my_n_idx_start * NR, n);
        const uint64_t my_n_end       = MIN(my_n_idx_end   * NR, n);
        const uint64_t my_n_size      = my_n_end - my_n_start;

        const double *A_start = A + my_m_start * lda                 ;
        const double *B_start = B +                    my_n_start * 1;
        double       *C_start = C + my_m_start * ldc + my_n_start * 1;

        tinfo[pid].nid = nid;
        tinfo[pid].m = my_m_size;
        tinfo[pid].n = my_n_size;
        tinfo[pid].k = k;
        tinfo[pid].A = A_start;
        tinfo[pid].lda = lda;
        tinfo[pid].B = B_start;
        tinfo[pid].ldb = ldb;
        tinfo[pid].C = C_start;
        tinfo[pid].ldc = ldc;

//        if (pid > 0) {
            CPU_ZERO(&mask);
            CPU_SET(pid, &mask);

            HANDLE_ERROR(pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &mask));
            HANDLE_ERROR(pthread_create(&tinfo[pid].thread_id, &attr, &middle_kernel, &tinfo[pid]));
            /*
        }
        else {
            middle_kernel((void*)&(tinfo[0]));
        }*/
    }

    HANDLE_ERROR(pthread_attr_destroy(&attr));

    //double t[64];
    for (uint64_t pid = 0; pid < TOTAL_CORE; ++pid) {
        void* res;
        HANDLE_ERROR(pthread_join(tinfo[pid].thread_id, &res));
        //t[pid] = *((double*)res);
        //free(res);
    }

    /*double mint = 9999;
    for (uint64_t pid = 0; pid < TOTAL_CORE; ++pid) {
        if (mint > t[pid])
            mint = t[pid];
    }*/
}
