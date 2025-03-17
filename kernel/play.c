/**
 * @file play.c
 * @author Enoch Jung
 * @date 2023-10-10
 */

#define _GNU_SOURCE

#include <assert.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include <arm_neon.h>
#include <armpl.h>
#include <numa.h>
#include <omp.h>
#include <sched.h>

#define PAGE_SIZE 4096
#define CACHE_LINE 64
#define CACHE_ELEM (CACHE_LINE / 8)

#define UNLIKELY(expr) __builtin_expect(!!(expr), 0)
#define LIKELY(expr)   __builtin_expect(!!(expr), 1)

#define MR 4
#define NR 8

#ifndef KB
#define KB 200
#endif

static void micro_kernel (
    uint64_t kk
) {
#pragma unroll(4)
    for (uint64_t i = 0; LIKELY(i < kk); ++i) {
        asm volatile(
                /*
        " fmla   v0.2d,  v0.2d,  v0.2d \t\n"
        " fmla   v1.2d,  v1.2d,  v1.2d \t\n"
        " fmla   v2.2d,  v2.2d,  v2.2d \t\n"
        " fmla   v3.2d,  v3.2d,  v3.2d \t\n"
        " fmla   v4.2d,  v4.2d,  v4.2d \t\n"
        " fmla   v5.2d,  v5.2d,  v5.2d \t\n"
        " fmla   v6.2d,  v6.2d,  v6.2d \t\n"
        " fmla   v7.2d,  v7.2d,  v7.2d \t\n"
        " fmla   v8.2d,  v8.2d,  v8.2d \t\n"
        " fmla   v9.2d,  v9.2d,  v9.2d \t\n"
        " fmla  v10.2d, v10.2d, v10.2d \t\n"
        " fmla  v11.2d, v11.2d, v11.2d \t\n"
        " fmla  v12.2d, v12.2d, v12.2d \t\n"
        " fmla  v13.2d, v13.2d, v13.2d \t\n"
        " fmla  v14.2d, v14.2d, v14.2d \t\n"
        " fmla  v15.2d, v15.2d, v15.2d \t\n"
        */

		" fmla   v0.2d, v16.2d, v24.d[0] \t\n"
		" fmla   v1.2d, v17.2d, v24.d[0] \t\n"
		" fmla   v2.2d, v18.2d, v24.d[0] \t\n"
		" fmla   v3.2d, v19.2d, v24.d[0] \t\n"
		" fmla   v4.2d, v16.2d, v24.d[1] \t\n"
		" fmla   v5.2d, v17.2d, v24.d[1] \t\n"
		" fmla   v6.2d, v18.2d, v24.d[1] \t\n"
		" fmla   v7.2d, v19.2d, v24.d[1] \t\n"
		" fmla   v8.2d, v16.2d, v25.d[0] \t\n"
		" fmla   v9.2d, v17.2d, v25.d[0] \t\n"
		" fmla  v10.2d, v18.2d, v25.d[0] \t\n"
		" fmla  v11.2d, v19.2d, v25.d[0] \t\n"
		" fmla  v12.2d, v16.2d, v25.d[1] \t\n"
		" fmla  v13.2d, v17.2d, v25.d[1] \t\n"
		" fmla  v14.2d, v18.2d, v25.d[1] \t\n"
		" fmla  v15.2d, v19.2d, v25.d[1] \t\n"
        :
        :
        : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
        );
    }
}

void call_dgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                CBLAS_TRANSPOSE TransB, int64_t m, int64_t n, int64_t k,
                double alpha, const double *A, int64_t lda, const double *B,
                int64_t ldb, double beta, double *C, int64_t ldc) {
    const uint64_t is_C_row = (layout == CblasRowMajor ? 1         : 0       );
    const uint64_t is_A_row = (TransA == CblasTrans    ? !is_C_row : is_C_row);
    const uint64_t is_B_row = (TransB == CblasTrans    ? !is_C_row : is_C_row);

    assert(is_A_row == 1);
    assert(is_B_row == 1);
    assert(is_C_row == 1);
    assert(alpha == 1.0);
    assert(beta == 1.0);

    const uint64_t total_m_jobs = (m + MR - 1) / MR;
    const uint64_t total_n_jobs = (n + NR - 1) / NR;
    const uint64_t total_k_jobs = (k + KB - 1) / KB;

    const uint64_t total_jobs = total_m_jobs * total_n_jobs * total_k_jobs;

#pragma omp parallel for
    for (uint64_t i = 0; i < total_jobs; ++i) {
        micro_kernel(KB);
    }
}
