/**
 * @file cblas.c
 * @author Enoch Jung
 * @brief adaptor. call_dgemm -> cblas_dgemm
 * @date 2023-08-16
 */

#include <stdint.h>
#include <armpl.h>
#ifdef OC
#include <omp.h>
#endif

void call_dgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                CBLAS_TRANSPOSE TransB, int64_t m, int64_t n, int64_t k,
                double alpha, const double *A, int64_t lda, const double *B,
                int64_t ldb, double beta, double *C, int64_t ldc)
{
#ifdef OC
    omp_set_num_threads(1);
#endif
    cblas_dgemm(layout, TransA, TransB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
