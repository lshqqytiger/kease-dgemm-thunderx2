#pragma once

#include <armpl.h>
#include <stdint.h>

void call_dgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                CBLAS_TRANSPOSE TransB, const int64_t m, const int64_t n,
                const int64_t k, const double alpha, const double *A,
                const int64_t lda, const double *B, const int64_t ldb,
                const double beta, double *C, const int64_t ldc);
