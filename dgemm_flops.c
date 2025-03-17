#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <armpl.h>
#include <omp.h>

#ifndef KERNEL
#define KERNEL "UNKNOWN"
#endif

#ifdef VERIFY
#endif

//#define NO_PRINT

void call_dgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                CBLAS_TRANSPOSE TransB, int64_t m, int64_t n, int64_t k,
                double alpha, const double *A, int64_t lda, const double *B,
                int64_t ldb, double beta, double *C, int64_t ldc);
void set_data(double *matrix, uint64_t size, uint64_t seed, double min_value,
              double max_value);

int main(int argc, char **argv) {
    if (argc != 11) {
        fprintf(stderr,
                "Usage : %s NumThreads Layout(Row/Col) TransA(T/N) TransB(T/N) M N K"
                " alpha beta iter\n", argv[0]);
        return 1;
    }

    const uint64_t        nt     = strtol(argv[1], NULL, 10);
    const CBLAS_LAYOUT    layout = (argv[2][0] == 'R') ? CblasRowMajor : CblasColMajor;
    const CBLAS_TRANSPOSE TransA = (argv[3][0] == 'T') ? CblasTrans    : CblasNoTrans ;
    const CBLAS_TRANSPOSE TransB = (argv[4][0] == 'T') ? CblasTrans    : CblasNoTrans ;
    const uint64_t        m      = strtol(argv[5], NULL, 10);
    const uint64_t        n      = strtol(argv[6], NULL, 10);
    const uint64_t        k      = strtol(argv[7], NULL, 10);
    const double          alpha  = strtod(argv[8], NULL);
    const double          beta   = strtod(argv[9], NULL);
    const uint64_t        iter   = strtol(argv[10], NULL, 10);

    const uint64_t lda = (TransA == CblasTrans) != (layout == CblasRowMajor) ? k : m;
    const uint64_t ldb = (TransB == CblasTrans) != (layout == CblasRowMajor) ? n : k;
    const uint64_t ldc = layout == CblasRowMajor ? n : m;

#ifndef NO_PRINT
    printf("-------------------\n");
    printf("Kernel:  %s\n", KERNEL);
    printf("-------------------\n");
    printf("Layout:  %s\n", (layout == CblasRowMajor) ? "Row" : "Column");
    printf("TransA:  %s\n", (TransA == CblasTrans) ? "Yes" : "No");
    printf("TransB:  %s\n", (TransB == CblasTrans) ? "Yes" : "No");
    printf("M:       %lu\n", m);
    printf("N:       %lu\n", n);
    printf("K:       %lu\n", k);
    printf("alpha:   %.3lf\n", alpha);
    printf("beta:    %.3lf\n", beta);
    printf("-------------------\n");
#endif

    double *A;
    double *B;
    double *C;

    A = malloc(sizeof(double) * m * k);
    B = malloc(sizeof(double) * k * n);
    C = malloc(sizeof(double) * m * n);
    omp_set_num_threads(nt);

    set_data(A, m * k, 100, 0.0, 2.0);
    set_data(B, k * n, 200, 0.0, 2.0);
    set_data(C, m * n, 300, 0.0, 2.0);

#ifdef VERIFY
    double *D;
    D = malloc(sizeof(double) * m * n);
    set_data(D, m * n, 300, 0.0, 2.0);
#endif

#ifndef NO_PRINT
    printf("case    duration(sec)    gflops(GFLOPS)\n");
#endif

    double min_duration = 1e10;
    double max_gflops = 0;
    for (uint64_t i = 0; i < iter; ++i) {
        const double start_time = omp_get_wtime();
        call_dgemm(layout, TransA, TransB, m, n, k, alpha, A, lda, B, ldb, beta,
                   C, ldc);
        const double end_time = omp_get_wtime();
        const double duration = end_time - start_time;
        const double gflops = 2.0e-9 * m * n * k / duration;

		if (min_duration > duration) {
        	min_duration = duration;
			max_gflops = gflops;
		}

#ifndef NO_PRINT
        printf("%4lu)%17.3lf%18.3lf\n", i + 1, duration, gflops);
#endif

#ifdef VERIFY
        if (i == 0) {
            cblas_dgemm(layout, TransA, TransB, m, n, k, alpha, A, lda, B, ldb,
                        beta, D, ldc);
            cblas_daxpy(m * n, -1.0, C, 1, D, 1);
            double difference = cblas_dnrm2(m * n, D, 1);
            if (difference > 0.0001) {
                printf("WRONG RESULT\n");
                printf("---------------------------------------\n");
                return 1;
            }
        }
#endif
    }

#ifndef NO_PRINT
    if (iter > 1) {
        printf("---------------------------------------\n");
        printf("best)%17.3lf%18.3lf\n", min_duration, max_gflops);
    }
    printf("---------------------------------------\n");
#else
    printf("%17.3lf%18.3lf\n", min_duration, max_gflops);
#endif

    free(A);
    free(B);
    free(C);
#ifdef VERIFY
    free(D);
#endif

    return 0;
}

void set_data(double *matrix, uint64_t size, uint64_t seed, double min_value,
              double max_value) {
#pragma omp parallel
    {
        uint64_t tid = omp_get_thread_num();
        uint64_t value = (tid * 1034871 + 10581) * seed;
        uint64_t mul = 192499;
        uint64_t add = 6837199;
        for (uint64_t i = 0; i < 50 + tid; ++i)
            value = value * mul + add;
#pragma omp for
        for (uint64_t i = 0; i < size; ++i) {
            value = value * mul + add;
            matrix[i] = (double)value / (double)(uint64_t)(-1) * (max_value - min_value)
                        + min_value;
        }
    }
}
