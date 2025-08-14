#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#include "cblas_format.h"

#define M 6000
#define N 6000
#define K 6000

#define lda K
#define ldb N
#define ldc N

void set_data(double *matrix, uint64_t size, uint64_t seed, double min_value,
              double max_value)
{
#pragma omp parallel
  {
    uint64_t tid = omp_get_thread_num();
    uint64_t value = (tid * 1034871 + 10581) * seed;
    uint64_t mul = 192499;
    uint64_t add = 6837199;
    for (uint64_t i = 0; i < 50 + tid; ++i)
      value = value * mul + add;
#pragma omp for
    for (uint64_t i = 0; i < size; ++i)
    {
      value = value * mul + add;
      matrix[i] = (double)value / (double)(uint64_t)(-1) * (max_value - min_value) + min_value;
    }
  }
}

void memcpy_parallel(void *restrict dest, const void *restrict src, size_t n)
{
#pragma omp parallel
  {
    uint64_t tid = omp_get_thread_num();
    uint64_t num_threads = omp_get_num_threads();
    size_t chunk_size = n / num_threads;
    size_t start = tid * chunk_size;
    size_t end = (tid == num_threads - 1) ? n : start + chunk_size;

    memcpy((char *)dest + start, (const char *)src + start, end - start);
  }
}

extern void initialize_blocks();
extern void finalize_blocks();

void initialize(void **arg_in, void **arg_out, void **arg_val)
{
  void **arr = malloc(sizeof(void *) * 3);
  *arg_in = arr;

  arr[0] = malloc(M * K * sizeof(double));
  arr[1] = malloc(K * N * sizeof(double));
  arr[2] = malloc(M * N * sizeof(double));

  *arg_out = malloc(M * N * sizeof(double));

  initialize_blocks();

  set_data(arr[0], M * K, 100, -1.0, 1.0);
  set_data(arr[1], K * N, 200, -1.0, 1.0);
  set_data(arr[2], M * N, 300, -1.0, 1.0);

  if (arg_val != NULL)
  {
    *arg_val = malloc(M * N * sizeof(double));
    memcpy_parallel(*arg_val, arr[2], sizeof(double) * M * N);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, arr[0], lda, arr[1], ldb, 1.0, *arg_val, ldc);
  }
}

void finalize(void *arg_in, void *arg_out, void *arg_val)
{
  void **arr = (void **)arg_in;

  free(arr[0]);
  free(arr[1]);
  free(arr[2]);

  free(arr);

  free(arg_out);

  finalize_blocks();

  if (arg_val != NULL)
  {
    free(arg_val);
  }
}

double evaluate(void *arg_in, void *arg_out)
{
  void **arr = (void **)arg_in;

  double *A = (double *)arr[0];
  double *B = (double *)arr[1];
  double *C = (double *)arr[2];

  memcpy_parallel(arg_out, C, sizeof(double) * M * N);

  const double start_time = omp_get_wtime();
  call_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, A, lda, B, ldb, 1.0, arg_out, ldc);
  const double end_time = omp_get_wtime();

  return end_time - start_time;
}

bool validate(const void *arg_val, const void *arg_out)
{
  double *temp = malloc(sizeof(double) * M * N);
  memcpy_parallel(temp, arg_val, sizeof(double) * M * N);

  cblas_daxpy(M * N, -1.0, arg_out, 1, temp, 1);
  double difference = cblas_dnrm2(M * N, temp, 1);
  free(temp);

  return difference < 0.0001;
}
