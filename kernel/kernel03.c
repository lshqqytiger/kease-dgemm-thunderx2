#include <arm_neon.h>
#include <armpl.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>

#define PAGE_SIZE 4096
#define CACHE_LINE 64

#define MR 6
#define NR 8

#ifndef MB
#define MB 48
#endif

#ifndef NB
#define NB 456
#endif

#ifndef KB
#define KB 200
#endif

#define NT1 1
#define NT2 64

// #define DIST_L1_A (MR * 20)
// #define DIST_L1_B (NR * 20)

// assume alpha=1.0, beta=1.0
static void micro_kernel(uint64_t kk, const double *restrict _A,
                         const double *restrict _B, double *restrict C,
                         uint64_t ldc, const double *restrict A_next,
                         const double *restrict B_next) {
    float64x2x3_t va0;
    float64x2x4_t vb0;
    float64x2x4_t vc0, vc1, vc2, vc3, vc4, vc5;
    float64x2x4_t vC0, vC1, vC2, vC3, vC4, vC5;

    __builtin_prefetch(C + 0 * ldc, 0, 3);
    __builtin_prefetch(C + 1 * ldc, 0, 3);
    __builtin_prefetch(C + 2 * ldc, 0, 3);
    __builtin_prefetch(C + 3 * ldc, 0, 3);
    __builtin_prefetch(C + 4 * ldc, 0, 3);
    __builtin_prefetch(C + 5 * ldc, 0, 3);

    vc0.val[0] = vdupq_n_f64(0.0);
    vc0.val[1] = vdupq_n_f64(0.0);
    vc0.val[2] = vdupq_n_f64(0.0);
    vc0.val[3] = vdupq_n_f64(0.0);

    vc1.val[0] = vdupq_n_f64(0.0);
    vc1.val[1] = vdupq_n_f64(0.0);
    vc1.val[2] = vdupq_n_f64(0.0);
    vc1.val[3] = vdupq_n_f64(0.0);

    vc2.val[0] = vdupq_n_f64(0.0);
    vc2.val[1] = vdupq_n_f64(0.0);
    vc2.val[2] = vdupq_n_f64(0.0);
    vc2.val[3] = vdupq_n_f64(0.0);

    vc3.val[0] = vdupq_n_f64(0.0);
    vc3.val[1] = vdupq_n_f64(0.0);
    vc3.val[2] = vdupq_n_f64(0.0);
    vc3.val[3] = vdupq_n_f64(0.0);

    vc4.val[0] = vdupq_n_f64(0.0);
    vc4.val[1] = vdupq_n_f64(0.0);
    vc4.val[2] = vdupq_n_f64(0.0);
    vc4.val[3] = vdupq_n_f64(0.0);

    vc5.val[0] = vdupq_n_f64(0.0);
    vc5.val[1] = vdupq_n_f64(0.0);
    vc5.val[2] = vdupq_n_f64(0.0);
    vc5.val[3] = vdupq_n_f64(0.0);

#ifdef __GNUC__
#pragma GCC unroll(3)
#else
#pragma unroll(3)
#endif
    for (uint64_t i = 0; i < kk; ++i) {
        float64x2_t a;
        va0 = vld1q_f64_x3(_A);
        vb0 = vld1q_f64_x4(_B);

        a = vdupq_laneq_f64(va0.val[0], 0);
        vc0.val[0] = vfmaq_f64(vc0.val[0], vb0.val[0], a);
        vc0.val[1] = vfmaq_f64(vc0.val[1], vb0.val[1], a);
        vc0.val[2] = vfmaq_f64(vc0.val[2], vb0.val[2], a);
        vc0.val[3] = vfmaq_f64(vc0.val[3], vb0.val[3], a);

        a = vdupq_laneq_f64(va0.val[0], 1);
        vc1.val[0] = vfmaq_f64(vc1.val[0], vb0.val[0], a);
        vc1.val[1] = vfmaq_f64(vc1.val[1], vb0.val[1], a);
        vc1.val[2] = vfmaq_f64(vc1.val[2], vb0.val[2], a);
        vc1.val[3] = vfmaq_f64(vc1.val[3], vb0.val[3], a);

        a = vdupq_laneq_f64(va0.val[1], 0);
        vc2.val[0] = vfmaq_f64(vc2.val[0], vb0.val[0], a);
        vc2.val[1] = vfmaq_f64(vc2.val[1], vb0.val[1], a);
        vc2.val[2] = vfmaq_f64(vc2.val[2], vb0.val[2], a);
        vc2.val[3] = vfmaq_f64(vc2.val[3], vb0.val[3], a);

        a = vdupq_laneq_f64(va0.val[1], 1);
        vc3.val[0] = vfmaq_f64(vc3.val[0], vb0.val[0], a);
        vc3.val[1] = vfmaq_f64(vc3.val[1], vb0.val[1], a);
        vc3.val[2] = vfmaq_f64(vc3.val[2], vb0.val[2], a);
        vc3.val[3] = vfmaq_f64(vc3.val[3], vb0.val[3], a);

        a = vdupq_laneq_f64(va0.val[2], 0);
        vc4.val[0] = vfmaq_f64(vc4.val[0], vb0.val[0], a);
        vc4.val[1] = vfmaq_f64(vc4.val[1], vb0.val[1], a);
        vc4.val[2] = vfmaq_f64(vc4.val[2], vb0.val[2], a);
        vc4.val[3] = vfmaq_f64(vc4.val[3], vb0.val[3], a);

        a = vdupq_laneq_f64(va0.val[2], 1);
        vc5.val[0] = vfmaq_f64(vc5.val[0], vb0.val[0], a);
        vc5.val[1] = vfmaq_f64(vc5.val[1], vb0.val[1], a);
        vc5.val[2] = vfmaq_f64(vc5.val[2], vb0.val[2], a);
        vc5.val[3] = vfmaq_f64(vc5.val[3], vb0.val[3], a);

        _A += MR;
        _B += NR;
    }

    __builtin_prefetch(A_next + 0x00, 0, 3);
    __builtin_prefetch(A_next + 0x08, 0, 3);
    __builtin_prefetch(A_next + 0x10, 0, 3);
    __builtin_prefetch(B_next + 0x00, 0, 3);
    __builtin_prefetch(B_next + 0x08, 0, 3);
    __builtin_prefetch(B_next + 0x10, 0, 3);

    // TODO: vc0~vc5 *= alpha

    // TODO: somehow *= beta

    vC0 = vld1q_f64_x4(C + 0 * ldc);
    vC1 = vld1q_f64_x4(C + 1 * ldc);

    vC0.val[0] = vaddq_f64(vC0.val[0], vc0.val[0]);
    vC0.val[1] = vaddq_f64(vC0.val[1], vc0.val[1]);
    vC0.val[2] = vaddq_f64(vC0.val[2], vc0.val[2]);
    vC0.val[3] = vaddq_f64(vC0.val[3], vc0.val[3]);

    vC1.val[0] = vaddq_f64(vC1.val[0], vc1.val[0]);
    vC1.val[1] = vaddq_f64(vC1.val[1], vc1.val[1]);
    vC1.val[2] = vaddq_f64(vC1.val[2], vc1.val[2]);
    vC1.val[3] = vaddq_f64(vC1.val[3], vc1.val[3]);

    vst1q_f64_x4(C + 0 * ldc, vC0);
    vst1q_f64_x4(C + 1 * ldc, vC1);

    vC2 = vld1q_f64_x4(C + 2 * ldc);
    vC3 = vld1q_f64_x4(C + 3 * ldc);
    vC4 = vld1q_f64_x4(C + 4 * ldc);
    vC5 = vld1q_f64_x4(C + 5 * ldc);

    vC2.val[0] = vaddq_f64(vC2.val[0], vc2.val[0]);
    vC2.val[1] = vaddq_f64(vC2.val[1], vc2.val[1]);
    vC2.val[2] = vaddq_f64(vC2.val[2], vc2.val[2]);
    vC2.val[3] = vaddq_f64(vC2.val[3], vc2.val[3]);

    vC3.val[0] = vaddq_f64(vC3.val[0], vc3.val[0]);
    vC3.val[1] = vaddq_f64(vC3.val[1], vc3.val[1]);
    vC3.val[2] = vaddq_f64(vC3.val[2], vc3.val[2]);
    vC3.val[3] = vaddq_f64(vC3.val[3], vc3.val[3]);

    vC4.val[0] = vaddq_f64(vC4.val[0], vc4.val[0]);
    vC4.val[1] = vaddq_f64(vC4.val[1], vc4.val[1]);
    vC4.val[2] = vaddq_f64(vC4.val[2], vc4.val[2]);
    vC4.val[3] = vaddq_f64(vC4.val[3], vc4.val[3]);

    vC5.val[0] = vaddq_f64(vC5.val[0], vc5.val[0]);
    vC5.val[1] = vaddq_f64(vC5.val[1], vc5.val[1]);
    vC5.val[2] = vaddq_f64(vC5.val[2], vc5.val[2]);
    vC5.val[3] = vaddq_f64(vC5.val[3], vc5.val[3]);

    vst1q_f64_x4(C + 2 * ldc, vC2);
    vst1q_f64_x4(C + 3 * ldc, vC3);
    vst1q_f64_x4(C + 4 * ldc, vC4);
    vst1q_f64_x4(C + 5 * ldc, vC5);
}

static void micro_dxpy(uint64_t m, uint64_t n, double *restrict C, int ldc,
                       const double *restrict _C) {
  for (uint64_t i = 0; i < m; ++i) {
    for (uint64_t j = 0; j < n; ++j) {
      C[j] += _C[j];
    }
    C += ldc;
    _C += NR;
  }
}

static void inner_kernel(uint64_t mm, uint64_t nn, uint64_t kk,
                         const double *restrict _A, const double *restrict _B,
                         double *restrict C, uint64_t ldc) {
    const uint64_t mmc = (mm + MR - 1) / MR;
    const uint64_t mmr = mm % MR;
    const uint64_t nnc = (nn + NR - 1) / NR;
    const uint64_t nnr = nn % NR;

    const double *A_next;
    const double *B_next;

    A_next = _A;
    B_next = _B;
    for (uint64_t nni = 0; nni < nnc; ++nni) {
        const uint64_t nnn = (nni != nnc - 1 || nnr == 0) ? NR : nnr;

        if (nni != nnc - 1) {
          B_next += NR * kk;
		    } else {
            B_next = _B;
		    }

        for (uint64_t mmi = 0; mmi < mmc; ++mmi) {
            const uint64_t mmm = (mmi != mmc - 1 || mmr == 0) ? MR : mmr;

            if (mmi != mmc - 1) {
                A_next += MR * kk;
			      } else {
                A_next = _A;
			      }

            if (mmm == MR && nnn == NR) {
                micro_kernel(kk, _A + mmi * MR * kk, _B + nni * NR * kk,
                             C + mmi * MR * ldc + nni * NR, ldc, A_next,
                             B_next);
            } else {
                double _C[MR * NR] __attribute__((aligned(CACHE_LINE))) = {};
                micro_kernel(kk, _A + mmi * MR * kk, _B + nni * NR * kk, _C, NR,
                             A_next, B_next);
                micro_dxpy(mmm, nnn, C + mmi * MR * ldc + nni * NR, ldc, _C);
            }
        }
    }
}

static void packarc(uint64_t mm, uint64_t kk, const double *restrict A,
                    uint64_t lda, double *restrict _A) {
    const uint64_t mmc = (mm + MR - 1) / MR;
    const uint64_t mmr = mm % MR;
    for (uint64_t mmi = 0; mmi < mmc; ++mmi) {
        const uint64_t mmm = (mmi != mmc - 1 || mmr == 0) ? MR : mmr;
        for (uint64_t i = 0; i < mmm; ++i) {
            for (uint64_t j = 0; j < kk; ++j) {
                _A[mmi * MR * kk + i + j * MR] =
                    A[mmi * MR * lda + i * lda + j];
            }
        }
    }
}

static void packbrr(uint64_t kk, uint64_t nn, const double *restrict B,
                    uint64_t ldb, double *restrict _B) {
    const uint64_t nnc = (nn + NR - 1) / NR;
    const uint64_t nnr = nn % NR;
#pragma omp parallel for num_threads(NT2)
    for (uint64_t j = 0; j < kk; ++j) {
        for (uint64_t nni = 0; nni < nnc; ++nni) {
            const uint64_t nnn = (nni != nnc - 1 || nnr == 0) ? NR : nnr;
            memcpy(_B + j * NR + nni * kk * NR, B + j * ldb + nni * NR,
                   sizeof(double) * nnn);
        }
    }
}

void call_dgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                CBLAS_TRANSPOSE TransB, int64_t m, int64_t n, int64_t k,
                double alpha, const double *A, int64_t lda, const double *B,
                int64_t ldb, double beta, double *C, int64_t ldc) {
    //const uint64_t is_C_row = (layout == CblasRowMajor ? 1         : 0       );
    //const uint64_t is_A_row = (TransA == CblasTrans    ? !is_C_row : is_C_row);
    //const uint64_t is_B_row = (TransB == CblasTrans    ? !is_C_row : is_C_row);

  const uint64_t mc = (m + MB - 1) / MB;
  const uint64_t mr = m % MB;
  const uint64_t nc = (n + NB - 1) / NB;
  const uint64_t nr = n % NB;
  const uint64_t kc = (k + KB - 1) / KB;
  const uint64_t kr = k % KB;

    /*
void userdgemm(uint64_t m, uint64_t n, uint64_t k, double *A, uint64_t lda,
               double *B, uint64_t ldb, double *C, uint64_t ldc) {
    const uint64_t mc = (m + MB - 1) / MB;
    const uint64_t mr = m % MB;
    const uint64_t nc = (n + NB - 1) / NB;
    const uint64_t nr = n % NB;
    const uint64_t kc = (k + KB - 1) / KB;
    const uint64_t kr = k % KB;
    */

#pragma omp parallel num_threads(NT1)
  {
    double *_A_arr[NT2];
    double *_B;
    for (uint64_t i = 0; i < NT2; ++i) {
      posix_memalign((void **)(&_A_arr[i]), PAGE_SIZE,
                      sizeof(double) * MB * KB);
    }
    posix_memalign((void **)&_B, PAGE_SIZE, sizeof(double) * KB * NB);

#pragma omp for
    for (uint64_t ni = 0; ni < nc; ++ni) {
      const uint64_t nn = (ni != nc - 1 || nr == 0) ? NB : nr;

      for (uint64_t ki = 0; ki < kc; ++ki) {
        const uint64_t kk = (ki != kc - 1 || kr == 0) ? KB : kr;

        packbrr(kk, nn, B + ki * KB * ldb + ni * NB, ldb, _B);

#pragma omp parallel num_threads(NT2)
        {
          double *_A = _A_arr[omp_get_thread_num()];

#pragma omp for
          for (uint64_t mi = 0; mi < mc; ++mi) {
            const uint64_t mm = (mi != mc - 1 || mr == 0) ? MB : mr;

            packarc(mm, kk, A + mi * MB * lda + ki * KB, lda, _A);

            inner_kernel(mm, nn, kk, _A, _B,
                          C + mi * MB * ldc + ni * NB, ldc);
          }
        }
      }
    }

    for (uint64_t i = 0; i < NT2; ++i) {
      free(_A_arr[i]);
    }
    free(_B);
  }
}
