/**
 * @file grace/kernel.sve.c
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

#include "sve.h"
#include "common.h"

#ifdef OC
#define TOTAL_CORE 1
#else
#define TOTAL_CORE 72
#endif
#define NUMA_NODE 1

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
#define MB (MR * 7)
#endif

#ifndef NB
#define NB (NR * 49)
#endif

#ifndef KB
#define KB (4 * 66)
#endif

#define USE_LDP

#ifndef MK_UNROLL_DEPTH
#define MK_UNROLL_DEPTH 2
#endif

#ifndef MK_PREFETCH_A_DISTANCE
#define MK_PREFETCH_A_DISTANCE (MR * 7)
#endif

#ifndef MK_PREFETCH_A_LOCALITY
#define MK_PREFETCH_A_LOCALITY LOCALITY_NONE
#endif

#ifndef MK_PREFETCH_B_DISTANCE
#define MK_PREFETCH_B_DISTANCE (NR * 10)
#endif

// 0 ~ 4
#ifndef MK_PREFETCH_B_DEPTH
#define MK_PREFETCH_B_DEPTH 2
#endif

#ifndef MK_PREFETCH_B_LOCALITY
#define MK_PREFETCH_B_LOCALITY LOCALITY_NONE
#endif

#ifndef MK_PREFETCH_C_DEPTH
#define MK_PREFETCH_C_DEPTH 4
#endif

#ifndef MK_PREFETCH_C_LOCALITY
#define MK_PREFETCH_C_LOCALITY LOCALITY_HIGH
#endif

#ifndef ARC_PREFETCH_DEPTH
#define ARC_PREFETCH_DEPTH 2
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

static void micro_kernel(
    uint64_t kk,
    const double *restrict _A,
    const double *restrict _B,
    double *restrict C,
    const uint64_t ldc)
{
    float64x2x2_t va0;
    float64x2x4_t vb0;
    float64x2x4_t vc0, vc1, vc2, vc3;
    float64x2x4_t vC0, vC1, vC2, vC3;

#pragma unroll(MK_PREFETCH_C_DEPTH)
    for (uint8_t i = 0; i < MK_PREFETCH_C_DEPTH; ++i)
    {
        __builtin_prefetch(C + i * ldc, READ, MK_PREFETCH_C_LOCALITY);
    }

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

#pragma unroll(3)
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

        float64x2_t a;
        va0 = vld1q_f64_x2(_A);
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

        _A += MR;
        _B += NR;
    }

    vC0 = vld1q_f64_x4(C + 0 * ldc);
    vC1 = vld1q_f64_x4(C + 1 * ldc);
    vC2 = vld1q_f64_x4(C + 2 * ldc);
    vC3 = vld1q_f64_x4(C + 3 * ldc);

    vC0.val[0] = vaddq_f64(vC0.val[0], vc0.val[0]);
    vC0.val[1] = vaddq_f64(vC0.val[1], vc0.val[1]);
    vC0.val[2] = vaddq_f64(vC0.val[2], vc0.val[2]);
    vC0.val[3] = vaddq_f64(vC0.val[3], vc0.val[3]);

    vC1.val[0] = vaddq_f64(vC1.val[0], vc1.val[0]);
    vC1.val[1] = vaddq_f64(vC1.val[1], vc1.val[1]);
    vC1.val[2] = vaddq_f64(vC1.val[2], vc1.val[2]);
    vC1.val[3] = vaddq_f64(vC1.val[3], vc1.val[3]);

    vC2.val[0] = vaddq_f64(vC2.val[0], vc2.val[0]);
    vC2.val[1] = vaddq_f64(vC2.val[1], vc2.val[1]);
    vC2.val[2] = vaddq_f64(vC2.val[2], vc2.val[2]);
    vC2.val[3] = vaddq_f64(vC2.val[3], vc2.val[3]);

    vC3.val[0] = vaddq_f64(vC3.val[0], vc3.val[0]);
    vC3.val[1] = vaddq_f64(vC3.val[1], vc3.val[1]);
    vC3.val[2] = vaddq_f64(vC3.val[2], vc3.val[2]);
    vC3.val[3] = vaddq_f64(vC3.val[3], vc3.val[3]);

    vst1q_f64_x4(C + 0 * ldc, vC0);
    vst1q_f64_x4(C + 1 * ldc, vC1);
    vst1q_f64_x4(C + 2 * ldc, vC2);
    vst1q_f64_x4(C + 3 * ldc, vC3);
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
            const register uint64_t mmm = (mmi != mmc - 1 || mmr == 0) ? MR : mmr;

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
            const register uint64_t kkk = kki != kkc - 1 ? CACHE_ELEM : kkr;

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

    static double *_A[TOTAL_CORE] = {
        NULL,
    };
    static double *_B[TOTAL_CORE] = {
        NULL,
    };

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

#ifndef DISABLE_MEMORY_BUFFER
        if (_A[pid] == NULL)
        {
#endif
            _A[pid] = numa_alloc_onnode(sizeof(double) * (MB + MR) * KB, nid);
            _B[pid] = numa_alloc_onnode(sizeof(double) * KB * (NB + NR), nid);
#ifndef DISABLE_MEMORY_BUFFER
        }
#endif

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
        pthread_create(&tinfo[pid].thread_id, &attr, &middle_kernel, &tinfo[pid]);
    }

    pthread_attr_destroy(&attr);

    for (uint64_t pid = 0; pid < TOTAL_CORE; ++pid)
    {
        void *res;
        pthread_join(tinfo[pid].thread_id, &res);

#ifdef DISABLE_MEMORY_BUFFER
        numa_free(_A[pid], sizeof(double) * (MB + MR) * KB);
        numa_free(_B[pid], sizeof(double) * KB * (NB + NR));
#endif
    }
}
