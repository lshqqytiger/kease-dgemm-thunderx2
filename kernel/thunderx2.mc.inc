#define TOTAL_CORE 64
#define NUMA_NODE 2

#ifndef CM
#define CM 8
#endif
#define CN (TOTAL_CORE / CM)

#ifndef MB
#define MB (MR * 8)
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
#define ARC_PREFETCH_DEPTH 3
#endif

#ifndef ARC_PREFETCH_LOCALITY
#define ARC_PREFETCH_LOCALITY LOCALITY_NONE
#endif
