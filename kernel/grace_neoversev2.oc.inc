#define TOTAL_CORE 1

#define CM 1
#define CN 1

#ifndef MB
#define MB (MR * 1)
#endif

#ifndef NB
#define NB (NR * 1)
#endif

#ifndef KB
#define KB (4 * 1)
#endif

// #define USE_LDP

#ifndef MK_UNROLL_DEPTH
#define MK_UNROLL_DEPTH 1
#endif

#ifndef MK_PREFETCH_A_DISTANCE
#define MK_PREFETCH_A_DISTANCE (MR * 0)
#endif

#ifndef MK_PREFETCH_A_LOCALITY
#define MK_PREFETCH_A_LOCALITY LOCALITY_NONE
#endif

#ifndef MK_PREFETCH_B_DISTANCE
#define MK_PREFETCH_B_DISTANCE (NR * 0)
#endif

#ifndef MK_PREFETCH_B_DEPTH
#define MK_PREFETCH_B_DEPTH 0
#endif

#ifndef MK_PREFETCH_B_LOCALITY
#define MK_PREFETCH_B_LOCALITY LOCALITY_NONE
#endif

#ifndef MK_PREFETCH_C_DEPTH
#define MK_PREFETCH_C_DEPTH 0
#endif

#ifndef MK_PREFETCH_C_LOCALITY
#define MK_PREFETCH_C_LOCALITY LOCALITY_NONE
#endif

#ifndef ARC_PREFETCH_DEPTH
#define ARC_PREFETCH_DEPTH 0
#endif

#ifndef ARC_PREFETCH_LOCALITY
#define ARC_PREFETCH_LOCALITY LOCALITY_NONE
#endif
