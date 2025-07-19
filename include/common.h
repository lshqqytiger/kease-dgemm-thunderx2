#pragma once

#define UNLIKELY(expr) __builtin_expect(!!(expr), 0)
#define LIKELY(expr) __builtin_expect(!!(expr), 1)

#define ROUND_UP(a, b) ((a + b - 1) / b)

#define CACHE_LINE 64
#define CACHE_ELEM (CACHE_LINE / 8)

#define READ 0
#define WRITE 1

#define LOCALITY_NONE 0
#define LOCALITY_LOW 1
#define LOCALITY_MODERATE 2
#define LOCALITY_HIGH 3
