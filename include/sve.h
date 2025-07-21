#pragma once

#include <arm_sve.h>

#define svzero_f64() svdup_n_f64(0.0)
#define svzero4_f64() svcreate4_f64(svzero_f64(), svzero_f64(), svzero_f64(), svzero_f64())
