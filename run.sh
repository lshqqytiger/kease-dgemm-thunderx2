#!/bin/bash

set -e

export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_NUM_THREADS=64

OUT_PATH=./out

#for kernel in $@; do
for kernel in kernel.latest; do
  path=${OUT_PATH}/${kernel}.out
  make -s ${path}

  for size in {10000..10000..100}; do
    m=$size
    n=$size
    k=$size

    ${path} 64 Row N N $m $n $k 1.0 1.0 5
  done
done
