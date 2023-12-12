#ifndef _FRESCO_OMP_UTILS_HPP_
#define _FRESCO_OMP_UTILS_HPP_

#include <iostream>

// #if __APPLE__
// #include "/usr/local/opt/libomp/include/omp.h"
// #else
// #include <omp.h>
// #endif

#include <omp.h>

namespace {

// Credits: http://stackoverflow.com/a/13328691/140510
// and https://github.com/rmax/omp-thread-count/blob/master/src/omp_thread_count/include/omp_thread_count.h
size_t omp_get_thread_count() {
  size_t nthreads = 0;
  #ifdef _OPENMP
  #pragma omp parallel reduction(+:nthreads)
  nthreads += 1;
  #else
  nthreads = 1;
  #endif
  return nthreads;
}

}

#endif
