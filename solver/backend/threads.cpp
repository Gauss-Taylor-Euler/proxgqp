#include "backend/threads.hpp"

#include <algorithm>

extern "C" {
void openblas_set_num_threads(int) __attribute__((weak));
int openblas_get_num_procs(void) __attribute__((weak));
void MKL_Set_Num_Threads(int) __attribute__((weak));
int MKL_Get_Max_Threads(void) __attribute__((weak));
void omp_set_num_threads(int) __attribute__((weak));
int omp_get_max_threads(void) __attribute__((weak));
}

namespace proxgqp {

std::size_t configure_dense_threads(std::size_t requested) {
  const int count = static_cast<int>(std::max<std::size_t>(1, requested));
  if (count == 1) return 1;
  bool applied = false;
  if (openblas_set_num_threads) {
    openblas_set_num_threads(count);
    applied = true;
  }
  if (MKL_Set_Num_Threads) {
    MKL_Set_Num_Threads(count);
    applied = true;
  }
  if (omp_set_num_threads) {
    omp_set_num_threads(count);
    applied = true;
  }
  return applied ? static_cast<std::size_t>(count) : 1;
}

std::size_t available_dense_threads() {
  if (openblas_get_num_procs) {
    const int count = openblas_get_num_procs();
    if (count > 0) return static_cast<std::size_t>(count);
  }
  if (MKL_Get_Max_Threads) {
    const int count = MKL_Get_Max_Threads();
    if (count > 0) return static_cast<std::size_t>(count);
  }
  if (omp_get_max_threads) {
    const int count = omp_get_max_threads();
    if (count > 0) return static_cast<std::size_t>(count);
  }
  return 1;
}

}
