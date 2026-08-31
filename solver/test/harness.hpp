#pragma once

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace proxgqp {
namespace test {

inline int& failure_count() {
  static int count = 0;
  return count;
}

inline const char*& active_name() {
  static const char* name = "";
  return name;
}

inline void report(int line, const std::string& detail) {
  std::fflush(stdout);
  ++failure_count();
  std::printf("  FAIL %s line %d: %s\n", active_name(), line, detail.c_str());
}

inline bool close(double left, double right, double tolerance) {
  const double scale = std::max({1.0, std::abs(left), std::abs(right)});
  return std::abs(left - right) <= tolerance * scale;
}

}
}

#define CHECK(condition)                                                    \
  do {                                                                      \
    if (!(condition)) proxgqp::test::report(__LINE__, #condition);          \
  } while (false)

#define CHECK_CLOSE(left, right, tolerance)                                 \
  do {                                                                      \
    const double lhs = (left), rhs = (right);                               \
    if (!proxgqp::test::close(lhs, rhs, tolerance))                         \
      proxgqp::test::report(__LINE__,                                       \
                            std::string(#left " != " #right " (") +         \
                                std::to_string(lhs) + " vs " +              \
                                std::to_string(rhs) + ")");                 \
  } while (false)

#define RUN(function)                                                       \
  do {                                                                      \
    proxgqp::test::active_name() = #function;                               \
    std::printf("  run  %s\n", #function);                                  \
    std::fflush(stdout);                               \
    const int before = proxgqp::test::failure_count();                      \
    function();                                                             \
    if (proxgqp::test::failure_count() == before)                           \
      std::printf("  ok   %s\n", #function);                                \
  } while (false)

#define TEST_MAIN                                                           \
  int main() {                                                              \
    run_all();                                                              \
    const int failures = proxgqp::test::failure_count();                    \
    std::printf("%s: %d failure%s\n", TEST_NAME, failures,                  \
                failures == 1 ? "" : "s");                                  \
    return failures == 0 ? 0 : 1;                                           \
  }
