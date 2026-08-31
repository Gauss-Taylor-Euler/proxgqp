#pragma once

#include <cstddef>

namespace proxgqp {

std::size_t configure_dense_threads(std::size_t requested);
std::size_t available_dense_threads();

}
