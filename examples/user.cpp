
#include "user.hpp"

#include "Kokkos_Core.hpp"

#include <iostream>
#include <random>
#include <chrono>

view_type generate_view(size_t n) {
  // obtain a seed from the system clock:
  auto seed = std::chrono::system_clock::now().time_since_epoch().count();

  std::default_random_engine generator(seed);
  auto get_random = [&]() {
    return std::generate_canonical<double, 12>(generator);
  };
  view_type _v("random_view", n, 2);
  double sum = 0.0;
  for (size_t i = 0; i < n; ++i) {
    _v(i, 0) = 0.0;
    _v(i, 1) = 0.0;
  }
  for (size_t i = 0; i < n; ++i) _v(i, i % 2) = i;
  for (size_t i = 0; i < n; ++i) {
    sum += _v(i, 0) + _v(i, 1);
    std::cout << "    view(" << i << ") = " << std::setw(2) << _v(i, 0) << " "
              << std::setw(2) << _v(i, 1) << std::endl;
  }
  std::cout << "Sum of view: " << sum << std::endl;
  std::cout << "extent(0): " << _v.extent(0) << std::endl;
  std::cout << "stride(0): " << _v.stride(0) << std::endl;
  return _v;
}
