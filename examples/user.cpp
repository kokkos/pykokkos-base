
#include "user.hpp"

#include "Kokkos_Core.hpp"

#include <iostream>
#include <random>
#include <chrono>
#include <iomanip>

///
/// \fn generate_view
/// \brief This is meant to emulate some function that exists in a user library
/// which returns a Kokkos::View and will have a python binding
///
view_type generate_view(size_t n) {
  view_type _v("random_view", n, 2);
  double sum = 0.0;
  for (size_t i = 0; i < n; ++i) {
    sum += (_v(i, i % 2) = i);
    std::cout << "    view(" << i << ") = " << std::setw(2) << _v(i, 0) << " "
              << std::setw(2) << _v(i, 1) << std::endl;
  }
  std::cout << "Sum of view: " << sum << std::endl;
  std::cout << "extent(0): " << _v.extent(0) << std::endl;
  std::cout << "stride(0): " << _v.stride(0) << std::endl;
  return _v;
}
