
#include "user.hpp"

#include "Kokkos_Core.hpp"

#include <iostream>
#include <random>
#include <chrono>
#include <iomanip>

struct InitView {
  explicit InitView(view_type _v) : m_view(_v) {
    std::cout << "extent: " << m_view.extent(0) << ", " << m_view.extent(1)
              << '\n';
    std::cout << "stride: " << m_view.stride(0) << ", " << m_view.stride(1)
              << '\n';
    std::cout << std::flush;
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    m_view(i, i % 2) = i;
    std::cout.precision(0);
    std::cout << std::showpoint << "    view(" << i << ") = [" << std::setw(2)
              << m_view(i, 0) << ' ' << std::setw(2) << m_view(i, 1) << "]\n";
  }

 private:
  view_type m_view;
};

///
/// \fn generate_view
/// \brief This is meant to emulate some function that exists in a user library
/// which returns a Kokkos::View and will have a python binding
///
view_type generate_view(size_t n) {
  view_type _v("user_view", n, 2);
  Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace, int> range(0, n);
  Kokkos::parallel_for("generate_view", range, InitView{_v});
  return _v;
}
