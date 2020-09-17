
#include "user.hpp"

#include "Kokkos_Core.hpp"

#include <cstdint>

struct InitView {
  explicit InitView(view_type _v) : m_view(_v) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const { m_view(i, i % 2) = i; }

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
  Kokkos::RangePolicy<exec_space, int> range(0, n);
  Kokkos::parallel_for("generate_view", range, InitView{_v});
  return _v;
}
