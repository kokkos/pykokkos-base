
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

struct ModifyView {
  explicit ModifyView(view_type _v) : m_view(_v) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const { m_view(i, i % 2) *= 2; }

 private:
  view_type m_view;
};

using exec_space = typename view_type::traits::execution_space;
///
/// \fn generate_view
/// \brief This is meant to emulate some function that exists in a user library
/// which returns a Kokkos::View and will have a python binding
///
view_type generate_view(size_t n) {
  if (!Kokkos::is_initialized()) {
    std::cerr << "[user-bindings]> Initializing Kokkos..." << std::endl;
    Kokkos::initialize();
  }
  view_type _v("user_view", n, 2);
  Kokkos::RangePolicy<exec_space, int> range(0, n);
  Kokkos::parallel_for("generate_view", range, InitView{_v});
  return _v;
}

void modify_view(view_type _v) {
  Kokkos::RangePolicy<exec_space, int> range(0, _v.extent(1));
  Kokkos::parallel_for("modify_view", range, ModifyView{_v});
}
