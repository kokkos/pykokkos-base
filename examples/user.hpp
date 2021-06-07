
#include "Kokkos_Core.hpp"

#include <cstdint>

using view_type = Kokkos::View<double**, Kokkos::DefaultExecutionSpace>;

view_type generate_view(size_t);
void modify_view(view_type);
