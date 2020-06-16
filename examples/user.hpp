
#include "Kokkos_Core.hpp"

using view_type = Kokkos::View<double**, Kokkos::HostSpace>;
view_type generate_view(size_t);
