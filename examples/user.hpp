
#include "Kokkos_Core.hpp"

#include <cstdint>

#if defined(KOKKOS_ENABLE_CUDA_UVM)
using exec_space = Kokkos::Cuda;
using view_space = Kokkos::CudaUVMSpace;
#elif defined(KOKKOS_ENABLE_CUDA)
using exec_space = Kokkos::Cuda;
using view_space = Kokkos::CudaSpace;
#elif defined(KOKKOS_ENABLE_HIP)
using exec_space = Kokkos::Experimental::HIP;
using view_space = Kokkos::Experimental::HIPSpace;
#else
using exec_space = Kokkos::DefaultHostExecutionSpace;
using view_space = Kokkos::HostSpace;
#endif

using view_type = Kokkos::View<double**, Kokkos::LayoutRight, view_space>;
view_type generate_view(size_t);
