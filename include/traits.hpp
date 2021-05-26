/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#pragma once

#include "Kokkos_Core_fwd.hpp"
#include "Kokkos_Layout.hpp"
#include "Kokkos_MemoryTraits.hpp"

#include "common.hpp"
#include "fwd.hpp"

//--------------------------------------------------------------------------------------//

VIEW_DATA_DIMS(1, T *)
VIEW_DATA_DIMS(2, T **)
VIEW_DATA_DIMS(3, T ***)
VIEW_DATA_DIMS(4, T ****)
VIEW_DATA_DIMS(5, T *****)
VIEW_DATA_DIMS(6, T ******)
VIEW_DATA_DIMS(7, T *******)
VIEW_DATA_DIMS(8, T ********)

//--------------------------------------------------------------------------------------//

VIEW_DATA_TYPE(Int16, int16_t, "int16", "short")
VIEW_DATA_TYPE(Int32, int32_t, "int32", "int")
VIEW_DATA_TYPE(Int64, int64_t, "int64", "long")
VIEW_DATA_TYPE(Uint16, uint16_t, "uint16", "unsigned_short")
VIEW_DATA_TYPE(Uint32, uint32_t, "uint32", "unsigned", "unsigned_int")
VIEW_DATA_TYPE(Uint64, uint64_t, "uint64", "unsigned_long")
VIEW_DATA_TYPE(Float32, float, "float32", "float")
VIEW_DATA_TYPE(Float64, double, "float64", "double")

//--------------------------------------------------------------------------------------//

#if !defined(ENABLE_LAYOUTS)
DISABLE_TYPE(Kokkos::LayoutLeft)
#endif
DISABLE_TYPE(Kokkos::LayoutStride)

VIEW_LAYOUT_TYPE(Left, Kokkos::LayoutLeft, "LayoutLeft")
VIEW_LAYOUT_TYPE(Right, Kokkos::LayoutRight, "LayoutRight")
VIEW_LAYOUT_TYPE(Stride, Kokkos::LayoutStride, "LayoutStride")

ENABLE_IMPLICIT(Kokkos::LayoutRight)

//--------------------------------------------------------------------------------------//

#if !defined(ENABLE_MEMORY_TRAITS)
DISABLE_TYPE(Kokkos::MemoryTraits<Kokkos::Atomic>)
DISABLE_TYPE(Kokkos::MemoryTraits<Kokkos::RandomAccess>)
DISABLE_TYPE(Kokkos::MemoryTraits<Kokkos::Restrict>)
#endif

VIEW_MEMORY_TRAIT(Managed, Kokkos::MemoryTraits<0>, "Managed")
VIEW_MEMORY_TRAIT(Unmanaged, Kokkos::MemoryTraits<Kokkos::Unmanaged>,
                  "Unmanaged")
VIEW_MEMORY_TRAIT(Atomic, Kokkos::MemoryTraits<Kokkos::Atomic>, "Atomic")
VIEW_MEMORY_TRAIT(RandomAccess, Kokkos::MemoryTraits<Kokkos::RandomAccess>,
                  "RandomAccess")
VIEW_MEMORY_TRAIT(Restrict, Kokkos::MemoryTraits<Kokkos::Restrict>, "Restrict")

ENABLE_IMPLICIT(Kokkos::MemoryTraits<0>)

//--------------------------------------------------------------------------------------//

#if !defined(KOKKOS_ENABLE_SERIAL)
DISABLE_TYPE(Kokkos::Serial)
#endif

#if !defined(KOKKOS_ENABLE_THREADS)
DISABLE_TYPE(Kokkos::Threads)
#endif

#if !defined(KOKKOS_ENABLE_OPENMP)
DISABLE_TYPE(Kokkos::OpenMP)
#endif

#if !defined(KOKKOS_ENABLE_CUDA)
DISABLE_TYPE(Kokkos::Cuda)
#endif

#if !defined(KOKKOS_ENABLE_OPENMPTARGET)
DISABLE_TYPE(Kokkos::Experimental::OpenMPTarget)
#endif

#if !defined(KOKKOS_ENABLE_HIP)
DISABLE_TYPE(Kokkos::Experimental::HIP)
#endif

#if !defined(KOKKOS_ENABLE_SYCL)
DISABLE_TYPE(Kokkos::Experimental::SYCL)
#endif

VIEW_SPACE_IDX(Kokkos::Serial, HostSpace)
VIEW_SPACE_IDX(Kokkos::Threads, HostSpace)
VIEW_SPACE_IDX(Kokkos::OpenMP, HostSpace)
VIEW_SPACE_IDX(Kokkos::Cuda, CudaSpace)
VIEW_SPACE_IDX(Kokkos::Experimental::HPX, HostSpace)
VIEW_SPACE_IDX(Kokkos::Experimental::HIP, HIPSpace)
VIEW_SPACE_IDX(Kokkos::Experimental::SYCL, SYCLSharedUSMSpace)
VIEW_SPACE_IDX(Kokkos::Experimental::OpenMPTarget, OpenMPTargetSpace)

//--------------------------------------------------------------------------------------//
//  declare any spaces that might not be available and mark them as unavailable
//  we declare these so that we can map the enum value to the type along with a
//  label
//
DISABLE_TYPE(Kokkos::AnonymousSpace)

#if !defined(KOKKOS_ENABLE_CUDA)
DISABLE_TYPE(Kokkos::CudaSpace)
DISABLE_TYPE(Kokkos::CudaHostPinnedSpace)
#endif

#if !defined(KOKKOS_ENABLE_CUDA_UVM)
DISABLE_TYPE(Kokkos::CudaUVMSpace)
#endif

#if !defined(KOKKOS_ENABLE_HBWSPACE)
DISABLE_TYPE(Kokkos::Experimental::HBWSpace)
#endif

#if !defined(KOKKOS_ENABLE_HIP)
DISABLE_TYPE(Kokkos::Experimental::HIPSpace)
DISABLE_TYPE(Kokkos::Experimental::HIPHostPinnedSpace)
#endif

#if !defined(KOKKOS_ENABLE_HPX)
DISABLE_TYPE(Kokkos::Experimental::HPX)
#endif

#if !defined(KOKKOS_ENABLE_OPENMPTARGET)
DISABLE_TYPE(Kokkos::Experimental::OpenMPTargetSpace)
#endif

#if !defined(KOKKOS_ENABLE_SYCL)
DISABLE_TYPE(Kokkos::Experimental::SYCLSharedUSMSpace)
DISABLE_TYPE(Kokkos::Experimental::SYCLDeviceUSMSpace)
#endif

VIEW_SPACE(Kokkos::HostSpace, HostSpace, "HostSpace", "Host", "Serial",
           "Threads", "OpenMP", "HPX")
VIEW_SPACE(Kokkos::AnonymousSpace, AnonymousSpace, "AnonymousSpace")
VIEW_SPACE(Kokkos::CudaSpace, CudaSpace, "CudaSpace", "Cuda")
VIEW_SPACE(Kokkos::CudaUVMSpace, CudaUVMSpace, "CudaUVMSpace")
VIEW_SPACE(Kokkos::CudaHostPinnedSpace, CudaHostPinnedSpace,
           "CudaHostPinnedSpace")
VIEW_SPACE(Kokkos::Experimental::HBWSpace, HBWSpace, "HBWSpace")
VIEW_SPACE(Kokkos::Experimental::HIPSpace, HIPSpace, "HIPSpace", "HIP")
VIEW_SPACE(Kokkos::Experimental::HIPHostPinnedSpace, HIPHostPinnedSpace,
           "HIPHostPinnedSpace")
VIEW_SPACE(Kokkos::Experimental::OpenMPTargetSpace, OpenMPTargetSpace,
           "OpenMPTargetSpace", "OpenMPTarget")
VIEW_SPACE(Kokkos::Experimental::SYCLSharedUSMSpace, SYCLSharedUSMSpace,
           "SYCLSharedUSMSpace", "SYCL")
VIEW_SPACE(Kokkos::Experimental::SYCLDeviceUSMSpace, SYCLDeviceUSMSpace,
           "SYCLDeviceUSMSpace")
