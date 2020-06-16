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

#include <string>
#include <cstdint>
#include <initializer_list>
#include <type_traits>

#if defined(ENABLE_DEMANGLE)
#include <cxxabi.h>
#endif

//--------------------------------------------------------------------------------------//

#define FOLD_EXPRESSION(...) \
  ::consume_parameters(::std::initializer_list<int>{(__VA_ARGS__, 0)...})

//--------------------------------------------------------------------------------------//

template <typename... Args>
void consume_parameters(Args&&...) {}

//--------------------------------------------------------------------------------------//

inline std::string demangle(const char* _cstr) {
#if defined(ENABLE_DEMANGLE)
  // demangling a string when delimiting
  int _ret      = 0;
  char* _demang = abi::__cxa_demangle(_cstr, 0, 0, &_ret);
  if (_demang && _ret == 0)
    return std::string(const_cast<const char*>(_demang));
  else
    return _cstr;
#else
  return _cstr;
#endif
}

//--------------------------------------------------------------------------------------//

inline std::string demangle(const std::string& _str) {
  return demangle(_str.c_str());
}

//--------------------------------------------------------------------------------------//

template <typename Tp>
inline std::string demangle() {
  return demangle(typeid(Tp).name());
}

//--------------------------------------------------------------------------------------//
//  this is used to mark memory spaces as unavailable
//
template <typename Tp>
struct is_available : std::true_type {};

#define DISABLE_TYPE(TYPE) \
  template <>              \
  struct is_available<TYPE> : std::false_type {};

//--------------------------------------------------------------------------------------//

static constexpr size_t ViewDataMaxDimensions = 7;

template <typename T, size_t Dim>
struct ViewDataTypeRepr;

#define VIEW_DATA_DIMS(NUM, ...)        \
  template <typename T>                 \
  struct ViewDataTypeRepr<T, NUM - 1> { \
    using type = __VA_ARGS__;           \
  };

VIEW_DATA_DIMS(1, T*)
VIEW_DATA_DIMS(2, T**)
VIEW_DATA_DIMS(3, T***)
VIEW_DATA_DIMS(4, T****)
VIEW_DATA_DIMS(5, T*****)
VIEW_DATA_DIMS(6, T******)
VIEW_DATA_DIMS(7, T*******)

//--------------------------------------------------------------------------------------//

enum KokkosViewDataType {
  Int16 = 0,
  Int32,
  Int64,
  Uint16,
  Uint32,
  Uint64,
  Float,
  Double,
  ViewDataTypesEnd
};

template <size_t DataT>
struct ViewDataTypeSpecialization;

#define VIEW_DATA_TYPE(ENUM_ID, DATA_TYPE, LABEL) \
  template <>                                     \
  struct ViewDataTypeSpecialization<ENUM_ID> {    \
    using type = DATA_TYPE;                       \
    static std::string label() { return LABEL; }  \
  };

VIEW_DATA_TYPE(Int16, int16_t, "int16")
VIEW_DATA_TYPE(Int32, int32_t, "int32")
VIEW_DATA_TYPE(Int64, int64_t, "int64")
VIEW_DATA_TYPE(Uint16, uint16_t, "uint16")
VIEW_DATA_TYPE(Uint32, uint32_t, "uint32")
VIEW_DATA_TYPE(Uint64, uint64_t, "uint64")
VIEW_DATA_TYPE(Float, float, "float")
VIEW_DATA_TYPE(Double, double, "double")

//--------------------------------------------------------------------------------------//
//  declare any spaces that might not be available and mark them as unavailable
//  we declare these so that we can map the enum value to the type along with a
//  label
//
#ifndef KOKKOS_ENABLE_HBWSPACE
namespace Kokkos {
namespace Experimental {
class HBWSpace;
}  // namespace Experimental
}  // namespace Kokkos
DISABLE_TYPE(Kokkos::Experimental::HBWSpace)
#endif

#if !defined(KOKKOS_ENABLE_OPENMPTARGET)
namespace Kokkos {
namespace Experimental {
class OpenMPTargetSpace;
}  // namespace Experimental
}  // namespace Kokkos
DISABLE_TYPE(Kokkos::Experimental::OpenMPTargetSpace)
#endif

#if !defined(KOKKOS_ENABLE_ROCM)
namespace Kokkos {
namespace Experimental {
class ROCmSpace;
}  // namespace Experimental
}  // namespace Kokkos
DISABLE_TYPE(Kokkos::Experimental::ROCmSpace)
#endif

#if !defined(KOKKOS_ENABLE_HIP)
namespace Kokkos {
namespace Experimental {
class HIPSpace;
}  // namespace Experimental
}  // namespace Kokkos
DISABLE_TYPE(Kokkos::Experimental::HIPSpace)
#endif

#if !defined(KOKKOS_ENABLE_CUDA)
namespace Kokkos {
class CudaSpace;
}  // namespace Kokkos
DISABLE_TYPE(Kokkos::CudaSpace)
#endif

/// \enum KokkosViewSpace
/// \brief An enumeration identifying all the memory spaces for a view
enum KokkosViewSpace {
  Host = 0,
  Anonymous,
  HBW,
  OpenMPTarget,
  ROCm,
  HIP,
  Cuda,
  ViewSpacesEnd
};

/// \class ViewSpaceSpecialization
/// \brief Maps a \ref KokkosViewSpace enumeration to a type a label
template <size_t SpaceT>
struct ViewSpaceSpecialization;

#define VIEW_SPACE(ENUM_ID, VIEW_SPACE, LABEL)   \
  template <>                                    \
  struct ViewSpaceSpecialization<ENUM_ID> {      \
    using type = VIEW_SPACE;                     \
    static std::string label() { return LABEL; } \
  };

VIEW_SPACE(Host, Kokkos::HostSpace, "HostSpace")
VIEW_SPACE(Anonymous, Kokkos::AnonymousSpace, "AnonymousSpace")
VIEW_SPACE(HBW, Kokkos::Experimental::HBWSpace, "HBWSpace")
VIEW_SPACE(OpenMPTarget, Kokkos::Experimental::OpenMPTargetSpace,
           "OpenMPTargetSpace")
VIEW_SPACE(ROCm, Kokkos::Experimental::ROCmSpace, "ROCmSpace")
VIEW_SPACE(HIP, Kokkos::Experimental::HIPSpace, "HIPSpace")
VIEW_SPACE(Cuda, Kokkos::CudaSpace, "CudaSpace")

template <size_t Idx>
using space_t = typename ViewSpaceSpecialization<Idx>::type;

//--------------------------------------------------------------------------------------//
