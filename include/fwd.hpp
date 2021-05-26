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

#include <set>
#include <string>

//--------------------------------------------------------------------------------------//
//
//                                   Execution Spaces
//
//--------------------------------------------------------------------------------------//

namespace Kokkos {
class Serial;
class Threads;
class OpenMP;
class Cuda;
namespace Experimental {
class HIP;
class HPX;
class OpenMPTarget;
class SYCL;
}  // namespace Experimental
}  // namespace Kokkos

//--------------------------------------------------------------------------------------//
//
//                                   Memory spaces
//
//--------------------------------------------------------------------------------------//

namespace Kokkos {
class HostSpace;
class AnonymousSpace;
class CudaSpace;
class CudaUVMSpace;
class CudaHostPinnedSpace;
// experimental spaces
namespace Experimental {
class HBWSpace;
class HIPSpace;
class HIPHostPinnedSpace;
class OpenMPTargetSpace;
class SYCLSharedUSMSpace;
class SYCLDeviceUSMSpace;
}  // namespace Experimental
}  // namespace Kokkos

//--------------------------------------------------------------------------------------//
//
//                                  Enumerations
//
//--------------------------------------------------------------------------------------//

enum KokkosViewDataType {
  Int16 = 0,
  Int32,
  Int64,
  Uint16,
  Uint32,
  Uint64,
  Float32,
  Float64,
  ViewDataTypesEnd
};

/// \enum KokkosViewSpace
/// \brief An enumeration identifying all the memory spaces for a view
enum KokkosViewSpace {
  HostSpace = 0,
  AnonymousSpace,
  CudaSpace,
  CudaUVMSpace,
  CudaHostPinnedSpace,
  HBWSpace,
  HIPSpace,
  HIPHostPinnedSpace,
  OpenMPTargetSpace,
  SYCLSharedUSMSpace,
  SYCLDeviceUSMSpace,
  ViewSpacesEnd
};

/// \enum KokkosViewLayoutType
/// \brief An enumeration identifying all the layouts for a view
enum KokkosViewLayoutType { Right = 0, Left, Stride, ViewLayoutEnd };

/// \enum KokkosViewMemoryTrait
/// \brief An enumeration identifying all the memory traits for a view
enum KokkosViewMemoryTrait {
  Managed = 0,
  Unmanaged,
  Atomic,
  RandomAccess,
  Restrict,
  ViewMemoryTraitEnd
};

//--------------------------------------------------------------------------------------//
//
//                              Trait specialization types
//
//--------------------------------------------------------------------------------------//

static constexpr size_t ViewDataMaxDimensions = 8;

template <typename T, size_t Dim>
struct ViewDataTypeRepr;

template <size_t DataT>
struct ViewDataTypeSpecialization;

template <size_t SpaceT>
struct ViewSpaceSpecialization;

template <typename Tp>
struct ViewSpaceIndex;

template <size_t DataT>
struct ViewLayoutSpecialization;

template <typename Tp>
struct ViewLayoutIndex;

template <size_t DataT>
struct ViewMemoryTraitSpecialization;

template <typename Tp>
struct ViewMemoryTraitIndex;

template <size_t Idx>
using space_t = typename ViewSpaceSpecialization<Idx>::type;

template <size_t Idx>
using layout_t = typename ViewLayoutSpecialization<Idx>::type;

template <size_t Idx>
using memory_trait_t = typename ViewMemoryTraitSpecialization<Idx>::type;

//--------------------------------------------------------------------------------------//
//
//                                      macros
//
//--------------------------------------------------------------------------------------//

#define GET_FIRST_STRING(...)                         \
  static std::string _value = []() {                  \
    return std::get<0>(std::make_tuple(__VA_ARGS__)); \
  }();                                                \
  return _value

#define GET_STRING_SET(...)                               \
  static auto _value = []() {                             \
    auto _ret = std::set<std::string>{};                  \
    for (auto itr : std::set<std::string>{__VA_ARGS__}) { \
      if (!itr.empty()) {                                 \
        _ret.insert(itr);                                 \
      }                                                   \
    }                                                     \
    return _ret;                                          \
  }();                                                    \
  return _value

#define VIEW_DATA_DIMS(NUM, ...)        \
  template <typename T>                 \
  struct ViewDataTypeRepr<T, NUM - 1> { \
    using type = __VA_ARGS__;           \
  };

#define VIEW_DATA_TYPE(ENUM_ID, DATA_TYPE, ...)                   \
  template <>                                                     \
  struct ViewDataTypeSpecialization<ENUM_ID> {                    \
    using type = DATA_TYPE;                                       \
    static std::string label() { GET_FIRST_STRING(__VA_ARGS__); } \
    static const auto& labels() { GET_STRING_SET(__VA_ARGS__); }  \
  };

#define VIEW_SPACE_IDX(VIEW_SPACE, ENUM_ID) \
  template <>                               \
  struct ViewSpaceIndex<VIEW_SPACE> {       \
    static constexpr auto value = ENUM_ID;  \
  };

#define VIEW_LAYOUT_IDX(VIEW_LAYOUT, ENUM_ID) \
  template <>                                 \
  struct ViewLayoutIndex<VIEW_LAYOUT> {       \
    static constexpr auto value = ENUM_ID;    \
  };

#define VIEW_MEMORY_TRAIT_IDX(VIEW_MEMORY_TRAIT, ENUM_ID) \
  template <>                                             \
  struct ViewMemoryTraitIndex<VIEW_MEMORY_TRAIT> {        \
    static constexpr auto value = ENUM_ID;                \
  };

#define VIEW_SPACE(VIEW_SPACE, ENUM_ID, ...)                      \
  template <>                                                     \
  struct ViewSpaceSpecialization<ENUM_ID> {                       \
    using type = VIEW_SPACE;                                      \
    static std::string label() { GET_FIRST_STRING(__VA_ARGS__); } \
    static const auto& labels() { GET_STRING_SET(__VA_ARGS__); }  \
  };                                                              \
  VIEW_SPACE_IDX(VIEW_SPACE, ENUM_ID)

#define VIEW_LAYOUT_TYPE(ENUM_ID, LAYOUT_TYPE, LABEL)         \
  template <>                                                 \
  struct ViewLayoutSpecialization<ENUM_ID> {                  \
    using type = LAYOUT_TYPE;                                 \
    static std::string label() { return LABEL; }              \
    static std::set<std::string> labels() { return {LABEL}; } \
  };                                                          \
  VIEW_LAYOUT_IDX(LAYOUT_TYPE, ENUM_ID)

#define VIEW_MEMORY_TRAIT(ENUM_ID, MEMORY_TRAIT, LABEL)       \
  template <>                                                 \
  struct ViewMemoryTraitSpecialization<ENUM_ID> {             \
    using type = MEMORY_TRAIT;                                \
    static std::string label() { return LABEL; }              \
    static std::set<std::string> labels() { return {LABEL}; } \
  };                                                          \
  VIEW_MEMORY_TRAIT_IDX(MEMORY_TRAIT, ENUM_ID)
