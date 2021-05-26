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

/// \enum KokkosExecutionSpace
/// \brief An enumeration identifying all the device execution spaces
enum KokkosExecutionSpace {
  Serial_Backend = 0,
  Threads_Backend,
  OpenMP_Backend,
  Cuda_Backend,
  HPX_Backend,
  HIP_Backend,
  SYCL_Backend,
  OpenMPTarget_Backend,
  ExecutionSpacesEnd
};

/// \enum KokkosMemorySpace
/// \brief An enumeration identifying all the memory spaces for a view
enum KokkosMemorySpace {
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
  MemorySpacesEnd
};

/// \enum KokkosMemoryLayoutType
/// \brief An enumeration identifying all the layouts for a view
enum KokkosMemoryLayoutType { Right = 0, Left, Stride, MemoryLayoutEnd };

/// \enum KokkosMemoryTrait
/// \brief An enumeration identifying all the memory traits for a view
enum KokkosMemoryTrait {
  Managed = 0,
  Unmanaged,
  Atomic,
  RandomAccess,
  Restrict,
  MemoryTraitEnd
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
struct ExecutionSpaceSpecialization;

template <typename Tp>
struct ExecutionSpaceIndex;

template <size_t SpaceT>
struct MemorySpaceSpecialization;

template <typename Tp>
struct MemorySpaceIndex;

template <size_t DataT>
struct MemoryLayoutSpecialization;

template <typename Tp>
struct MemoryLayoutIndex;

template <size_t DataT>
struct MemoryTraitSpecialization;

template <typename Tp>
struct MemoryTraitIndex;

template <size_t Idx>
using execution_space_t = typename ExecutionSpaceSpecialization<Idx>::type;

template <size_t Idx>
using memory_space_t = typename MemorySpaceSpecialization<Idx>::type;

template <size_t Idx>
using memory_layout_t = typename MemoryLayoutSpecialization<Idx>::type;

template <size_t Idx>
using memory_trait_t = typename MemoryTraitSpecialization<Idx>::type;

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

#define VIEW_DATA_TYPE(TYPE, ENUM_ID, ...)                        \
  template <>                                                     \
  struct ViewDataTypeSpecialization<ENUM_ID> {                    \
    using type = TYPE;                                            \
    static std::string label() { GET_FIRST_STRING(__VA_ARGS__); } \
    static const auto& labels() { GET_STRING_SET(__VA_ARGS__); }  \
  };

#define EXECUTION_SPACE_IDX(EXECUTION_SPACE, ENUM_ID) \
  template <>                                         \
  struct ExecutionSpaceIndex<EXECUTION_SPACE> {       \
    static constexpr auto value = ENUM_ID;            \
  };

#define MEMORY_SPACE_IDX(SPACE, ENUM_ID)   \
  template <>                              \
  struct MemorySpaceIndex<SPACE> {         \
    static constexpr auto value = ENUM_ID; \
  };

#define MEMORY_LAYOUT_IDX(LAYOUT, ENUM_ID) \
  template <>                              \
  struct MemoryLayoutIndex<LAYOUT> {       \
    static constexpr auto value = ENUM_ID; \
  };

#define MEMORY_TRAIT_IDX(TRAIT, ENUM_ID)   \
  template <>                              \
  struct MemoryTraitIndex<TRAIT> {         \
    static constexpr auto value = ENUM_ID; \
  };

#define EXECUTION_SPACE(SPACE, ENUM_ID, ...)                      \
  template <>                                                     \
  struct ExecutionSpaceSpecialization<ENUM_ID> {                  \
    using type = SPACE;                                           \
    static std::string label() { GET_FIRST_STRING(__VA_ARGS__); } \
    static const auto& labels() { GET_STRING_SET(__VA_ARGS__); }  \
  };                                                              \
  EXECUTION_SPACE_IDX(SPACE, ENUM_ID)

#define MEMORY_SPACE(SPACE, ENUM_ID, ...)                         \
  template <>                                                     \
  struct MemorySpaceSpecialization<ENUM_ID> {                     \
    using type = SPACE;                                           \
    static std::string label() { GET_FIRST_STRING(__VA_ARGS__); } \
    static const auto& labels() { GET_STRING_SET(__VA_ARGS__); }  \
  };                                                              \
  MEMORY_SPACE_IDX(SPACE, ENUM_ID)

#define MEMORY_LAYOUT(LAYOUT, ENUM_ID, LABEL)                 \
  template <>                                                 \
  struct MemoryLayoutSpecialization<ENUM_ID> {                \
    using type = LAYOUT;                                      \
    static std::string label() { return LABEL; }              \
    static std::set<std::string> labels() { return {LABEL}; } \
  };                                                          \
  MEMORY_LAYOUT_IDX(LAYOUT, ENUM_ID)

#define MEMORY_TRAIT(TRAIT, ENUM_ID, LABEL)                   \
  template <>                                                 \
  struct MemoryTraitSpecialization<ENUM_ID> {                 \
    using type = TRAIT;                                       \
    static std::string label() { return LABEL; }              \
    static std::set<std::string> labels() { return {LABEL}; } \
  };                                                          \
  MEMORY_TRAIT_IDX(TRAIT, ENUM_ID)
