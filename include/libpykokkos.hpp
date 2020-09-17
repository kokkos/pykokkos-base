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

#include <array>
#include <cstdint>
#include <initializer_list>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>

#if defined(ENABLE_DEMANGLE)
#include <cxxabi.h>
#endif

//--------------------------------------------------------------------------------------//

#define FOLD_EXPRESSION(...) \
  ::consume_parameters(::std::initializer_list<int>{(__VA_ARGS__, 0)...})

//--------------------------------------------------------------------------------------//

template <typename... Args>
void consume_parameters(Args &&...) {}

//--------------------------------------------------------------------------------------//

inline std::string demangle(const char *_cstr) {
#if defined(ENABLE_DEMANGLE)
  // demangling a string when delimiting
  int _ret      = 0;
  char *_demang = abi::__cxa_demangle(_cstr, 0, 0, &_ret);
  if (_demang && _ret == 0)
    return std::string(const_cast<const char *>(_demang));
  else
    return _cstr;
#else
  return _cstr;
#endif
}

//--------------------------------------------------------------------------------------//

inline std::string demangle(const std::string &_str) {
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

static constexpr size_t ViewDataMaxDimensions = 8;

template <typename T, size_t Dim>
struct ViewDataTypeRepr;

#define VIEW_DATA_DIMS(NUM, ...)        \
  template <typename T>                 \
  struct ViewDataTypeRepr<T, NUM - 1> { \
    using type = __VA_ARGS__;           \
  };

VIEW_DATA_DIMS(1, T *)
VIEW_DATA_DIMS(2, T **)
VIEW_DATA_DIMS(3, T ***)
VIEW_DATA_DIMS(4, T ****)
VIEW_DATA_DIMS(5, T *****)
VIEW_DATA_DIMS(6, T ******)
VIEW_DATA_DIMS(7, T *******)
VIEW_DATA_DIMS(8, T ********)

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

enum KokkosViewLayoutType { Left = 0, Right, Stride, ViewLayoutEnd };

template <size_t DataT>
struct ViewLayoutSpecialization;

#define VIEW_LAYOUT_TYPE(ENUM_ID, DATA_TYPE, LABEL) \
  template <>                                       \
  struct ViewLayoutSpecialization<ENUM_ID> {        \
    using type = DATA_TYPE;                         \
    static std::string label() { return LABEL; }    \
  };

VIEW_LAYOUT_TYPE(Left, Kokkos::LayoutLeft, "LayoutLeft")
VIEW_LAYOUT_TYPE(Right, Kokkos::LayoutRight, "LayoutRight")
VIEW_LAYOUT_TYPE(Stride, Kokkos::LayoutStride, "LayoutStride")

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

#if !defined(KOKKOS_ENABLE_CUDA_UVM)
namespace Kokkos {
class CudaUVMSpace;
}  // namespace Kokkos
DISABLE_TYPE(Kokkos::CudaUVMSpace)
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
  CudaUVM,
  ViewSpacesEnd
};

/// \class ViewSpaceSpecialization
/// \brief Maps a \ref KokkosViewSpace enumeration to a type a label
template <size_t SpaceT>
struct ViewSpaceSpecialization;
template <typename Tp>
struct ViewSpaceIndex;

#define VIEW_SPACE(ENUM_ID, VIEW_SPACE, LABEL)   \
  template <>                                    \
  struct ViewSpaceSpecialization<ENUM_ID> {      \
    using type = VIEW_SPACE;                     \
    static std::string label() { return LABEL; } \
  };                                             \
  template <>                                    \
  struct ViewSpaceIndex<VIEW_SPACE> {            \
    static constexpr auto value = ENUM_ID;       \
  };

VIEW_SPACE(Host, Kokkos::HostSpace, "HostSpace")
VIEW_SPACE(Anonymous, Kokkos::AnonymousSpace, "AnonymousSpace")
VIEW_SPACE(HBW, Kokkos::Experimental::HBWSpace, "HBWSpace")
VIEW_SPACE(OpenMPTarget, Kokkos::Experimental::OpenMPTargetSpace,
           "OpenMPTargetSpace")
VIEW_SPACE(ROCm, Kokkos::Experimental::ROCmSpace, "ROCmSpace")
VIEW_SPACE(HIP, Kokkos::Experimental::HIPSpace, "HIPSpace")
VIEW_SPACE(Cuda, Kokkos::CudaSpace, "CudaSpace")
VIEW_SPACE(CudaUVM, Kokkos::CudaUVMSpace, "CudaUVMSpace")

template <size_t Idx>
using space_t = typename ViewSpaceSpecialization<Idx>::type;

//--------------------------------------------------------------------------------------//

template <bool B, typename T = char>
using enable_if_t = typename std::enable_if<B, T>::type;

template <typename Tp, size_t... Idx,
          typename RetT = std::array<size_t, sizeof...(Idx)>>
RetT get_extents(Tp &m, std::index_sequence<Idx...>) {
  return RetT{m.extent(Idx)...};
}

template <typename Up, size_t Idx, typename Tp>
constexpr auto get_stride(Tp &m);

template <typename Up, typename Tp, size_t... Idx,
          typename RetT = std::array<size_t, sizeof...(Idx)>>
RetT get_strides(Tp &m, std::index_sequence<Idx...>) {
  return RetT{(sizeof(Up) * m.stride(Idx))...};
}

#define GET_STRIDE(IDX_NUM)                                             \
  template <size_t Idx, typename Tp, enable_if_t<(Idx == IDX_NUM)> = 0> \
  constexpr auto get_stride(Tp &m) {                                    \
    return m.stride_##IDX_NUM();                                        \
  }

GET_STRIDE(0)
GET_STRIDE(1)
GET_STRIDE(2)
GET_STRIDE(3)
GET_STRIDE(4)
GET_STRIDE(5)
GET_STRIDE(6)
GET_STRIDE(7)

template <typename Up, typename Tp, size_t... Idx,
          typename RetT = std::array<size_t, sizeof...(Idx)>>
RetT get_stride(Tp &m, std::index_sequence<Idx...>) {
  return RetT{(sizeof(Up) * get_stride<Idx>(m))...};
}

//--------------------------------------------------------------------------------------//

namespace impl {
template <typename Tp, typename... Args>
struct get_item {
  static constexpr auto sequence = std::make_index_sequence<sizeof...(Args)>{};
  using tuple_type               = std::tuple<Args...>;

  template <size_t... Idx>
  static decltype(auto) get(Tp &_obj, tuple_type &&_args,
                            std::index_sequence<Idx...>) {
    return _obj.access(std::get<Idx>(std::forward<tuple_type>(_args))...);
  }

  static auto get() {
    return [](Tp &_obj, tuple_type _args) {
      return get(_obj, std::move(_args), sequence);
    };
  }
  template <typename Vp>
  static auto set() {
    return [](Tp &_obj, tuple_type _args, Vp _val) {
      get(_obj, std::move(_args), sequence) = _val;
    };
  }
};

template <typename Tp>
struct get_item<Tp, size_t> {
  static auto get() {
    return [](Tp &_obj, size_t _arg) { return _obj.access(_arg); };
  }
  template <typename Vp>
  static auto set() {
    return [](Tp &_obj, size_t _arg, Vp _val) { _obj.access(_arg) = _val; };
  }
};

template <typename Tp, size_t Idx>
struct get_type;

template <typename Tp>
struct get_type<Tp, 1> {
  using type = impl::get_item<Tp, size_t>;
};

template <typename Tp>
struct get_type<Tp, 2> {
  using type = impl::get_item<Tp, size_t, size_t>;
};

template <typename Tp>
struct get_type<Tp, 3> {
  using type = impl::get_item<Tp, size_t, size_t, size_t>;
};

template <typename Tp>
struct get_type<Tp, 4> {
  using type = impl::get_item<Tp, size_t, size_t, size_t, size_t>;
};

template <typename Tp>
struct get_type<Tp, 5> {
  using type = impl::get_item<Tp, size_t, size_t, size_t, size_t, size_t>;
};

template <typename Tp>
struct get_type<Tp, 6> {
  using type =
      impl::get_item<Tp, size_t, size_t, size_t, size_t, size_t, size_t>;
};

template <typename Tp>
struct get_type<Tp, 7> {
  using type = impl::get_item<Tp, size_t, size_t, size_t, size_t, size_t,
                              size_t, size_t>;
};

template <typename Tp>
struct get_type<Tp, 8> {
  using type = impl::get_item<Tp, size_t, size_t, size_t, size_t, size_t,
                              size_t, size_t, size_t>;
};

}  // namespace impl

template <typename Tp, size_t Idx>
struct get_item {
  using impl_type                = typename impl::get_type<Tp, Idx>::type;
  using array_type               = std::array<Tp, Idx>;
  static constexpr auto sequence = std::make_index_sequence<Idx>{};

  static auto get() { return impl_type::get(); }

  template <typename Vp>
  static auto set() {
    return impl_type::template set<Vp>();
  }

  template <typename Vp>
  static auto fill(Tp &_obj, Vp _val) {
    auto arr = array_type{};
    arr.fill(0);
    for (size_t i = 0; i < Idx; ++i) {
      for (auto &itr : get_extents(_obj, sequence)) {
        for (size_t j = 0; j < itr; ++j) {
          arr[i] = j;
        }
      }
    }
    return impl_type::template set<Vp>();
  }
};

//--------------------------------------------------------------------------------------//

template <typename... Args>
std::string construct_name(const std::string &delim, Args &&... args) {
  std::stringstream ss;
  FOLD_EXPRESSION(ss << delim << std::forward<Args>(args));
  return ss.str().substr(delim.length());
}

//--------------------------------------------------------------------------------------//
