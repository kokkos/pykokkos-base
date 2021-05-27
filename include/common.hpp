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

#if defined(__GNUC__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#if defined(__GNUC__)
#  pragma GCC diagnostic pop
#endif

namespace py = pybind11;

#include <array>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <typeinfo>

#if defined(ENABLE_DEMANGLE)
#  include <cxxabi.h>
#endif

//--------------------------------------------------------------------------------------//

template <typename...>
struct type_list {};

//--------------------------------------------------------------------------------------//

// default definition
template <typename... T>
struct concat {
  using type = type_list<T...>;
};

// append to type_list
template <typename... T, typename... Tail>
struct concat<type_list<T...>, Tail...> : concat<T..., Tail...> {};

// combine consecutive type_lists
template <typename... T, typename... U, typename... Tail>
struct concat<type_list<T...>, type_list<U...>, Tail...>
    : concat<type_list<T..., U...>, Tail...> {};

template <typename... T>
using concat_t = typename concat<T...>::type;

template <template <typename> class PredicateT, bool ValueT, typename... T>
struct gather {
  using type = concat_t<std::conditional_t<PredicateT<T>::value == ValueT,
                                           concat_t<T>, type_list<>>...>;
};

//--------------------------------------------------------------------------------------//

template <typename In, typename Out>
struct convert {
  using type = Out;
};

template <template <typename...> class InTuple, typename... In,
          template <typename...> class OutTuple, typename... Out>
struct convert<InTuple<In...>, OutTuple<Out...>> {
  using type = OutTuple<Out..., In...>;
};

//--------------------------------------------------------------------------------------//

template <typename T, typename U>
using convert_t = typename convert<T, U>::type;

template <template <typename> class PredicateT, bool ValueT, typename... T>
using gather_t = typename gather<PredicateT, ValueT, T...>::type;

//--------------------------------------------------------------------------------------//

template <typename...>
struct concrete_view_type_list;

template <typename...>
struct dynamic_view_type_list;

template <size_t Idx>
struct index_val;

//--------------------------------------------------------------------------------------//

template <bool B, typename T = int>
using enable_if_t = typename std::enable_if<B, T>::type;

//--------------------------------------------------------------------------------------//

#define FOLD_EXPRESSION(...) \
  ::consume_parameters(::std::initializer_list<int>{(__VA_ARGS__, 0)...})

//--------------------------------------------------------------------------------------//

#if !defined(NDEBUG)
#  define DEBUG_OUTPUT true
#else
#  define DEBUG_OUTPUT false
#endif

//--------------------------------------------------------------------------------------//

#if !defined(EXPAND)
#  define EXPAND(...) __VA_ARGS__
#endif

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
  // static because a type demangle will always be the same
  static auto _val = []() {
    // wrap the type in type_list and then extract ... from type_list<...>
    auto _tmp = demangle(typeid(type_list<Tp>).name());
    auto _key = std::string{"type_list"};
    auto _idx = _tmp.find(_key);
    _idx      = _tmp.find("<", _idx);
    _tmp      = _tmp.substr(_idx + 1);
    _idx      = _tmp.find_last_of(">");
    _tmp      = _tmp.substr(0, _idx);
    // strip trailing whitespaces
    while ((_idx = _tmp.find_last_of(" ")) == _tmp.length() - 1)
      _tmp = _tmp.substr(0, _idx);
    return _tmp;
  }();
  return _val;
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
//  this is used to mark template parameters as implicit
//
template <typename Tp>
struct is_implicit : std::false_type {};

#define ENABLE_IMPLICIT(TYPE) \
  template <>                 \
  struct is_implicit<TYPE> : std::true_type {};

template <typename Tp>
struct is_memory_traits : std::false_type {};

namespace Kokkos {
//
template <unsigned T>
struct MemoryTraits;
//
template <typename DataType, class... Properties>
class DynRankView;  // forward declare
//
template <typename DataType, class... Properties>
class View;  // forward declare
//
template <class ExecutionSpace, class MemorySpace>
struct Device;
}  // namespace Kokkos

template <unsigned T>
struct is_memory_traits<Kokkos::MemoryTraits<T>> : std::true_type {};

//--------------------------------------------------------------------------------------//
//  this is used to convert Kokkos::Device<ExecSpace, MemSpace> to MemSpace
//
template <typename Tp>
struct remove_device {
  using type = Tp;
};

template <typename ExecT, typename MemT>
struct remove_device<Kokkos::Device<ExecT, MemT>> {
  using type = MemT;
};

template <typename Tp>
using remove_device_t = typename remove_device<Tp>::type;

//--------------------------------------------------------------------------------------//
//  this is used to get the view type
//
template <typename, typename...>
struct view_type;

template <template <typename...> class ViewT, typename ValueT,
          typename... Types>
struct view_type<ViewT<ValueT>, type_list<Types...>> {
  using type = ViewT<ValueT, remove_device_t<Types>...>;
};

template <template <typename...> class ViewT, typename ValueT,
          typename... Types>
struct view_type<ViewT<ValueT>, Types...>
    : view_type<ViewT<ValueT>, gather_t<is_implicit, false, Types...>> {};

template <template <typename...> class ViewT, typename ValueT,
          typename... Types>
struct view_type<ViewT<ValueT, Types...>>
    : view_type<ViewT<ValueT>, gather_t<is_implicit, false, Types...>> {};

template <typename... T>
using view_type_t = typename view_type<T...>::type;

//--------------------------------------------------------------------------------------//
//  this is used to get the deep copy view type
//
template <typename, typename...>
struct deep_copy_view_type;

template <template <typename...> class ViewT, typename ValueT,
          typename... Types>
struct deep_copy_view_type<ViewT<ValueT>, type_list<Types...>> {
  using type = ViewT<ValueT, Types...>;
};

template <template <typename...> class ViewT, typename ValueT,
          typename... Types>
struct deep_copy_view_type<ViewT<ValueT>, Types...>
    : deep_copy_view_type<ViewT<ValueT>,
                          gather_t<is_memory_traits, false, Types...>> {};

template <template <typename...> class ViewT, typename ValueT,
          typename... Types>
struct deep_copy_view_type<ViewT<ValueT, Types...>>
    : deep_copy_view_type<ViewT<ValueT>,
                          gather_t<is_memory_traits, false, Types...>> {};

template <typename... T>
using deep_copy_view_type_t = typename deep_copy_view_type<T...>::type;

//--------------------------------------------------------------------------------------//
//  this is used to get the deep copy view type
//
template <typename, typename...>
struct uniform_view_type;

template <typename ValueT, typename... Types>
struct uniform_view_type<Kokkos::View<ValueT, Types...>> {
  using type =
      view_type_t<typename Kokkos::View<ValueT, Types...>::uniform_type>;
};

template <typename ValueT, typename... Types>
struct uniform_view_type<Kokkos::DynRankView<ValueT, Types...>> {
  using type = view_type_t<Kokkos::DynRankView<ValueT, Types...>>;
};

template <typename... T>
using uniform_view_type_t = typename uniform_view_type<T...>::type;
