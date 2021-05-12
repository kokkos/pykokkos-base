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
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#if defined(__GNUC__)
#pragma GCC diagnostic pop
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
#include <cxxabi.h>
#endif

//--------------------------------------------------------------------------------------//

template <bool B, typename T = int>
using enable_if_t = typename std::enable_if<B, T>::type;

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
