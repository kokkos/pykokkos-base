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

#include "libpykokkos.hpp"

//--------------------------------------------------------------------------------------//

template <template <size_t> class SpecT, size_t Idx, size_t... Tail,
          enable_if_t<(sizeof...(Tail) == 0)> = 0>
auto get_available(size_t i, std::index_sequence<Idx, Tail...>) {
  if (i == Idx) return is_available<typename SpecT<Idx>::type>::value;
  std::stringstream ss;
  ss << "Error! Index " << i << " does not match any known enumeration type";
  throw std::runtime_error(ss.str());
}

template <template <size_t> class SpecT, size_t Idx, size_t... Tail,
          enable_if_t<(sizeof...(Tail) > 0)> = 0>
auto get_available(size_t i, std::index_sequence<Idx, Tail...>) {
  if (i == Idx)
    return is_available<typename SpecT<Idx>::type>::value;
  else
    return get_available<SpecT>(i, std::index_sequence<Tail...>{});
}

//--------------------------------------------------------------------------------------//

template <template <size_t> class SpecT, size_t Idx, size_t... Tail,
          enable_if_t<(sizeof...(Tail) == 0)> = 0>
auto get_available(const std::string &str, std::index_sequence<Idx, Tail...>) {
  if (str == SpecT<Idx>::label())
    return is_available<typename SpecT<Idx>::type>::value;
  std::stringstream ss;
  ss << "Error! Identifier " << str
     << " does not match any known enumeration type";
  throw std::runtime_error(ss.str());
}

template <template <size_t> class SpecT, size_t Idx, size_t... Tail,
          enable_if_t<(sizeof...(Tail) > 0)> = 0>
auto get_available(const std::string &str, std::index_sequence<Idx, Tail...>) {
  if (str == SpecT<Idx>::label())
    return is_available<typename SpecT<Idx>::type>::value;
  else
    return get_available<SpecT>(str, std::index_sequence<Tail...>{});
}

//--------------------------------------------------------------------------------------//

void generate_available(py::module &kokkos) {
  kokkos.attr("max_concrete_rank") = ViewDataMaxDimensions;

  //----------------------------------------------------------------------------//
  //
  //                                memory spaces
  //
  //----------------------------------------------------------------------------//
  auto _get_memspace_idx = [](int idx) {
    return get_available<ViewSpaceSpecialization>(
        idx, std::make_index_sequence<ViewSpacesEnd>{});
  };
  auto _get_memspace_name = [](std::string str) {
    return get_available<ViewSpaceSpecialization>(
        str, std::make_index_sequence<ViewSpacesEnd>{});
  };
  kokkos.def("get_memory_space_available", _get_memspace_name,
             "Get whether the memory space is available");
  kokkos.def("get_memory_space_available", _get_memspace_idx,
             "Get whether the memory space is available");

  //----------------------------------------------------------------------------//
  //
  //                                  layouts
  //
  //----------------------------------------------------------------------------//
  auto _get_ltype_name = [](int idx) {
    return get_available<ViewLayoutSpecialization>(
        idx, std::make_index_sequence<ViewLayoutEnd>{});
  };
  auto _get_ltype_idx = [](std::string str) {
    return get_available<ViewLayoutSpecialization>(
        str, std::make_index_sequence<ViewLayoutEnd>{});
  };
  kokkos.def("get_layout_available", _get_ltype_name,
             "Get whether the layout type is available");
  kokkos.def("get_layout_available", _get_ltype_idx,
             "Get whether the layout type is available");

  //----------------------------------------------------------------------------//
  //
  //                                memory traits
  //
  //----------------------------------------------------------------------------//
  auto _get_memtrait_name = [](int idx) {
    return get_available<ViewMemoryTraitSpecialization>(
        idx, std::make_index_sequence<ViewMemoryTraitEnd>{});
  };
  auto _get_memtype_idx = [](std::string str) {
    return get_available<ViewMemoryTraitSpecialization>(
        str, std::make_index_sequence<ViewMemoryTraitEnd>{});
  };
  kokkos.def("get_memory_trait_available", _get_memtrait_name,
             "Get whether the memory trait is available");
  kokkos.def("get_memory_trait_available", _get_memtype_idx,
             "Get whether the memory trait is available");
}
