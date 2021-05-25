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

#include "libpykokkos.hpp"

namespace Space {
namespace SpaceDim {

template <size_t DataIdx, size_t SpaceIdx, size_t DimIdx, size_t LayoutIdx, size_t TraitIdx>
void generate_concrete_view_variant(py::module &_mod) {
  using data_spec_t   = ViewDataTypeSpecialization<DataIdx>;
  using space_spec_t  = ViewSpaceSpecialization<SpaceIdx>;
  using layout_spec_t = ViewLayoutSpecialization<LayoutIdx>;
  using trait_spec_t  = ViewMemoryTraitSpecialization<TraitIdx>;
  using Tp            = typename data_spec_t::type;
  using Vp            = typename ViewDataTypeRepr<Tp, DimIdx>::type;
  using Sp            = typename space_spec_t::type;
  using Lp            = typename layout_spec_t::type;
  using Mp            = typename trait_spec_t::type;
  using View_t        = typename view_type<Kokkos::View<Vp>, Lp, Sp, Mp>::type;

  constexpr bool explicit_layout = !is_implicit<Lp>::value;
  constexpr bool explicit_trait = !is_implicit<Mp>::value;

  auto name = construct_name(
      "_", "KokkosView", data_spec_t::label(), space_spec_t::label(),
      (explicit_layout) ? layout_spec_t::label() : std::string{},
      (explicit_trait) ? trait_spec_t::label() : std::string{}, DimIdx + 1);

  auto desc =
      std::string{"Kokkos::View<"} +
      construct_name(", ", demangle<Vp>(),
                     (explicit_layout) ? demangle<Lp>() : std::string{},
                     demangle<Sp>(),
                     (explicit_trait) ? demangle<Mp>() : std::string{}) +
      ">";

  Common::generate_view<View_t, Sp, Tp, Lp, Mp, DimIdx, DimIdx>(_mod, name, desc);
}
}  // namespace SpaceDim

template <size_t LayoutIdx, size_t TraitIdx, size_t DataIdx, size_t SpaceIdx,
          size_t... DimIdx>
void generate_concrete_view_variant(
    py::module &, std::index_sequence<DimIdx...>,
    std::enable_if_t<!is_available<space_t<SpaceIdx>>::value, int> = 0) {}

template <size_t LayoutIdx, size_t TraitIdx, size_t DataIdx, size_t SpaceIdx,
          size_t... DimIdx>
void generate_concrete_view_variant(
    py::module &_mod, std::index_sequence<DimIdx...>,
    std::enable_if_t<is_available<space_t<SpaceIdx>>::value, int> = 0) {
  FOLD_EXPRESSION(
      SpaceDim::generate_concrete_view_variant<DataIdx, SpaceIdx, DimIdx,
                                              LayoutIdx, TraitIdx>(_mod));
}
}  // namespace Space

namespace variants {

// generate data-type, memory-space buffers for concrete dimension
template <size_t LayoutIdx, size_t TraitIdx, size_t DataIdx, size_t... SpaceIdx>
void generate_concrete_view_variant(py::module &_mod,
                                    std::index_sequence<SpaceIdx...>) {
  FOLD_EXPRESSION(Space::generate_concrete_view_variant<LayoutIdx, TraitIdx,
                                                        DataIdx, SpaceIdx>(
      _mod, std::make_index_sequence<ViewDataMaxDimensions>{}));
}

}  // namespace variants

namespace {
// generate data type buffers for each memory space
template <size_t LayoutIdx, size_t TraitIdx, size_t... DataIdx>
void generate_concrete_view_variant(py::module &_mod,
                                    std::index_sequence<DataIdx...>) {
  FOLD_EXPRESSION(variants::generate_concrete_view_variant<LayoutIdx, TraitIdx, DataIdx>(
      _mod, std::make_index_sequence<ViewSpacesEnd>{}));
}
}  // namespace
