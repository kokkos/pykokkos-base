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

namespace Space {
namespace SpaceDim {

#if defined(ENABLE_LAYOUTS)
template <size_t DataIdx, size_t SpaceIdx, size_t DimIdx, size_t LayoutIdx>
void generate_concrete_layout_view(py::module &_mod) {
  using data_spec_t   = ViewDataTypeSpecialization<DataIdx>;
  using space_spec_t  = ViewSpaceSpecialization<SpaceIdx>;
  using layout_spec_t = ViewLayoutSpecialization<LayoutIdx>;
  using Tp            = typename data_spec_t::type;
  using Vp            = typename ViewDataTypeRepr<Tp, DimIdx>::type;
  using Sp            = typename space_spec_t::type;
  using Lp            = typename layout_spec_t::type;
  using Mp            = Kokkos::MemoryTraits<0>;
  using View_t        = Kokkos::View<Vp, Lp, Sp>;

  auto name =
      construct_name("_", "KokkosView", data_spec_t::label(),
                     layout_spec_t::label(), space_spec_t::label(), DimIdx + 1);
  auto desc = construct_name("", "Kokkos::View<", demangle<Vp>(), ", ",
                             demangle<Lp>(), ", ", demangle<Sp>());

  Common::generate_view<View_t, Sp, Tp, Mp, DimIdx, DimIdx>(_mod, name, desc);
}
#endif

#if defined(ENABLE_MEMORY_TRAITS)
template <size_t DataIdx, size_t SpaceIdx, size_t DimIdx, size_t TraitIdx>
void generate_concrete_trait_view(py::module &_mod) {
  using data_spec_t  = ViewDataTypeSpecialization<DataIdx>;
  using space_spec_t = ViewSpaceSpecialization<SpaceIdx>;
  using trait_spec_t = ViewMemoryTraitSpecialization<TraitIdx>;
  using Tp           = typename data_spec_t::type;
  using Vp           = typename ViewDataTypeRepr<Tp, DimIdx>::type;
  using Sp           = typename space_spec_t::type;
  using Mp           = typename trait_spec_t::type;
  using View_t       = Kokkos::View<Vp, Sp, Mp>;

  auto name =
      construct_name("_", "KokkosView", data_spec_t::label(),
                     space_spec_t::label(), trait_spec_t::label(), DimIdx + 1);
  auto desc = construct_name("", "Kokkos::View<", demangle<Vp>(), ", ",
                             demangle<Sp>(), ", ", demangle<Mp>());

  Common::generate_view<View_t, Sp, Tp, Mp, DimIdx, DimIdx>(_mod, name, desc);
}
#endif

/// DataIdx --> data type, e.g. int
/// SpaceIdx --> the space of the view
/// DimIdx --> the dimensionality of the view, e.g. View<double*> is 0,
///   View<double**> is 1
template <size_t DataIdx, size_t SpaceIdx, size_t DimIdx>
void generate_concrete_view(py::module &_mod) {
  using data_spec_t  = ViewDataTypeSpecialization<DataIdx>;
  using space_spec_t = ViewSpaceSpecialization<SpaceIdx>;
  using Tp           = typename data_spec_t::type;
  using Vp           = typename ViewDataTypeRepr<Tp, DimIdx>::type;
  using Sp           = typename space_spec_t::type;
  using Mp           = Kokkos::MemoryTraits<0>;
  using View_t       = Kokkos::View<Vp, Sp>;

  auto name = construct_name("_", "KokkosView", data_spec_t::label(),
                             space_spec_t::label(), DimIdx + 1);
  auto desc =
      construct_name("", "Kokkos::View<", demangle<Vp>(), ", ", demangle<Sp>());

  Common::generate_view<View_t, Sp, Tp, Mp, DimIdx, DimIdx>(_mod, name, desc);
#if defined(ENABLE_LAYOUTS)
  generate_concrete_layout_view<DataIdx, SpaceIdx, DimIdx, Left>(_mod);
  // generate_concrete_layout_view<DataIdx, SpaceIdx, DimIdx, Right>(_mod);
  // generate_concrete_layout_view<DataIdx, SpaceIdx, DimIdx, Stride>(_mod);
#endif
#if defined(ENABLE_MEMORY_TRAITS)
  generate_concrete_trait_view<DataIdx, SpaceIdx, DimIdx, Unmanaged>(_mod);
  generate_concrete_trait_view<DataIdx, SpaceIdx, DimIdx, Atomic>(_mod);
  generate_concrete_trait_view<DataIdx, SpaceIdx, DimIdx, RandomAccess>(_mod);
  generate_concrete_trait_view<DataIdx, SpaceIdx, DimIdx, Restrict>(_mod);
#endif
}

//--------------------------------------------------------------------------------------//

// if the memory space is available, generate a class for it
template <size_t DataIdx, size_t SpaceIdx, size_t... DimIdx,
          std::enable_if_t<(is_available<space_t<SpaceIdx>>::value), int> = 0>
void generate_concrete_view(py::module &_mod, std::index_sequence<DimIdx...>) {
  FOLD_EXPRESSION(generate_concrete_view<DataIdx, SpaceIdx, DimIdx>(_mod));
}

// if the memory space is not available, do not generate a class for it
template <size_t DataIdx, size_t SpaceIdx, size_t... DimIdx,
          std::enable_if_t<!(is_available<space_t<SpaceIdx>>::value), int> = 0>
void generate_concrete_view(py::module &, std::index_sequence<DimIdx...>) {}
}  // namespace SpaceDim

// generate data-type, memory-space buffers for all the dimensions
template <size_t DataIdx, size_t... SpaceIdx>
void generate_concrete_view(py::module &_mod,
                            std::index_sequence<SpaceIdx...>) {
  FOLD_EXPRESSION(SpaceDim::generate_concrete_view<DataIdx, SpaceIdx>(
      _mod, std::make_index_sequence<ViewDataMaxDimensions>{}));
}

}  // namespace Space

// generate data type buffers for each memory space
template <size_t... DataIdx>
void generate_concrete_view(py::module &_mod, std::index_sequence<DataIdx...>) {
  FOLD_EXPRESSION(Space::generate_concrete_view<DataIdx>(
      _mod, std::make_index_sequence<ViewSpacesEnd>{}));
}

void generate_concrete_view(py::module &kokkos) {
  // generate buffers for all the data types
  generate_concrete_view(kokkos, std::make_index_sequence<ViewDataTypesEnd>{});
}
