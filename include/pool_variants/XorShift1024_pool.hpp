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

#include "common.hpp"
#include "concepts.hpp"
#include "pools.hpp"
#include "traits.hpp"

namespace Space {
namespace SpaceDim {

template <size_t SpaceIdx>
void generate_XorShift1024_pool_variant(py::module &_mod) {
  using space_spec_t  = MemorySpaceSpecialization<SpaceIdx>;
  using Sp            = typename space_spec_t::type;
  using UniformT      = Kokkos::Random_XorShift1024_Pool<Sp>;

  auto name = join("_", "KokkosXorShift1024Pool", space_spec_t::label());

  Common::generate_pool<UniformT, Sp>(_mod, name, demangle<UniformT>());

}
}

template <size_t SpaceIdx>
void generate_XorShift1024_pool_variant(
    py::module &,
    std::enable_if_t<!is_available<memory_space_t<SpaceIdx>>::value, int> = 0) {
}

template <size_t SpaceIdx>
void generate_XorShift1024_pool_variant(
    py::module &_mod,
    std::enable_if_t<is_available<memory_space_t<SpaceIdx>>::value, int> = 0) {
  // FOLD_EXPRESSION(
  //     SpaceDim::generate_XorShift1024_pool_variant<SpaceIdx>(_mod));

    SpaceDim::generate_XorShift1024_pool_variant<SpaceIdx>(_mod);
}
}

namespace {
// generate data-type, memory-space buffers for concrete dimension
template <size_t... SpaceIdx>
void generate_XorShift1024_pool_variant(py::module &_mod,
                                    std::index_sequence<SpaceIdx...>) {
  FOLD_EXPRESSION(Space::generate_XorShift1024_pool_variant<SpaceIdx>(_mod));
}
}  // namespace
