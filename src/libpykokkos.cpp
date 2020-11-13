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

#include "Kokkos_Core.hpp"
#include "Kokkos_View.hpp"
#include "Kokkos_DynamicView.hpp"
#include "Kokkos_DynRankView.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdio>
#include <typeinfo>

namespace py = pybind11;

//--------------------------------------------------------------------------------------//

template <typename ViewT, typename Up, size_t... Idx>
auto get_init(const std::string &lbl, const Up &arr,
              std::index_sequence<Idx...>) {
  return new ViewT(lbl, static_cast<const size_t>(std::get<Idx>(arr))...);
}

template <typename ViewT, size_t Idx>
auto get_init() {
  return [](std::string lbl, std::array<size_t, Idx> arr) {
    return get_init<ViewT>(lbl, arr, std::make_index_sequence<Idx>{});
  };
}

template <typename ViewT, typename Up, typename Tp, size_t... Idx>
auto get_unmanaged_init(const Up &arr, const Tp data,
                        std::index_sequence<Idx...>) {
  return new ViewT(data, static_cast<const size_t>(std::get<Idx>(arr))...);
}

template <typename ViewT, size_t Idx, typename Tp>
auto get_unmanaged_init() {
  return [](py::buffer buf, std::array<size_t, Idx> arr) {
    return get_unmanaged_init<ViewT>(arr, static_cast<Tp *>(buf.request().ptr),
                                     std::make_index_sequence<Idx>{});
  };
}

// define managed init
template <typename ViewT, size_t Idx, typename Tp, typename Mp, typename Vp,
          enable_if_t<!std::is_same<
              Mp, Kokkos::MemoryTraits<Kokkos::Unmanaged>>::value> = 0>
auto get_init(Vp &_view) {
  _view.def(py::init(get_init<ViewT, Idx>()));
}

// define unmanaged init
template <typename ViewT, size_t Idx, typename Tp, typename Mp, typename Vp,
          enable_if_t<std::is_same<
              Mp, Kokkos::MemoryTraits<Kokkos::Unmanaged>>::value> = 0>
auto get_init(Vp &_view) {
  _view.def(py::init(get_unmanaged_init<ViewT, Idx, Tp>()));
}

template <typename ViewT, size_t Idx, typename Up, typename Tp, typename Mp,
          typename Vp,
          enable_if_t<!std::is_same<Up, Kokkos::AnonymousSpace>::value> = 0>
auto get_init(Vp &_view) {
  get_init<ViewT, Idx, Tp, Mp>(_view);
}

template <typename ViewT, size_t Idx, typename Up, typename Tp, typename Mp,
          typename Vp,
          enable_if_t<std::is_same<Up, Kokkos::AnonymousSpace>::value> = 0>
auto get_init(Vp &) {}

//--------------------------------------------------------------------------------------//

namespace Common {
// creates overloads for data access from python
template <typename Tp, typename View_t, size_t... Idx>
void generate_view_access(py::class_<View_t> &_view,
                          std::index_sequence<Idx...>) {
  FOLD_EXPRESSION(_view.def("__getitem__", get_item<View_t, Idx + 1>::get(),
                            "Get the element"));
  FOLD_EXPRESSION(_view.def("__setitem__",
                            get_item<View_t, Idx + 1>::template set<Tp>(),
                            "Set the element"));
}

// generic function to generate a view once the view type has been specified
template <typename View_t, typename Sp, typename Tp, typename Mp, size_t DimIdx,
          size_t... Idx>
void generate_view(py::module &_mod, const std::string &_name,
                   const std::string &_msg, size_t _ndim = DimIdx + 1) {
  bool debug = false;
#if !defined(NDEBUG)
  debug = true;
#endif

  if (debug)
    std::cout << "Registering " << _msg << " as python class '" << _name
              << "'..." << std::endl;

  // class decl
  py::class_<View_t> _view(_mod, _name.c_str(), py::buffer_protocol());

  // default initializer
  _view.def(py::init([]() { return new View_t{}; }));

  // initializer with extents
  get_init<View_t, DimIdx + 1, Sp, Tp, Mp>(_view);

  // conversion to/from numpy
  _view.def_buffer([_ndim](View_t &m) -> py::buffer_info {
    auto _extents = get_extents(m, std::make_index_sequence<DimIdx + 1>{});
    auto _strides = get_stride<Tp>(m, std::make_index_sequence<DimIdx + 1>{});
    return py::buffer_info(m.data(),    // Pointer to buffer
                           sizeof(Tp),  // Size of one scalar
                           py::format_descriptor<Tp>::format(),  // Descriptor
                           _ndim,     // Number of dimensions
                           _extents,  // Buffer dimensions
                           _strides   // Strides (in bytes) for each index
    );
  });

  // shape property
  _view.def_property_readonly(
      "shape",
      [](View_t &m) {
        return get_extents(m, std::make_index_sequence<DimIdx + 1>{});
      },
      "Get the shape of the array (extents)");

  // memory space
  _view.def_property_readonly(
      "memory_space", [](View_t &) { return ViewSpaceIndex<Sp>::value; },
      "Memory space of the view");

  // support []
  generate_view_access<Tp>(_view, std::index_sequence<Idx...>{});
}

template <typename View_t, typename Sp, typename Tp, typename Mp, size_t DimIdx,
          size_t... Idx>
void generate_view(py::module &_mod, const std::string &_name,
                   const std::string &_msg, size_t _ndim,
                   std::index_sequence<Idx...>) {
  generate_view<View_t, Sp, Tp, Mp, DimIdx, Idx...>(_mod, _name, _msg, _ndim);
}
}  // namespace Common
//
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
  using Mp           = typename trait_spec_t::trait;
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

#if defined(ENABLE_LAYOUTS)
template <size_t DataIdx, size_t SpaceIdx, size_t LayoutIdx>
void generate_dynamic_layout_view(py::module &_mod) {
  constexpr auto DimIdx = ViewDataMaxDimensions;
  using data_spec_t     = ViewDataTypeSpecialization<DataIdx>;
  using space_spec_t    = ViewSpaceSpecialization<SpaceIdx>;
  using layout_spec_t   = ViewLayoutSpecialization<LayoutIdx>;
  using Tp              = typename data_spec_t::type;
  using Vp              = Tp;
  using Sp              = typename space_spec_t::type;
  using Lp              = typename layout_spec_t::type;
  using Mp              = Kokkos::MemoryTraits<0>;
  using View_t          = Kokkos::DynRankView<Vp, Lp, Sp>;

  auto name = construct_name("_", "KokkosDynRankView", data_spec_t::label(),
                             layout_spec_t::label(), space_spec_t::label());
  auto desc = construct_name("", "Kokkos::DynRankView<", demangle<Vp>(), ", ",
                             demangle<Lp>(), ", ", demangle<Sp>());

  constexpr auto nIdx = DimIdx - 1;
  Common::generate_view<View_t, Sp, Tp, Mp, nIdx>(
      _mod, name, desc, DimIdx, std::make_index_sequence<nIdx>{});
}
#endif

#if defined(ENABLE_MEMORY_TRAITS)
template <size_t DataIdx, size_t SpaceIdx, size_t TraitIdx>
void generate_dynamic_trait_view(py::module &_mod) {
  constexpr auto DimIdx = ViewDataMaxDimensions;
  using data_spec_t     = ViewDataTypeSpecialization<DataIdx>;
  using space_spec_t    = ViewSpaceSpecialization<SpaceIdx>;
  using trait_spec_t    = ViewMemoryTraitSpecialization<TraitIdx>;
  using Tp              = typename data_spec_t::type;
  using Vp              = Tp;
  using Sp              = typename space_spec_t::type;
  using Mp              = typename trait_spec_t::trait;
  using View_t          = Kokkos::DynRankView<Vp, Sp, Mp>;

  auto name = construct_name("_", "KokkosDynRankView", data_spec_t::label(),
                             space_spec_t::label(), trait_spec_t::label());
  auto desc = construct_name("", "Kokkos::DynRankView<", demangle<Vp>(), ", ",
                             demangle<Sp>(), ", ", demangle<Mp>());

  constexpr auto nIdx = DimIdx - 1;
  Common::generate_view<View_t, Sp, Tp, Mp, nIdx>(
      _mod, name, desc, DimIdx, std::make_index_sequence<nIdx>{});
}
#endif

/// DataIdx --> data type, e.g. int
/// SpaceIdx --> the space of the view
template <size_t DataIdx, size_t SpaceIdx,
          std::enable_if_t<is_available<space_t<SpaceIdx>>::value, int> = 0>
void generate_dynamic_view(py::module &_mod) {
  constexpr auto DimIdx = ViewDataMaxDimensions;
  using data_spec_t     = ViewDataTypeSpecialization<DataIdx>;
  using space_spec_t    = ViewSpaceSpecialization<SpaceIdx>;
  using Tp              = typename data_spec_t::type;
  using Vp              = Tp;
  using Sp              = typename space_spec_t::type;
  using Mp              = Kokkos::MemoryTraits<0>;
  using View_t          = Kokkos::DynRankView<Vp, Sp>;

  auto name = construct_name("_", "KokkosDynRankView", data_spec_t::label(),
                             space_spec_t::label());
  auto desc = construct_name("", "Kokkos::DynRankView<", demangle<Vp>(), ", ",
                             demangle<Sp>());

  constexpr auto nIdx = DimIdx - 1;
  Common::generate_view<View_t, Sp, Tp, Mp, nIdx>(
      _mod, name, desc, DimIdx, std::make_index_sequence<nIdx>{});

#if defined(ENABLE_LAYOUTS)
  generate_dynamic_layout_view<DataIdx, SpaceIdx, Left>(_mod);
  // generate_dynamic_layout_view<DataIdx, SpaceIdx, Right>(_mod);
  // generate_dynamic_layout_view<DataIdx, SpaceIdx, Stride>(_mod);
#endif
#if defined(ENABLE_MEMORY_TRAITS)
  generate_dynamic_trait_view<DataIdx, SpaceIdx, Unmanaged>(_mod);
  generate_dynamic_trait_view<DataIdx, SpaceIdx, Atomic>(_mod);
  generate_dynamic_trait_view<DataIdx, SpaceIdx, RandomAccess>(_mod);
  generate_dynamic_trait_view<DataIdx, SpaceIdx, Restrict>(_mod);
#endif
}

template <size_t DataIdx, size_t SpaceIdx,
          std::enable_if_t<!is_available<space_t<SpaceIdx>>::value, int> = 0>
void generate_dynamic_view(py::module &) {}

// generate data-type, memory-space buffers for dynamic dimension
template <size_t DataIdx, size_t... SpaceIdx>
void generate_dynamic_view(py::module &_mod, std::index_sequence<SpaceIdx...>) {
  FOLD_EXPRESSION(generate_dynamic_view<DataIdx, SpaceIdx>(_mod));
}
}  // namespace Space

// generate data type buffers for each memory space
template <size_t... DataIdx>
void generate_concrete_view(py::module &_mod, std::index_sequence<DataIdx...>) {
  FOLD_EXPRESSION(Space::generate_concrete_view<DataIdx>(
      _mod, std::make_index_sequence<ViewSpacesEnd>{}));
}

// generate data type buffers for each memory space
template <size_t... DataIdx>
void generate_dynamic_view(py::module &_mod, std::index_sequence<DataIdx...>) {
  FOLD_EXPRESSION(Space::generate_dynamic_view<DataIdx>(
      _mod, std::make_index_sequence<ViewSpacesEnd>{}));
}

template <template <size_t> class SpecT, typename Tp, size_t... Idx>
void generate_enumeration(py::enum_<Tp> &_enum, std::index_sequence<Idx...>) {
  FOLD_EXPRESSION(
      _enum.value(SpecT<Idx>::label().c_str(), static_cast<Tp>(Idx)));
}

//--------------------------------------------------------------------------------------//

template <template <size_t> class SpecT, size_t Idx, size_t... Tail,
          enable_if_t<(sizeof...(Tail) == 0)> = 0>
auto get_enumeration(size_t i, std::index_sequence<Idx, Tail...>) {
  if (i == Idx) return SpecT<Idx>::label();
  std::stringstream ss;
  ss << "Error! Index " << i << " does not match any known enumeration type";
  throw std::runtime_error(ss.str());
}

template <template <size_t> class SpecT, size_t Idx, size_t... Tail,
          enable_if_t<(sizeof...(Tail) > 0)> = 0>
auto get_enumeration(size_t i, std::index_sequence<Idx, Tail...>) {
  if (i == Idx)
    return SpecT<Idx>::label();
  else
    return get_enumeration<SpecT>(i, std::index_sequence<Tail...>{});
}

//--------------------------------------------------------------------------------------//

template <template <size_t> class SpecT, size_t Idx, size_t... Tail,
          enable_if_t<(sizeof...(Tail) == 0)> = 0>
auto get_enumeration(const std::string &str,
                     std::index_sequence<Idx, Tail...>) {
  if (str == SpecT<Idx>::label()) return Idx;
  std::stringstream ss;
  ss << "Error! Identifier " << str
     << " does not match any known enumeration type";
  throw std::runtime_error(ss.str());
}

template <template <size_t> class SpecT, size_t Idx, size_t... Tail,
          enable_if_t<(sizeof...(Tail) > 0)> = 0>
auto get_enumeration(const std::string &str,
                     std::index_sequence<Idx, Tail...>) {
  if (str == SpecT<Idx>::label())
    return Idx;
  else
    return get_enumeration<SpecT>(str, std::index_sequence<Tail...>{});
}

//--------------------------------------------------------------------------------------//
//
//        The python module
//
//--------------------------------------------------------------------------------------//

PYBIND11_MODULE(libpykokkos, kokkos) {
  // Initialize kokkos
  auto _initialize = [&]() {
    // python system module
    py::module sys = py::module::import("sys");
    // get the arguments for python system module
    py::object args = sys.attr("argv");
    auto argv       = args.cast<py::list>();
    int _argc       = argv.size();
    char **_argv    = new char *[argv.size()];
    for (int i = 0; i < _argc; ++i)
      _argv[i] = strdup(argv[i].cast<std::string>().c_str());
    Kokkos::initialize(_argc, _argv);
    for (int i = 0; i < _argc; ++i) free(_argv[i]);
    delete[] _argv;
  };

  // Finalize kokkos
  auto _finalize = []() {
    py::module gc = py::module::import("gc");
    gc.attr("collect")();
    Kokkos::finalize();
  };

  kokkos.def("initialize", _initialize, "Initialize Kokkos");
  kokkos.def("finalize", _finalize, "Finalize Kokkos");

  // an enumeration of the data types for views
  py::enum_<KokkosViewDataType> _dtype(kokkos, "dtype", "View data types");
  generate_enumeration<ViewDataTypeSpecialization>(
      _dtype, std::make_index_sequence<ViewDataTypesEnd>{});
  _dtype.export_values();

  auto _get_dtype_name = [](int idx) {
    return get_enumeration<ViewDataTypeSpecialization>(
        idx, std::make_index_sequence<ViewDataTypesEnd>{});
  };
  auto _get_dtype_idx = [](std::string str) {
    return get_enumeration<ViewDataTypeSpecialization>(
        str, std::make_index_sequence<ViewDataTypesEnd>{});
  };
  kokkos.def("get_dtype", _get_dtype_name, "Get the data type");
  kokkos.def("get_dtype", _get_dtype_idx, "Get the data type");

  // an enumeration of the memory spaces for views
  py::enum_<KokkosViewSpace> _memspace(kokkos, "memory_space",
                                       "View memory spaces");
  generate_enumeration<ViewSpaceSpecialization>(
      _memspace, std::make_index_sequence<ViewSpacesEnd>{});
  _memspace.export_values();

  auto _get_memspace_name = [](int idx) {
    return get_enumeration<ViewSpaceSpecialization>(
        idx, std::make_index_sequence<ViewSpacesEnd>{});
  };
  auto _get_memspace_idx = [](std::string str) {
    return get_enumeration<ViewSpaceSpecialization>(
        str, std::make_index_sequence<ViewSpacesEnd>{});
  };
  kokkos.def("get_memory_space", _get_memspace_name, "Get the memory space");
  kokkos.def("get_memory_space", _get_memspace_idx, "Get the memory space");

  // an enumeration of the layout types for views
  py::enum_<KokkosViewLayoutType> _ltype(kokkos, "layout", "View layout types");
  generate_enumeration<ViewLayoutSpecialization>(
      _ltype, std::make_index_sequence<ViewLayoutEnd>{});
  _ltype.export_values();

  auto _get_ltype_name = [](int idx) {
    return get_enumeration<ViewLayoutSpecialization>(
        idx, std::make_index_sequence<ViewLayoutEnd>{});
  };
  auto _get_ltype_idx = [](std::string str) {
    return get_enumeration<ViewLayoutSpecialization>(
        str, std::make_index_sequence<ViewLayoutEnd>{});
  };
  kokkos.def("get_layout", _get_ltype_name, "Get the layout type");
  kokkos.def("get_layout", _get_ltype_idx, "Get the layout type");

  // an enumeration of the memory traits for views
  py::enum_<KokkosViewMemoryTrait> _memtrait(kokkos, "memory_trait",
                                             "View memory traits");
  generate_enumeration<ViewMemoryTraitSpecialization>(
      _memtrait, std::make_index_sequence<ViewMemoryTraitEnd>{});
  _memtrait.export_values();

  auto _get_memtrait_name = [](int idx) {
    return get_enumeration<ViewMemoryTraitSpecialization>(
        idx, std::make_index_sequence<ViewMemoryTraitEnd>{});
  };
  auto _get_memtype_idx = [](std::string str) {
    return get_enumeration<ViewMemoryTraitSpecialization>(
        str, std::make_index_sequence<ViewMemoryTraitEnd>{});
  };
  kokkos.def("get_memory_trait", _get_memtrait_name, "Get the memory trait");
  kokkos.def("get_memory_trait", _get_memtype_idx, "Get the memory trait");

  // generate buffers for all the data types
  generate_concrete_view(kokkos, std::make_index_sequence<ViewDataTypesEnd>{});
  generate_dynamic_view(kokkos, std::make_index_sequence<ViewDataTypesEnd>{});
}
