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
#include <unordered_map>

namespace py = pybind11;

// this is intended for creating a generic View in Python, e.g.
//
//    view = kokkos.View(dtype=..., space=..., dims=...)
//
#if defined(EXPERIMENTAL_GENERIC_VIEW)

template <typename KeyT, typename MappedT>
using uomap_t = std::unordered_map<KeyT, MappedT>;

struct GenericView;

using caster_pair_t =
    std::pair<std::function<void(const std::type_info *&)>,
              std::function<const void *(const GenericView *)>>;

using generic_view_map_t =
    uomap_t<KokkosViewSpace,
            uomap_t<KokkosViewDataType, uomap_t<size_t, caster_pair_t>>>;

static auto &get_generic_view_map() {
  static generic_view_map_t _instance;
  return _instance;
}

// Not polymorphic: has no virtual methods
struct GenericView {
  const KokkosViewDataType dtype;
  const KokkosViewSpace space;
  const size_t dims;

 protected:
  GenericView(KokkosViewDataType _dtype, KokkosViewSpace _space, size_t _dims)
      : dtype(_dtype), space(_space), dims(_dims) {}
};

namespace pybind11 {
template <>
struct polymorphic_type_hook<GenericView> {
  static const void *get(const GenericView *src, const std::type_info *&type) {
    // note that src may be nullptr
    if (src) {
      auto &casters = get_generic_view_map()[src->space][src->dtype][src->dims];
      casters.first(type);
      return casters.second(src);
    }
    return src;
  }
};
}  // namespace pybind11

#endif

//--------------------------------------------------------------------------------------//

template <typename Tp, typename Up, size_t... Idx>
auto get_init(const std::string &lbl, const Up &arr,
              std::index_sequence<Idx...>) {
  return new Tp(lbl, static_cast<const size_t>(std::get<Idx>(arr))...);
}

template <typename Tp, size_t Idx>
auto get_init() {
  return [](std::string lbl, std::array<size_t, Idx> arr) {
    return get_init<Tp>(lbl, arr, std::make_index_sequence<Idx>{});
  };
}

template <typename Tp, size_t Idx, typename Up, typename Vp,
          enable_if_t<!std::is_same<Up, Kokkos::AnonymousSpace>::value> = 0>
auto get_init(Vp &_view) {
  _view.def(py::init(get_init<Tp, Idx>()));
}

template <typename Tp, size_t Idx, typename Up, typename Vp,
          enable_if_t<std::is_same<Up, Kokkos::AnonymousSpace>::value> = 0>
auto get_init(Vp &) {}

//--------------------------------------------------------------------------------------//

namespace Space {
namespace SpaceDim {

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
  using View_t       = Kokkos::View<Vp, Sp>;

  std::stringstream name;

  // generate a name for the class
  name << "KokkosView_" << space_spec_t::label() << "_" << data_spec_t::label()
       << "_" << (DimIdx + 1);

#if !defined(NDEBUG)
  std::stringstream desc;
  // generate the description of the class
  desc << "Kokkos::View<" << demangle<Vp>() << ", " << demangle<Sp>() << ">";
  std::cout << "Registering " << desc.str() << " as python class '"
            << name.str() << "'..." << std::endl;
#endif

  py::class_<View_t> _view(_mod, name.str().c_str(), py::buffer_protocol());
  _view.def(py::init([]() { return new View_t{}; }));
  get_init<View_t, DimIdx + 1, Sp>(_view);
  _view.def_buffer([](View_t &m) -> py::buffer_info {
    auto _extents = get_extents(m, std::make_index_sequence<DimIdx + 1>{});
    auto _strides = get_strides<Tp>(m, std::make_index_sequence<DimIdx + 1>{});
    return py::buffer_info(m.data(),    // Pointer to buffer
                           sizeof(Tp),  // Size of one scalar
                           py::format_descriptor<Tp>::format(),  // Descriptor
                           DimIdx + 1,  // Number of dimensions
                           _extents,    // Buffer dimensions
                           _strides     // Strides (in bytes) for each index
    );
  });
  _view.def_property_readonly(
      "shape",
      [](View_t &m) {
        return get_extents(m, std::make_index_sequence<DimIdx + 1>{});
      },
      "Get the shape of the array (extents)");
  _view.def("__getitem__", get_item<View_t, DimIdx + 1>::get());
  _view.def("__setitem__", get_item<View_t, DimIdx + 1>::template set<Tp>());

#if defined(EXPERIMENTAL_GENERIC_VIEW)
  auto _espace = static_cast<KokkosViewSpace>(SpaceIdx);
  auto _edata  = static_cast<KokkosViewDataType>(DataIdx);

  auto _tcast = [](const std::type_info *&tinfo) { tinfo = &typeid(View_t); };
  auto _gcast = [](const GenericView *ptr) -> const void * {
    return reinterpret_cast<const View_t *>(ptr);
  };
  get_generic_view_map()[_espace][_edata][DimIdx] =
      caster_pair_t{_tcast, _gcast};
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
  using View_t          = Kokkos::DynRankView<Vp, Sp>;

  std::stringstream name;

  // generate a name for the class
  name << "KokkosDynRankView_" << space_spec_t::label() << "_"
       << data_spec_t::label();

#if !defined(NDEBUG)
  std::stringstream desc;
  // generate the description of the class
  desc << "Kokkos::DynRankView<" << demangle<Vp>() << ", " << demangle<Sp>()
       << ">";
  std::cout << "Registering " << desc.str() << " as python class '"
            << name.str() << "'..." << std::endl;
#endif

  py::class_<View_t> _view(_mod, name.str().c_str(), py::buffer_protocol());
  _view.def(py::init([]() { return new View_t{}; }));
  get_init<View_t, DimIdx, Sp>(_view);
  _view.def_buffer([](View_t &m) -> py::buffer_info {
    auto _extents = get_extents(m, std::make_index_sequence<DimIdx>{});
    auto _strides = get_stride<Tp>(m, std::make_index_sequence<DimIdx>{});
    return py::buffer_info(m.data(),    // Pointer to buffer
                           sizeof(Tp),  // Size of one scalar
                           py::format_descriptor<Tp>::format(),  // Descriptor
                           DimIdx + 1,  // Number of dimensions
                           _extents,    // Buffer dimensions
                           _strides     // Strides (in bytes) for each index
    );
  });
  _view.def_property_readonly(
      "shape",
      [](View_t &m) {
        return get_extents(m, std::make_index_sequence<DimIdx>{});
      },
      "Get the shape of the array (extents)");
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
    char** _argv    = new char*[argv.size()];
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
  py::enum_<KokkosViewDataType> _dtype(kokkos, "type", "View data types");
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

  // generate buffers for all the data types
  generate_concrete_view(kokkos, std::make_index_sequence<ViewDataTypesEnd>{});
  generate_dynamic_view(kokkos, std::make_index_sequence<ViewDataTypesEnd>{});
}
