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

#include "common.hpp"

#include <pybind11/operators.h>
#include <Kokkos_Core.hpp>

//----------------------------------------------------------------------------//
//
//        The Kokkos::complex dtypes
//
//----------------------------------------------------------------------------//

template <typename Tp>
void generate_complex_dtype(py::module& kokkos, const std::string& _name) {
  using ComplexTp = Kokkos::complex<Tp>;

  py::class_<ComplexTp>(kokkos, _name.c_str())
      .def(py::init<Tp>())      // Constructor for real part only
      .def(py::init<Tp, Tp>())  // Constructor for real and imaginary parts
      .def("imag_mutable", py::overload_cast<>(&ComplexTp::imag))
      .def("imag_const", py::overload_cast<>(&ComplexTp::imag, py::const_))
      .def("imag_set", py::overload_cast<Tp>(&ComplexTp::imag))
      .def("real_mutable", py::overload_cast<>(&ComplexTp::real))
      .def("real_const", py::overload_cast<>(&ComplexTp::real, py::const_))
      .def("real_set", py::overload_cast<Tp>(&ComplexTp::real))
      .def(py::self += py::self)
      .def(py::self += Tp())
      .def(py::self -= py::self)
      .def(py::self -= Tp())
      .def(py::self *= py::self)
      .def(py::self *= Tp())
      .def(py::self /= py::self)
      .def(py::self /= Tp());
}

void generate_complex_dtypes(py::module& kokkos) {
  generate_complex_dtype<float>(kokkos, "complex_float32");
  generate_complex_dtype<double>(kokkos, "complex_float64");
}