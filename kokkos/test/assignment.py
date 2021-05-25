#!@PYTHON_EXECUTABLE@
# ************************************************************************
#
#                        Kokkos v. 3.0
#       Copyright (2020) National Technology & Engineering
#               Solutions of Sandia, LLC (NTESS).
#
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the Corporation nor the names of the
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Questions? Contact Christian R. Trott (crtrott@sandia.gov)
#
# ************************************************************************
#

from __future__ import absolute_import

__author__ = "Jonathan R. Madsen"
__copyright__ = (
    "Copyright 2020, National Technology & Engineering Solutions of Sandia, LLC (NTESS)"
)
__credits__ = ["Kokkos"]
__license__ = "BSD-3"
__version__ = "@PROJECT_VERSION@"
__maintainer__ = "Jonathan R. Madsen"
__email__ = "jrmadsen@lbl.gov"
__status__ = "Development"


import unittest
import numpy as np

try:
    from . import _conftest as conf
except ImportError:
    import kokkos.test._conftest as conf


class PyKokkosBaseViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        import kokkos

        kokkos.initialize()

    @classmethod
    def tearDownClass(self):
        import kokkos

        kokkos.finalize()

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def _generate(self, *_args, **_kwargs):
        import kokkos

        con_arr = None
        dyn_arr = None
        if _kwargs["trait"] == kokkos.Unmanaged:
            con_arr = np.zeros(_args[0], dtype=kokkos.convert_dtype(_kwargs["dtype"]))
            con_view = kokkos.unmanaged_array(con_arr, **_kwargs, dynamic=False)
            dyn_arr = np.zeros(_args[0], dtype=kokkos.convert_dtype(_kwargs["dtype"]))
            dyn_view = kokkos.unmanaged_array(dyn_arr, **_kwargs, dynamic=True)
        else:
            con_view = kokkos.array(*_args, **_kwargs, dynamic=False)
            dyn_view = kokkos.array(*_args, **_kwargs, dynamic=True)
        # retain con_arr and dyn_arr since python might run GC and delete them
        return [con_view, dyn_view, con_arr, dyn_arr]

    def _get_variants(self):
        import kokkos

        _variants = []
        for _dims in range(1, conf.get_max_concrete_dims()):
            _shape = []
            _idx = []
            _zeros = []
            for i in range(_dims):
                _shape.append(2)
                _zeros.append(0)
                _idx.append(1)
            for _dtype in conf.get_dtypes():
                for _space in conf.get_memory_spaces():
                    if _space == kokkos.AnonymousSpace:
                        continue
                    for _layout in conf.get_layouts():
                        for _trait in conf.get_memory_traits():
                            _variant = {}
                            _variant["dtype"] = _dtype
                            _variant["space"] = _space
                            _variant["layout"] = _layout
                            _variant["trait"] = _trait
                            _variants.append([_shape, _idx, _zeros, _variant])
        return _variants

    def test_view_access(self):
        """view_access"""

        print("")
        for itr in self._get_variants():
            _shape = itr[0]
            _idx = itr[1]
            _zeros = itr[2]
            _kwargs = itr[3]

            _data = self._generate(_shape, **_kwargs)

            print("concrete type  : {}".format(type(_data[0]).__name__))
            print("dynamic type   : {}".format(type(_data[1]).__name__))

            _data[0][_idx] = 1
            _data[1][_idx] = 2

            print("concrete zero  : {}".format(_data[0][_zeros]))
            print("concrete value : {}".format(_data[0][_idx]))
            print("dynamic zero   : {}".format(_data[1][_zeros]))
            print("dynamic value  : {}".format(_data[1][_idx]))

            self.assertEqual(_data[0][_zeros], 0)
            self.assertEqual(_data[1][_zeros], 0)
            self.assertEqual(_data[0][_idx], 1)
            self.assertEqual(_data[1][_idx], 2)

    def test_view_iadd(self):
        """view_iadd"""

        print("")
        for itr in self._get_variants():
            _shape = itr[0]
            _idx = itr[1]
            _zeros = itr[2]
            _kwargs = itr[3]

            _data = self._generate(_shape, **_kwargs)

            print("concrete type  : {}".format(type(_data[0]).__name__))
            print("dynamic type   : {}".format(type(_data[1]).__name__))

            _data[0][_idx] = 1
            _data[1][_idx] = 2

            _data[0][_idx] += 3
            _data[1][_idx] += 3

            self.assertEqual(_data[0][_zeros], 0)
            self.assertEqual(_data[1][_zeros], 0)
            self.assertEqual(_data[0][_idx], 4)
            self.assertEqual(_data[1][_idx], 5)

    def test_view_isub(self):
        """view_isub"""

        print("")
        for itr in self._get_variants():
            _shape = itr[0]
            _idx = itr[1]
            _zeros = itr[2]
            _kwargs = itr[3]

            _data = self._generate(_shape, **_kwargs)

            print("concrete type  : {}".format(type(_data[0]).__name__))
            print("dynamic type   : {}".format(type(_data[1]).__name__))

            _data[0][_idx] = 10
            _data[1][_idx] = 20

            _data[0][_idx] -= 3
            _data[1][_idx] -= 3

            self.assertEqual(_data[0][_zeros], 0)
            self.assertEqual(_data[1][_zeros], 0)
            self.assertEqual(_data[0][_idx], 7)
            self.assertEqual(_data[1][_idx], 17)

    def test_view_imul(self):
        """view_imul"""

        print("")
        for itr in self._get_variants():
            _shape = itr[0]
            _idx = itr[1]
            _zeros = itr[2]
            _kwargs = itr[3]

            _data = self._generate(_shape, **_kwargs)

            print("concrete type  : {}".format(type(_data[0]).__name__))
            print("dynamic type   : {}".format(type(_data[1]).__name__))

            _data[0][_idx] = 1
            _data[1][_idx] = 2

            _data[0][_idx] *= 3
            _data[1][_idx] *= 3

            self.assertEqual(_data[0][_zeros], 0)
            self.assertEqual(_data[1][_zeros], 0)
            self.assertEqual(_data[0][_idx], 3)
            self.assertEqual(_data[1][_idx], 6)


# main runner
def run():
    # run all tests
    unittest.main()


if __name__ == "__main__":
    import kokkos

    kokkos.initialize()
    run()
    kokkos.finalize()
