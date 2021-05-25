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

    def test_view_assignment(self):
        """view_assignment"""
        import kokkos

        def _generate(*_args, **_kwargs):
            if _kwargs["trait"] == kokkos.Unmanaged:
                arr = np.zeros(_args[0], dtype=kokkos.convert_dtype(_kwargs["dtype"]))
                con_view = kokkos.unmanaged_array(arr[:], **_kwargs, dynamic=False)
                dyn_view = kokkos.unmanaged_array(arr[:], **_kwargs, dynamic=True)
            else:
                con_view = kokkos.array(*_args, **_kwargs, dynamic=False)
                dyn_view = kokkos.array(*_args, **_kwargs, dynamic=True)
            return [con_view, dyn_view]

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
                            if _trait == kokkos.Atomic:
                                continue
                            _data = _generate(
                                _shape,
                                dtype=_dtype,
                                space=_space,
                                layout=_layout,
                                trait=_trait,
                            )
                            print("concrete type  : {}".format(type(_data[0]).__name__))
                            print("dynamic type   : {}".format(type(_data[1]).__name__))

                            _data[0][_idx] = 1
                            print("concrete zero  : {}".format(_data[0][_zeros]))
                            print("concrete value : {}".format(_data[0][_idx]))
                            self.assertEqual(_data[0][_zeros], 0)
                            self.assertEqual(_data[0][_idx], 1)

                            _data[1][_idx] = 2
                            print("dynamic zero   : {}".format(_data[1][_zeros]))
                            print("dynamic value  : {}".format(_data[1][_idx]))
                            self.assertEqual(_data[1][_zeros], 0)
                            self.assertEqual(_data[1][_idx], 2)


# main runner
def run():
    # run all tests
    unittest.main()


if __name__ == "__main__":
    import kokkos

    kokkos.initialize()
    run()
    kokkos.finalize()
