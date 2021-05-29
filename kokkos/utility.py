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
from . import libpykokkos as lib

__author__ = "Jonathan R. Madsen"
__copyright__ = (
    "Copyright 2020, National Technology & Engineering Solutions of Sandia, LLC (NTESS)"
)
__credits__ = ["Kokkos"]
__license__ = "BSD-3"
__version__ = "3.1.1"
__maintainer__ = "Jonathan R. Madsen"
__email__ = "jrmadsen@lbl.gov"
__status__ = "Development"


def array(
    shape_label_or_array,
    shape=None,
    label=None,
    array=None,
    dtype=lib.double,
    space=lib.HostSpace,
    layout=lib.LayoutRight,
    trait=lib.Managed,
    dynamic=False,
    order=None,
):
    # support non-labeled variant
    if isinstance(shape_label_or_array, str) and label is None:
        label = f"{shape_label_or_array}"
    elif isinstance(shape_label_or_array, (list, tuple)) and shape is None:
        shape = list(shape_label_or_array)[:]
    elif array is None:
        array = shape_label_or_array
        shape = array.shape
        try:
            _order = array.order
            if _order == "C":
                layout = lib.LayoutRight
            elif _order == "F":
                layout = lib.LayoutLeft
        except AttributeError:
            pass

    # layout was specified via numpy "order" field
    if order is not None and layout == lib.LayoutRight and isinstance(order, str):
        if order.upper() == "C":
            layout = lib.LayoutRight
        elif order.upper() == "F":
            layout = lib.LayoutLeft

    _prefix = "KokkosView" if not dynamic else "KokkosDynRankView"
    _space = lib.get_memory_space(space)
    _dtype = lib.get_dtype(dtype)
    _layout = lib.get_layout(layout)
    _name = None
    _label = None
    _ndim = len(shape)

    if dynamic:
        _dtype_str = "{}".format(_dtype)
    else:
        _dtype_str = "{}{}".format(_dtype, "*" * _ndim)

    _name = f"{_prefix}_{_dtype}_{_space}_{_layout}"
    _label = f"{_prefix}<{_dtype_str}, {_layout}, {_space}"

    # handle the trait argument
    if trait is not None:
        _trait = lib.get_memory_trait(trait)
        if _trait not in ("Managed", "Unmanaged"):
            _name = f"{_name}_{_trait}"
            _label = f"{_label}, {_trait}"

    # if fixed view
    if not dynamic:
        _name = f"{_name}_{_ndim}"

    # if a label was not provided
    if label is None:
        label = f"{_label}>"

    return getattr(lib, _name)(label if array is None else array, shape)


def unmanaged_array(*_args, **_kwargs):
    """This is kept for backwards compatibility"""
    return array(*_args, **_kwargs)


def convert_dtype(_dtype, _module=None):
    """Converts kokkos data types into numpy dtype"""
    if isinstance(_dtype, (str, int)):
        _true_dtype = lib.get_dtype(_dtype)
    else:
        _true_dtype = lib.get_dtype(int(_dtype))
    if _module is None:
        import numpy as np

        _module = np
    return getattr(_module, _true_dtype)


def create_mirror(dst, src):
    """Performs Kokkos::create_mirror"""
    return dst.create_mirror(src)


def create_mirror_view(dst, src):
    """Performs Kokkos::create_mirror_view"""
    return dst.create_mirror_view(src)


def deep_copy(dst, src):
    """Performs Kokkos::deep_copy"""
    return dst.deep_copy(src)
