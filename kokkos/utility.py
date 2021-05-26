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
    shape_or_label,
    shape=None,
    label=None,
    dtype=lib.double,
    space=lib.HostSpace,
    layout=lib.LayoutRight,
    trait=lib.Managed,
    dynamic=False,
    order=None,
):
    # support non-labeled variant
    if not isinstance(shape_or_label, str) and shape is None:
        shape = list(shape_or_label)[:]
    elif isinstance(shape_or_label, str) and label is None:
        label = f"{shape_or_label}"

    # print("dtype = {}, space = {}".format(dtype, space))
    _prefix = "KokkosView"
    if dynamic:
        _prefix = "KokkosDynRankView"
    _space = lib.get_memory_space(space)
    _dtype = lib.get_dtype(dtype)
    _name = None
    _label = None
    _ndim = len(shape)

    if dynamic:
        _dtype_str = "{}".format(_dtype)
    else:
        _dtype_str = "{}{}".format(_dtype, "*" * _ndim)

    _name = f"{_prefix}_{_dtype}_{_space}"
    _label = f"{_prefix}<{_dtype_str}, {_space}"

    # layout was specified via numpy "order" field
    if order is not None and layout == lib.LayoutRight and isinstance(order, str):
        if order.upper() == "C":
            layout = lib.LayoutRight
        elif order.upper() == "F":
            layout = lib.LayoutLeft

    # handle the layout argument
    if layout is not None:
        _layout = lib.get_layout(layout)
        # LayoutRight is the default
        if _layout != "LayoutRight":
            _name = f"{_name}_{_layout}"
            _label = f"{_label}, {_layout}"

    # handle the trait argument
    if trait is not None:
        _trait = lib.get_memory_trait(trait)
        if _trait == "Unmanaged":
            raise ValueError(
                "Use unmanaged_array() for the unmanaged view memory trait"
            )
        else:
            if _trait != "Managed":
                _name = f"{_name}_{_trait}"
                _label = f"{_label}, {_trait}"

    # if fixed view
    if not dynamic:
        _name = f"{_name}_{_ndim}"

    # if a label was not provided
    if label is None:
        label = f"{_label}>"

    return getattr(lib, _name)(str(label), shape)


def unmanaged_array(
    array,
    dtype=lib.double,
    space=lib.HostSpace,
    layout=None,
    trait=None,
    dynamic=False,
):
    _prefix = "KokkosView"
    if dynamic:
        _prefix = "KokkosDynRankView"
    _dtype = lib.get_dtype(dtype)
    if layout is None:
        layout = lib.LayoutRight
        try:
            _order = array.order
            if _order == "F":
                layout = lib.LayoutLeft
        except AttributeError:
            pass
    _layout = lib.get_layout(layout)
    _space = lib.get_memory_space(space)
    _trait = lib.get_memory_trait(lib.Unmanaged)
    _shape = array.shape
    if array.ndim < 1:
        raise ValueError(array.ndim)
    _ndim = array.ndim

    _name = f"{_prefix}_{_dtype}_{_space}"
    if layout == lib.LayoutLeft:
        _name = f"{_name}_{_layout}"
    _name = f"{_name}_{_trait}"
    if dynamic is False:
        _name = f"{_name}_{_ndim}"

    return getattr(lib, _name)(array, _shape)


def convert_dtype(_dtype, _module=None):
    if isinstance(_dtype, (str, int)):
        _true_dtype = lib.get_dtype(_dtype)
    else:
        _true_dtype = lib.get_dtype(int(_dtype))
    if _module is None:
        import numpy as np

        _module = np
    return getattr(_module, _true_dtype)
