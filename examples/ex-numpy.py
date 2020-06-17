#!/usr/bin/env python

import argparse
import gc
import numpy as np

#
# The python bindings for generate_view are in ex-generate.cpp
# The declaration and definition of generate_view are in user.hpp and user.cpp
# The generate_view function will return a Kokkos::View and will be converted
# to a numpy array
from ex_generate import generate_view

#
# Importing this module is necessary to call kokkos init/finalize and
# import the python bindings to Kokkos::View which generate_view will
# return
#
import kokkos


def main(args):
    # get the kokkos view
    view = generate_view(args.ndim)
    # verify the type id
    print("Kokkos View : {}".format(type(view).__name__))
    # wrap the buffer protocal as numpy array without copying the data
    arr = np.array(view, copy=False)
    # verify type id
    print("Numpy Array : {} (shape={})".format(type(arr).__name__, arr.shape))
    # demonstrate the data is the same as what was printed by generate_view
    for i in range(arr.shape[0]):
        print("    view({}) = {}".format(i, arr[i]))

if __name__ == "__main__":
    kokkos.initialize()
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--ndim", default=10,
                        help="X dimension", type=int)
    args = parser.parse_args()
    main(args)
    # make sure all views are garbage collected
    gc.collect()
    kokkos.finalize()
