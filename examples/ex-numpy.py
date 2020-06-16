
from ex_generate import generate_view
import numpy as np
import argparse
import kokkos

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--ndim", default=100,
                        help="X dimension", type=int)

    args = parser.parse_args()

    kokkos.initialize()

    view = generate_view(args.ndim)
    print("Kokkos View : {}".format(view))
    # for i in range(args.ndim):
    #    for j in range(args.mdim):
    #        print("    view({}, {}) = {}".format(i, j, view(i, j)))

    arr = np.array(view)
    print("Numpy Array : {}, {}".format(arr, arr.shape))
    for i in range(arr.shape[0]):
        print("    view({}) = {}".format(i, arr[i]))
    del view
    del arr

    kokkos.finalize()
