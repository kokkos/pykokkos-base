# kokkos-python

## Example

### Overview

This example is designed to emulate a work-flow where the user has code using Kokkos in C++ and writes python bindings to those functions. A python script is used as the `"main"`:
  - `ex-numpy.py` imports the kokkos bindings
  - Calls a routine in the "users" python bindings to a C++ function which returns a `Kokkos::View`
  - This view is then converted to a numpy array in python and printed via the numpy capabilities.

### Files

- [ex-generate.cpp](https://github.com/kokkos/kokkos-python/blob/main/examples/ex-generate.cpp)
  - This is the python bindings to the user code
- [user.cpp](https://github.com/kokkos/kokkos-python/blob/main/examples/user.cpp)
  - This is the implementation of the user's code which returns a `Kokkos::View<double**, Kokkos::HostSpace>`
- [ex-numpy.py](https://github.com/kokkos/kokkos-python/blob/main/examples/ex-numpy.py)
  - This is the "main"
  
#### ex-numpy.py

```python
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
```

### Build and Run

```console
mkdir build
cd build
cmake -DBUILD_EXAMPLES=ON ..
make
python ./ex-numpy.py
```

### Expected Output

```console
    view(0) =  0  0
    view(1) =  0  1
    view(2) =  2  0
    view(3) =  0  3
    view(4) =  4  0
    view(5) =  0  5
    view(6) =  6  0
    view(7) =  0  7
    view(8) =  8  0
    view(9) =  0  9
Sum of view: 45
extent(0): 10
stride(0): 2
Kokkos View : KokkosView_HostSpace_double_2
Numpy Array : ndarray (shape=(10, 2))
    view(0) = [0. 0.]
    view(1) = [0. 1.]
    view(2) = [2. 0.]
    view(3) = [0. 3.]
    view(4) = [4. 0.]
    view(5) = [0. 5.]
    view(6) = [6. 0.]
    view(7) = [0. 7.]
    view(8) = [8. 0.]
    view(9) = [0. 9.]
```
