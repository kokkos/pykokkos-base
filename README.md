# pykokkos-base

This package contains the minimal set of bindings for [Kokkos](https://github.com/kokkos/kokkos)
interoperability with Python:

- `Kokkos::initialize(...)`
- `Kokkos::finalize()`
- `Kokkos::View<...>`
- `Kokkos::DynRankView<...>`

By importing this package in Python, you can pass the supported Kokkos Views and DynRankViews
from C++ to Python and vice-versa. Furthermore, in Python, these bindings provide interoperability
with numpy and cupy arrays:

```python
import kokkos
import numpy as np

view = kokkos.array([2, 2], dtype=kokkos.double, space=kokkos.CudaUVMSpace,
                    layout=Kokkos.LayoutRight, trait=kokkos.RandomAccess,
                    dynamic=False)

arr = np.array(view, copy=False)
```

> This package depends on a pre-existing installation of Kokkos

## Writing Kokkos in Python

In order to write native Kokkos in Python, see [pykokkos](https://github.com/kokkos/pykokkos).

## Installation

You can install this package via CMake or Python's `setup.py`. There are two important cmake options:

- `ENABLE_LAYOUTS`
- `ENABLE_MEMORY_TRAITS`

By default, CMake will enable these options if the Kokkos installation was not built with CUDA support.
If Kokkos was built with CUDA support, these options will be disabled by default due to unreasonable
compilation times (> 1 hour).

### Configuring Options via CMake

```console
cmake -DENABLE_LAYOUTS=ON -DENABLE_MEMORY_TRAITS=OFF /path/to/source`
```

### Configuring Options via `setup.py`

There are three ways to configure the options:

1. Via the Python argparse options `--enable-<option>` and `--disable-<option>`
2. Setting the `PYKOKKOS_BASE_SETUP_ARGS` environment variable to the CMake options
3. Passing in the CMake options after a `--`

All three lines below are equivalent:

```console
python setup.py install --enable-layouts --disable-memory-traits
PYKOKKOS_BASE_SETUP_ARGS="-DENABLE_LAYOUTS=ON -DENABLE_MEMORY_TRAITS=OFF" python setup.py install
python setup.py install -- -DENABLE_LAYOUTS=ON -DENABLE_MEMORY_TRAITS=OFF
```

### Configuring Options via `pip`

Pip does not handle build options well. Thus, it is recommended to use the `PYKOKKOS_BASE_SETUP_ARGS`
environment variable noted above. However, using the `--install-option` for pip is possible but
each "space" must have it's own `--install-option`, e.g. all of the following are equivalent:
All three lines below are equivalent:

```console
pip install pykokkos-base --install-option=--enable-layouts --install-option=--disable-memory-traits
pip install pykokkos-base --install-option=-- --install-option=-DENABLE_LAYOUTS=ON --install-option=-DENABLE_MEMORY_TRAITS=OFF
pip install pykokkos-base --install-option={--enable-layouts,--disable-memory-traits}
pip install pykokkos-base --install-option={--,-DENABLE_LAYOUTS=ON,-DENABLE_MEMORY_TRAITS=OFF}
```

> `pip install pykokkos-base` will build against the latest release in the PyPi repository.
> In order to pip install from this repository, use `pip install --user -e .`

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
