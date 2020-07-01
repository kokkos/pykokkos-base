# kokkos-python

## Example

This example is designed to emulate a work-flow where the user has code using Kokkos in C++ and writes python bindings to those functions. A python script is used as the `"main"`:
  - `ex-numpy.py` imports the kokkos bindings
  - Calls a routine in the "users" python bindings to a C++ function which returns a `Kokkos::View`
  - This view is then converted to a numpy array in python and printed via the numpy capabilities.

```console
mkdir build
cd build
cmake -DBUILD_EXAMPLES=ON ..
make
python ./ex-numpy.py
```
