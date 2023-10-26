# This should only be run after kokkos is installed

import glob
from pathlib import Path
import shutil
import subprocess
import sys

import kokkos

if shutil.which("patchelf") is None:
    sys.exit("ERROR: Cannot run multi_gpu_copy.py without 'patchelf'")

kokkos_path = kokkos.__path__[0]
base_path = Path(kokkos_path).parent

lib_path = None
if (base_path / "lib").is_dir():
    lib_path = base_path / "lib"
elif (base_path / "lib64").is_dir():
    lib_path = base_path / "lib64"

assert(lib_path is not None)
package_paths = [kokkos_path] + glob.glob(f"{str(base_path)}/gpu*")

for package in package_paths:
    package_path = Path(package)
    package_lib_path = package_path / "lib"
    shutil.copytree(lib_path, package_lib_path, dirs_exist_ok=True)

    if package.endswith("kokkos"):
        continue

    package_id = package[-1]

    libkokkoscore_remove = None
    libkokkoscore_add = None
    libkokkoscontainers_remove = None
    libkokkoscontainers_add = None

    for lib in package_lib_path.iterdir():
        if lib.name == "cmake":
            continue

        # Add the suffix to the end of each copy
        suffix = lib.name.split(".")
        lib_name = suffix[0]
        suffix = suffix[1:]
        suffix = ".".join(suffix)

        new_name = f"{lib_name}_{package_id}.{suffix}"

        if new_name.count(".") == 3: # this is the library that's listed as a dependency
            if "libkokkoscore" in new_name:
                libkokkoscore_remove = lib.name
                libkokkoscore_add = new_name
            elif "libkokkoscontainers" in new_name:
                libkokkoscontainers_remove = lib.name
                libkokkoscontainers_add = new_name

        new_lib_path = Path(lib.parent) / new_name
        lib.rename(new_lib_path)

        # Add the suffix to the end of the SONAME
        so_name = subprocess.run(["patchelf", "--print-soname", new_lib_path], capture_output=True).stdout.decode("utf-8")
        so_name = so_name.split(".")
        lib_name = so_name[0]
        suffix = so_name[1:]
        suffix = [s.strip() for s in suffix]
        suffix = ".".join(suffix)

        new_so_name = f"{lib_name}_{package_id}.{suffix}"
        subprocess.run(["patchelf", "--set-soname", new_so_name, new_lib_path])

    for file in package_path.iterdir():
        if "libpykokkos" in file.name:
            libpykokkos_path = file
            break

    assert(libkokkoscore_remove is not None)
    assert(libkokkoscore_add is not None)
    assert(libkokkoscontainers_remove is not None)
    assert(libkokkoscontainers_add is not None)

    subprocess.run(["patchelf", "--replace-needed", libkokkoscore_remove, libkokkoscore_add, libpykokkos_path])
    subprocess.run(["patchelf", "--replace-needed", libkokkoscontainers_remove, libkokkoscontainers_add, libpykokkos_path])
    subprocess.run(["patchelf", "--set-rpath", package_lib_path, libpykokkos_path])