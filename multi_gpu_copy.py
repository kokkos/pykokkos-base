# This should only be run after kokkos is installed

import glob
from pathlib import Path
import shutil

import kokkos
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
    package_path = Path(package) / "lib"
    shutil.copytree(lib_path, package_path, dirs_exist_ok=True)