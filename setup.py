#!/usr/bin/env python

import os
import sys
import argparse
import warnings
import platform
from skbuild import setup

# some Cray systems default to static libraries and the build
# will fail because BUILD_SHARED_LIBS will get set to off
if os.environ.get("CRAYPE_VERSION") is not None:
    os.environ["CRAYPE_LINK_TYPE"] = "dynamic"

cmake_args = [
    "-DPYTHON_EXECUTABLE:FILEPATH={}".format(sys.executable),
    "-DPython3_EXECUTABLE:FILEPATH={}".format(sys.executable),
    "-DCMAKE_INSTALL_RPATH_USE_LINK_PATH:BOOL=ON",
]

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("-h", "--help", help="Print help", action="store_true")


def set_cmake_bool_option(opt, enable_opt, disable_opt):
    global cmake_args
    try:
        if enable_opt:
            cmake_args.append("-D{}:BOOL={}".format(opt, "ON"))
        if disable_opt:
            cmake_args.append("-D{}:BOOL={}".format(opt, "OFF"))
    except Exception as e:
        print("Exception: {}".format(e))


def add_arg_bool_option(lc_name, disp_name, default=None):
    global parser
    # enable option
    parser.add_argument(
        "--enable-{}".format(lc_name),
        action="store_true",
        default=default,
        help="Explicitly enable {} build".format(disp_name),
    )
    # disable option
    parser.add_argument(
        "--disable-{}".format(lc_name),
        action="store_true",
        help="Explicitly disable {} build".format(disp_name),
    )


# add options
add_arg_bool_option("all", "ENABLE_ALL")
add_arg_bool_option("experimental", "ENABLE_EXPERIMENTAL")
add_arg_bool_option("layouts", "ENABLE_LAYOUTS")
add_arg_bool_option("memory-traits", "ENABLE_MEMORY_TRAITS")
add_arg_bool_option("thin-lto", "ENABLE_THIN_LTO")
add_arg_bool_option("werror", "ENABLE_WERROR")
add_arg_bool_option("timing", "ENABLE_TIMING")
parser.add_argument(
    "--cxx-standard",
    default=14,
    type=int,
    choices=[14, 17, 20],
    help="Set C++ language standard",
)
parser.add_argument(
    "--kokkos-root",
    default=None,
    type=str,
    help="Path to kokkos install prefix",
)
parser.add_argument(
    "--cmake-args",
    default=[],
    type=str,
    nargs="*",
    help="{}{}".format(
        "Pass arguments to cmake. Use w/ pip installations and --install-option, e.g. ",
        '--install-option=--cmake-args="-DENABLE_ALL=ON -DKokkos_DIR=/usr/local/lib/cmake/Kokkos"',
    ),
)

args, left = parser.parse_known_args()
# if help was requested, print these options and then add '--help' back
# into arguments so that the skbuild/setuptools argparse catches it
if args.help:
    parser.print_help()
    left.append("--help")
sys.argv = sys.argv[:1] + left

set_cmake_bool_option("ENABLE_ALL", args.enable_all, args.disable_all)
set_cmake_bool_option(
    "ENABLE_EXPERIMENTAL", args.enable_experimental, args.disable_experimental
)
set_cmake_bool_option("ENABLE_LAYOUTS", args.enable_layouts, args.disable_layouts)
set_cmake_bool_option(
    "ENABLE_MEMORY_TRAITS",
    args.enable_memory_traits,
    args.disable_memory_traits,
)
set_cmake_bool_option("ENABLE_THIN_LTO", args.enable_thin_lto, args.disable_thin_lto)
set_cmake_bool_option("ENABLE_WERROR", args.enable_werror, args.disable_werror)
set_cmake_bool_option("ENABLE_TIMING", args.enable_timing, args.disable_timing)

cmake_args.append("-DCMAKE_CXX_STANDARD={}".format(args.cxx_standard))

for itr in args.cmake_args:
    cmake_args += itr.split()

if args.kokkos_root is not None:
    os.environ["CMAKE_PREFIX_PATH"] = ":".join(
        [args.kokkos_root, os.environ.get("CMAKE_PREFIX_PATH", "")]
    )

if platform.system() == "Darwin":
    # scikit-build will set this to 10.6 and C++ compiler check will fail
    darwin_version = platform.mac_ver()[0].split(".")
    darwin_version = ".".join([darwin_version[0], darwin_version[1]])
    cmake_args += ["-DCMAKE_OSX_DEPLOYMENT_TARGET={}".format(darwin_version)]

# DO THIS LAST!
# support PYKOKKOS_BASE_SETUP_ARGS environment variables because
#  --install-option for pip is a pain to use
# PYKOKKOS_BASE_SETUP_ARGS should be space-delimited set of cmake arguments, e.g.:
#   export PYKOKKOS_BASE_SETUP_ARGS="-DENABLE_ALL=OFF -DENABLE_MEMORY_TRAITS=ON"
env_cmake_args = os.environ.get("PYKOKKOS_BASE_SETUP_ARGS", None)
if env_cmake_args is not None:
    cmake_args += env_cmake_args.split(" ")


# --------------------------------------------------------------------------- #
#
def get_project_version():
    # open "VERSION"
    with open(os.path.join(os.getcwd(), "VERSION"), "r") as f:
        data = f.read().replace("\n", "")
    # make sure is string
    if isinstance(data, list) or isinstance(data, tuple):
        return data[0]
    else:
        return data


# --------------------------------------------------------------------------- #
#
def get_long_description():
    long_descript = ""
    try:
        long_descript = open("README.md").read()
    except Exception:
        long_descript = ""
    return long_descript


# --------------------------------------------------------------------------- #
#
def parse_requirements(fname="requirements.txt", with_version=False):
    """
    Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if true include version specs

    Returns:
        List[str]: list of requirements items

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
        python -c "import setup; print(chr(10).join(
            setup.parse_requirements(with_version=True)))"
    """
    from os.path import exists
    import re

    require_fpath = fname

    def parse_line(line):
        """
        Parse information from a line in a requirements text file
        """
        if line.startswith("-r "):
            # Allow specifying requirements in other files
            target = line.split(" ")[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {"line": line}
            if line.startswith("-e "):
                info["package"] = line.split("#egg=")[1]
            else:
                # Remove versioning from the package
                pat = "(" + "|".join([">=", "==", ">"]) + ")"
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info["package"] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ";" in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip, rest.split(";"))
                        info["platform_deps"] = platform_deps
                    else:
                        version = rest  # NOQA
                    info["version"] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info["package"]]
                if with_version and "version" in info:
                    parts.extend(info["version"])
                if not sys.version.startswith("3.4"):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get("platform_deps")
                    if platform_deps is not None:
                        parts.append(";" + platform_deps)
                item = "".join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


# suppress:
#  "setuptools_scm/git.py:68: UserWarning: "/.../<PACKAGE>"
#       is shallow and may cause errors"
# since 'error' in output causes CDash to interpret warning as error
with warnings.catch_warnings():
    print("CMake arguments: {}".format(" ".join(cmake_args)))
    setup(
        name="pykokkos-base",
        packages=["kokkos"],
        version=get_project_version(),
        cmake_args=cmake_args,
        cmake_languages=("C", "CXX"),
        long_description=get_long_description(),
        long_description_content_type="text/markdown",
        install_requires=parse_requirements("requirements.txt"),
        project_urls={"kokkos": "https://github.com/kokkos/kokkos"},
    )
