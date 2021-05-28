#
#   Adds Kokkos as a git submodule
#
INCLUDE(KokkosPythonUtilities)

# check to see if ${PROJECT_SOURCE_DIR}/external/kokkos/CMakeLists.txt exists
# if it doesn't exist, try checking out submodule if .gitmodules exists.
# if .gitmodules does not exist, then run `git clone -b ${REPO_BRANCH} ${REPO_URL}`
CHECKOUT_GIT_SUBMODULE(
    RELATIVE_PATH     external/kokkos
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    TEST_FILE         CMakeLists.txt
    REPO_URL          https://github.com/kokkos/kokkos.git
    REPO_BRANCH       master)

UNSET(Kokkos_DIR CACHE)
SET(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries")
SET(CMAKE_UNITY_BUILD OFF)

MESSAGE(STATUS "")
MESSAGE(STATUS "Building Kokkos from submodule...")
MESSAGE("")

ADD_SUBDIRECTORY(kokkos)

MESSAGE("")
MESSAGE(STATUS "Kokkos submodule added")
MESSAGE(STATUS "")
