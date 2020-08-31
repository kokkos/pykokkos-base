CMAKE_MINIMUM_REQUIRED(VERSION 3.11 FATAL_ERROR)

PROJECT(Kokkos-python LANGUAGES C CXX VERSION 0.0.1)

SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules ${CMAKE_MODULE_PATH})

INCLUDE(KokkosPythonUtilities)
INCLUDE(CheckCxxSourceCompiles)

IF("${CMAKE_BUILD_TYPE}" STREQUAL "")
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
ENDIF()

ADD_FEATURE(CMAKE_BUILD_TYPE "Build type")
ADD_FEATURE(CMAKE_INSTALL_PREFIX "Installation prefix")
ADD_FEATURE(Kokkos_CXX_STANDARD "Kokkos C++ Standard")
ADD_FEATURE(Kokkos_DIR "Kokkos installation")
ADD_OPTION(ENABLE_INTERNAL_PYBIND11 "Build with pybind11 submodule" ON)
ADD_OPTION(ENABLE_EXPERIMENTAL "Build the experimental code" OFF)
ADD_OPTION(ENABLE_WERROR "Build with -Werror" OFF)
ADD_OPTION(BUILD_EXAMPLES "Build the examples" OFF)
# don't build a static python module
SET(BUILD_SHARED_LIBS ON)

FIND_PACKAGE(Kokkos REQUIRED)

# pybind11 has not migrated to CMAKE_CXX_STANDARD and neither has kokkos
SET(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_EXTENSIONS OFF)
IF(NOT Kokkos_CXX_STANDARD)
    SET(Kokkos_CXX_STANDARD ${CMAKE_CXX_STANDARD})
ENDIF()
IF(MSVC)
    SET(PYBIND11_CPP_STANDARD /std:c++${Kokkos_CXX_STANDARD} CACHE STRING "C++ Language standard used by Pybind11" FORCE)
ELSE()
    SET(PYBIND11_CPP_STANDARD -std=c++${Kokkos_CXX_STANDARD} CACHE STRING "C++ Language standard used by Pybind11" FORCE)
ENDIF()

# basically just used to get Python3_SITEARCH for installation
FIND_PACKAGE(Python3 REQUIRED COMPONENTS Interpreter Development)
# python binding library
IF(ENABLE_INTERNAL_PYBIND11)
    CHECKOUT_GIT_SUBMODULE(RECURSIVE
    RELATIVE_PATH pybind11
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    ADD_SUBDIRECTORY(pybind11)
ELSE()
    FIND_PACKAGE(pybind11 REQUIRED)
ENDIF()

PYBIND11_ADD_MODULE(libpykokkos MODULE THIN_LTO
    ${CMAKE_CURRENT_LIST_DIR}/src/libpykokkos.cpp
    ${CMAKE_CURRENT_LIST_DIR}/include/libpykokkos.hpp)

# add extra build properties to this target
ADD_LIBRARY(libpykokkos-build-options INTERFACE)
TARGET_INCLUDE_DIRECTORIES(libpykokkos-build-options INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
TARGET_COMPILE_OPTIONS(libpykokkos-build-options INTERFACE
    -W -Wall -Wextra -fvisibility=default)

TRY_COMPILE(ENABLE_DEMANGLE
    ${PROJECT_BINARY_DIR}
    ${PROJECT_SOURCE_DIR}/cmake/Templates/demangle.cpp
    OUTPUT_VARIABLE ENABLE_DEMANGLE_OUTPUT)

IF(ENABLE_DEMANGLE)
    TARGET_COMPILE_DEFINITIONS(libpykokkos-build-options INTERFACE ENABLE_DEMANGLE)
ENDIF()

IF(ENABLE_EXPERIMENTAL)
    TARGET_COMPILE_DEFINITIONS(libpykokkos-build-options INTERFACE ENABLE_EXPERIMENTAL)
ENDIF()

IF(ENABLE_WERROR)
    TARGET_COMPILE_OPTIONS(libpykokkos-build-options INTERFACE
        -Werror)
ENDIF()

# link to kokkos and the custom build properties
TARGET_LINK_LIBRARIES(libpykokkos PRIVATE Kokkos::kokkos libpykokkos-build-options)

# set the output path to <BINARY_DIR>/kokkos so one
# can test the python import from the build directory
# Really, only LIBRARY_* is needed for Unix but Windows
# builds are weird so just setting all of them
SET_TARGET_PROPERTIES(libpykokkos PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/kokkos
    ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/kokkos
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/kokkos
    PDB_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/kokkos)

SET(Kokkos_INSTALL_PYTHONDIR ${Python3_SITEARCH}/kokkos)

# figure out if we can install to Python3_SITEARCH
EXECUTE_PROCESS(
    COMMAND ${CMAKE_COMMAND} -E touch ${Python3_SITEARCH}/kokkos/__init__.py
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
    ERROR_VARIABLE ERR_MSG
    RESULT_VARIABLE ERR_CODE)

IF(ERR_CODE)
    # get the python directory name, e.g. 'python3.6' from
    # '/opt/local/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6'
    get_filename_component(PYDIR "${Python3_STDLIB}" NAME)
    # Should not be CMAKE_INSTALL_LIBDIR! Python won't look in a lib64 folder
    set(Kokkos_INSTALL_PYTHONDIR lib/${PYDIR}/site-packages/kokkos)
ENDIF()

INSTALL(TARGETS libpykokkos
    DESTINATION ${Kokkos_INSTALL_PYTHONDIR})

INSTALL(FILES ${PROJECT_BINARY_DIR}/kokkos/__init__.py
    DESTINATION ${Kokkos_INSTALL_PYTHONDIR})

# glob any python package files
FILE(GLOB_RECURSE PYPACKAGE_FILES ${CMAKE_CURRENT_LIST_DIR}/kokkos/*.py*)
FOREACH(_FILE ${PYPACKAGE_FILES})
    # make it a relative path
    STRING(REPLACE "${CMAKE_CURRENT_LIST_DIR}/" "" _OUT_NAME "${_FILE}")
    # get the directory of the relative path
    GET_FILENAME_COMPONENT(_OUT_PATH "${_OUT_NAME}" DIRECTORY)
    # get the name without the extension
    GET_FILENAME_COMPONENT(_OUT_NAME "${_OUT_NAME}" NAME_WE)
    # target file for configure
    SET(_OUT_FILE ${PROJECT_BINARY_DIR}/${_OUT_PATH}/${_OUT_NAME}.py)
    # put version, python interpreter, etc. in the file for reference
    CONFIGURE_FILE(${_FILE} ${_OUT_FILE} @ONLY)
    # install to the correct folder structure
    INSTALL(FILES ${_OUT_FILE} DESTINATION ${Kokkos_INSTALL_PYTHONDIR}/${_OUT_PATH})
ENDFOREACH()

# build the examples, not designed to be built stand-alone
IF(BUILD_EXAMPLES)
    ADD_SUBDIRECTORY(examples)
ENDIF()

PRINT_FEATURES()
