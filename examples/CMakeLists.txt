cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(Kokkos-Python-Example LANGUAGES C CXX)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_library(user SHARED user.cpp)
target_link_libraries(user PUBLIC Kokkos::kokkos)

pybind11_add_module(ex_generate
    ${PROJECT_SOURCE_DIR}/ex-generate.cpp)

target_link_libraries(ex_generate PRIVATE user)
# target_compile_options(ex_generate PRIVATE -fvisibility=default)
configure_file(${PROJECT_SOURCE_DIR}/ex-numpy.py
    ${CMAKE_BINARY_DIR}/ex-numpy.py @ONLY)