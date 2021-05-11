#
#   Finds the Packages
#
INCLUDE(KokkosPythonUtilities)

# basically just used to get Python3_SITEARCH for installation
FIND_PACKAGE(Python3 REQUIRED COMPONENTS Interpreter Development)
SET(PYBIND11_PYTHON_VERSION "${PYTHON_VERSION_STRING}" CACHE STRING "Python version" FORCE)
ADD_FEATURE(PYBIND11_PYTHON_VERSION "Python version used by PyBind11")
ADD_FEATURE(PYTHON_VERSION_STRING "Python version found")

# python binding library
IF(ENABLE_INTERNAL_PYBIND11)
    CHECKOUT_GIT_SUBMODULE(RECURSIVE
    RELATIVE_PATH pybind11
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    ADD_SUBDIRECTORY(pybind11)
ELSE()
    FIND_PACKAGE(pybind11 REQUIRED)
ENDIF()
