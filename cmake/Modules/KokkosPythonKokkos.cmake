
#----------------------------------------------------------------------------------------#
#   Kokkos submodule
#----------------------------------------------------------------------------------------#

INCLUDE(KokkosPythonUtilities)  # miscellaneous macros and functions

# if first time cmake is run and no external/internal preference is specified,
# try to find already installed kokkos
IF(NOT DEFINED ENABLE_INTERNAL_KOKKOS)
    FIND_PACKAGE(Kokkos QUIET)
    # set the default cache value
    IF(Kokkos_FOUND)
        SET(_INTERNAL_KOKKOS OFF)
    ELSE()
        SET(_INTERNAL_KOKKOS ON)
    ENDIF()
ELSE()
    # make sure ADD_OPTION in KokkosPythonOptions has a value
    SET(_INTERNAL_KOKKOS ${ENABLE_INTERNAL_KOKKOS})
ENDIF()

# force an error
IF(NOT _INTERNAL_KOKKOS)
    UNSET(FIND_PACKAGE_MESSAGE_DETAILS_Kokkos)
    FIND_PACKAGE(Kokkos REQUIRED)
ENDIF()

#
IF(_INTERNAL_KOKKOS)

    # try to find some packages quietly in order to set some defaults
    SET(OpenMP_FOUND OFF)
    SET(Threads_FOUND OFF)
    SET(CUDA_FOUND OFF)

    IF(NOT Kokkos_ENABLE_PTHREADS)
        FIND_PACKAGE(OpenMP QUIET)
    ENDIF()

    IF(NOT DEFINED Kokkos_ENABLE_PTHREADS AND NOT OpenMP_FOUND)
        FIND_PACKAGE(Threads QUIET)
    ENDIF()

    IF(NOT DEFINED Kokkos_ENABLE_CUDA)
        FIND_PACKAGE(CUDA QUIET)
    ENDIF()

    ADD_OPTION(ENABLE_SERIAL "Enable Serial backend when building Kokkos submodule" ON)
    ADD_OPTION(ENABLE_OPENMP "Enable OpenMP when building Kokkos submodule" ${OpenMP_FOUND})
    ADD_OPTION(ENABLE_THREADS "Enable Pthreads when building Kokkos submodule" ${Threads_FOUND})
    ADD_OPTION(ENABLE_CUDA "Enable CUDA when building Kokkos submodule" ${CUDA_FOUND})

    # if OpenMP defaulted to ON but Kokkos_ENABLE_PTHREADS was explicitly set,
    # disable OpenMP defaulting to ON
    IF(ENABLE_OPENMP AND Kokkos_ENABLE_PTHREADS)
        SET(ENABLE_OPENMP OFF)
        SET(Kokkos_ENABLE_OPENMP OFF)
    ENDIF()

    # always disable pthread backend since pthreads are not supported on Windows
    IF(WIN32)
        SET(ENABLE_THREADS OFF)
        SET(Kokkos_ENABLE_PTHREAD OFF)
    ENDIF()

    # make sure this pykokkos-base option is synced to Kokkos option
    IF(DEFINED Kokkos_ENABLE_SERIAL)
        SET(ENABLE_SERIAL ${Kokkos_ENABLE_SERIAL})
    ENDIF()

    # make sure this pykokkos-base option is synced to Kokkos option
    IF(DEFINED Kokkos_ENABLE_OPENMP)
        SET(ENABLE_OPENMP ${Kokkos_ENABLE_OPENMP})
    ENDIF()

    # make sure this pykokkos-base option is synced to Kokkos option
    IF(DEFINED Kokkos_ENABLE_PTHREAD)
        SET(ENABLE_THREADS ${Kokkos_ENABLE_PTHREAD})
    ENDIF()

    # make sure this pykokkos-base option is synced to Kokkos option
    IF(DEFINED Kokkos_ENABLE_CUDA)
        SET(ENABLE_CUDA ${Kokkos_ENABLE_CUDA})
    ENDIF()

    # define the kokkos option as default and/or get it to display
    IF(ENABLE_SERIAL OR Kokkos_ENABLE_SERIAL)
        ADD_OPTION(Kokkos_ENABLE_SERIAL "Build Kokkos submodule with serial support" ON)
    ENDIF()

    # define the kokkos option as default and/or get it to display
    IF(ENABLE_OPENMP OR Kokkos_ENABLE_OPENMP)
        ADD_OPTION(Kokkos_ENABLE_OPENMP "Build Kokkos submodule with OpenMP support" ON)
    ENDIF()

    # define the kokkos option as default and/or get it to display
    IF(ENABLE_THREADS OR Kokkos_ENABLE_PTHREAD)
        ADD_OPTION(Kokkos_ENABLE_PTHREAD "Build Kokkos submodule with Pthread support" ON)
    ENDIF()

    # define the kokkos option as default and/or get it to display
    IF(ENABLE_CUDA OR Kokkos_ENABLE_CUDA)
        ADD_OPTION(Kokkos_ENABLE_CUDA "Build Kokkos submodule with CUDA support" ON)
        ADD_OPTION(Kokkos_ENABLE_CUDA_UVM "Build Kokkos submodule with CUDA UVM support" ON)
        ADD_OPTION(Kokkos_ENABLE_CUDA_LAMBDA "Build Kokkos submodule with CUDA lambda support" ON)
        ADD_OPTION(Kokkos_ENABLE_CUDA_CONSTEXPR "Build Kokkos submodule with CUDA constexpr support" ON)
    ENDIF()

    ADD_SUBDIRECTORY(external)

ENDIF()
