
SET(_functions)

#
#   TODO:
#       Determine whether we want to remove expansion in code and replace with
#       all explicit instantiations.
#
#   TODO:
#       Reduce (or make it configurable) the number of dimensions supported by
#       concrete views. Currently, we instantiate up to 8 dimensions but we could
#       theoretically just say that python only supports up to 3 dimensions and
#       if higher than that, user bindings must convert View to DynRankView.
#
SET(_types              concrete dynamic)
SET(_variants           layout memory_trait)
SET(_data_types         Int16 Int32 Int64 Uint16 Uint32 Uint64 Float32 Float64)

SET(layout_enums        Right)
SET(memory_trait_enums  Managed)

IF(ENABLE_LAYOUTS)
    LIST(APPEND layout_enums        Left)
ENDIF()

#
#   TODO:
#       Are there any combinations of memory traits that are commonly used?
#       E.g. RandomAccess + Restrict
#
IF(ENABLE_MEMORY_TRAITS)
    LIST(APPEND memory_trait_enums  Aligned Atomic RandomAccess Restrict)
ENDIF()

# loop over dynamic and concrete
FOREACH(TYPE ${_types})
    FOREACH(DATA_VARIANT ${_data_types})
        FOREACH(LAYOUT_VARIANT ${layout_enums})
            FOREACH(MEMORY_TRAIT_VARIANT ${memory_trait_enums})
                STRING(TOLOWER "${DATA_VARIANT}_${LAYOUT_VARIANT}_${MEMORY_TRAIT_VARIANT}" VARIANT)
                STRING(TOLOWER "${TYPE}_view_${VARIANT}" _TAG)
                SET(ENUM "${DATA_VARIANT}, ${LAYOUT_VARIANT}, ${MEMORY_TRAIT_VARIANT}")
                SET(FUNC "generate_${_TAG}")
                IF(NOT ENABLE_QUIET)
                    MESSAGE(STATUS "Generating '${_TAG}.cpp'...")
                ENDIF()
                CONFIGURE_FILE(
                    ${CMAKE_CURRENT_SOURCE_DIR}/variant.cpp.in
                    ${CMAKE_CURRENT_BINARY_DIR}/${_TAG}.cpp
                    @ONLY)
                TARGET_SOURCES(libpykokkos PUBLIC
                    ${CMAKE_CURRENT_BINARY_DIR}/${_TAG}.cpp)
                LIST(APPEND _functions "${FUNC}")
            ENDFOREACH()
        ENDFOREACH()
    ENDFOREACH()
ENDFOREACH()

FOREACH(_FUNC ${_functions})
    SET(PROTOTYPES "${PROTOTYPES}void ${_FUNC}(py::module&);\n")
    SET(INVOCATIONS "${INVOCATIONS}  ${_FUNC}(kokkos);\n")
ENDFOREACH()

CONFIGURE_FILE(
    ${CMAKE_CURRENT_SOURCE_DIR}/view.cpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/views.cpp
    @ONLY)

FILE(GLOB EXISTING_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${PROJECT_SOURCE_DIR}/include/variants/*.hpp)

TARGET_SOURCES(libpykokkos PUBLIC
    ${EXISTING_SOURCES}
    ${CMAKE_CURRENT_BINARY_DIR}/views.cpp)
