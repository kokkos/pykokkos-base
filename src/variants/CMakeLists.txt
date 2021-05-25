
SET(_functions)

SET(_types              concrete dynamic)
SET(_variants           layout memory_trait)

SET(layout_enums        Right)
SET(memory_trait_enums  Managed Unmanaged)

IF(ENABLE_LAYOUTS)
    LIST(APPEND layout_enums        Left)
ENDIF()

IF(ENABLE_MEMORY_TRAITS)
    LIST(APPEND memory_trait_enums  Atomic RandomAccess Restrict)
ENDIF()

# loop over dynamic and concrete
FOREACH(TYPE ${_types})
    FOREACH(LAYOUT_VARIANT ${layout_enums})
        FOREACH(MEMORY_TRAIT_VARIANT ${memory_trait_enums})
            STRING(TOLOWER "${LAYOUT_VARIANT}_${MEMORY_TRAIT_VARIANT}" VARIANT)
            STRING(TOLOWER "${TYPE}_view_${VARIANT}" _TAG)
            SET(ENUM "${LAYOUT_VARIANT}, ${MEMORY_TRAIT_VARIANT}")
            SET(FUNC "generate_${_TAG}")
            MESSAGE(STATUS "Generating '${_TAG}.cpp'...")
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

FOREACH(_FUNC ${_functions})
    SET(PROTOTYPES "${PROTOTYPES}void ${_FUNC}(py::module&);\n")
    SET(INVOCATIONS "${INVOCATIONS}  ${_FUNC}(kokkos);\n")
ENDFOREACH()

CONFIGURE_FILE(
    ${CMAKE_CURRENT_SOURCE_DIR}/view.cpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/views.cpp
    @ONLY)

TARGET_SOURCES(libpykokkos PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/views.cpp)
