# Public headers and system dependencies, for modules and platforms that
# are compiled against core without linking it on their own.
add_library(mipk_core_headers INTERFACE)
add_library(mipk::core_headers ALIAS mipk_core_headers)
target_include_directories(mipk_core_headers INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/../include")
target_link_libraries(mipk_core_headers INTERFACE mipk::cjson)
if(NOT WIN32)
    target_link_libraries(mipk_core_headers INTERFACE pthread m)
endif()
if(MIPK_USE_SECUREC)
    target_compile_definitions(mipk_core_headers INTERFACE MIPK_USE_SECUREC)
endif()

add_library(mipk_core OBJECT
    src/mipk_manifest.c
    src/mipk_pipeline.c
    src/mipk_builder.c
    src/mipk_executor.c
    src/mipk_overrides.c
    src/mipk_value.c
    src/mipk_registry.c
    src/mipk_preprocess_geometry.c
    src/mipk_container.c
    src/mipk_tensor.c
    src/mipk_detection.c
    src/mipk_diag.c
    src/mipk_model.c)
add_library(mipk::core ALIAS mipk_core)
target_include_directories(mipk_core PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(mipk_core PUBLIC mipk::core_headers)
target_compile_options(mipk_core PRIVATE ${MIPK_WARNING_FLAGS})
set_target_properties(mipk_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Internal headers for the test suite only.
add_library(mipk_core_internal INTERFACE)
add_library(mipk::core_internal ALIAS mipk_core_internal)
target_include_directories(mipk_core_internal INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
