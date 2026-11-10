cmake_minimum_required(VERSION 3.19)

project(mipk VERSION 2.0.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include(MipkModule)

option(MIPK_WERROR "Treat warnings as errors" ON)
option(MIPK_BUILD_TESTS "Build the host test suite" ON)
option(MIPK_BUILD_EXAMPLES "Build the example programs" ON)
option(MIPK_USE_SECUREC "Use securec memcpy_s/memset_s" OFF)

# Operators
option(MIPK_OPS_IMAGE "image_preprocess, roi_extract, image_align" ON)
option(MIPK_OPS_DETECTION "nms, result_filter" ON)
option(MIPK_OPS_TRACKING "bytetrack" ON)
option(MIPK_OPS_AUDIO "tfmicrofrontend, wake_word_trigger (pulls in TF microfrontend)" OFF)

# Models. Every one is optional; a pipeline that names a disabled one fails
# to open with a message naming the module.
option(MIPK_MODEL_YOLO "yolo detection decoder" OFF)
option(MIPK_MODEL_YOLO_POSE "yolo_pose decoder" OFF)
option(MIPK_MODEL_CLASSIFICATION "classification decoder" OFF)
option(MIPK_MODEL_KEYPOINTS "keypoints decoder" OFF)
option(MIPK_MODEL_MICROWAKEWORD "microwakeword decoder (needs MIPK_OPS_AUDIO)" OFF)

# Modules kept outside this tree.  Each entry is a module directory holding
# module.json and a CMakeLists.txt that calls mipk_add_module().  It is
# registered and archived into libmipk.a exactly like a built-in module, so
# the application needs no code to enable it; the packaging tool learns about
# the same modules through --module-dir.
set(MIPK_EXTRA_MODULE_DIRS "" CACHE STRING "Out-of-tree module directories (semicolon separated)")

# Backends
option(MIPK_BACKEND_NULL "null backend for graph tests" ON)
option(MIPK_BACKEND_ONNXRUNTIME "onnxruntime backend" OFF)
set(MIPK_ONNXRUNTIME_ROOT "" CACHE PATH "Prebuilt onnxruntime directory (include/ and lib/)")

option(MIPK_BACKEND_SVP_ACL "HiSilicon SVP ACL backend (.om on the NPU)" OFF)
set(MIPK_HI_SDK_ROOT "" CACHE PATH "HiSilicon SoC SDK root (soc/include, soc/libs)")

# Platforms
option(MIPK_PLATFORM_POSIX "posix platform (malloc, no cache maintenance)" ON)
option(MIPK_PLATFORM_HISILICON "HiSilicon platform (MMZ, cache maintenance, NPU arbiter)" OFF)

# Resource ceilings. "host" keeps the generous defaults in
# core/include/mipk_config.h; "hi3516cv610" is the 32 MiB device profile.
set(MIPK_PROFILE "host" CACHE STRING "Resource profile: host or hi3516cv610")

if(MSVC)
    # /W3 minus the implicit-conversion warnings (C4244, C4267, C4305),
    # which GCC's -Wall -Wextra do not enable either.  The compat header
    # maps the POSIX calls core makes onto the CRT and third_party/win32
    # supplies pthread.h.
    set(MIPK_WARNING_FLAGS /W3 /wd4244 /wd4267 /wd4305)
    if(MIPK_WERROR)
        list(APPEND MIPK_WARNING_FLAGS /WX)
    endif()
    add_compile_options(/utf-8 "/FI${CMAKE_CURRENT_SOURCE_DIR}/third_party/win32/mipk_win32_compat.h")
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS _CRT_NONSTDC_NO_WARNINGS
        WIN32_LEAN_AND_MEAN NOMINMAX)
    include_directories(SYSTEM "${CMAKE_CURRENT_SOURCE_DIR}/third_party/win32")
else()
    set(MIPK_WARNING_FLAGS -Wall -Wextra)
    if(MIPK_WERROR)
        list(APPEND MIPK_WARNING_FLAGS -Werror)
    endif()
endif()

add_subdirectory(third_party)
add_subdirectory(core)
if(MIPK_PROFILE STREQUAL "hi3516cv610")
    target_compile_definitions(mipk_core_headers INTERFACE
        MIPK_CONTAINER_MANIFEST_MAX_SIZE=65536U
        MIPK_MAX_MODEL_BYTES=12582912U
        MIPK_MAX_OUTPUT_BYTES=524288U
        MIPK_MAX_SCRATCH_BYTES=1048576U
        MIPK_MAX_MMZ_BYTES=16777216U)
elseif(NOT MIPK_PROFILE STREQUAL "host")
    message(FATAL_ERROR "unknown MIPK_PROFILE '${MIPK_PROFILE}'")
endif()

if(MIPK_OPS_IMAGE)
    add_subdirectory(ops/image_preprocess)
    add_subdirectory(ops/roi_extract)
    add_subdirectory(ops/image_align)
endif()
if(MIPK_OPS_DETECTION)
    add_subdirectory(ops/nms)
    add_subdirectory(ops/result_filter)
endif()
if(MIPK_OPS_TRACKING)
    add_subdirectory(ops/bytetrack)
endif()
if(MIPK_OPS_AUDIO)
    add_subdirectory(ops/tfmicrofrontend)
    add_subdirectory(ops/wake_word_trigger)
endif()

if(MIPK_MODEL_YOLO)
    add_subdirectory(models/yolo)
endif()
if(MIPK_MODEL_YOLO_POSE)
    add_subdirectory(models/yolo_pose)
endif()
if(MIPK_MODEL_CLASSIFICATION)
    add_subdirectory(models/classification)
endif()
if(MIPK_MODEL_KEYPOINTS)
    add_subdirectory(models/keypoints)
endif()
if(MIPK_MODEL_MICROWAKEWORD)
    if(NOT MIPK_OPS_AUDIO)
        message(FATAL_ERROR "MIPK_MODEL_MICROWAKEWORD needs MIPK_OPS_AUDIO")
    endif()
    add_subdirectory(models/microwakeword)
endif()
foreach(_extra_dir IN LISTS MIPK_EXTRA_MODULE_DIRS)
    get_filename_component(_extra_dir "${_extra_dir}" ABSOLUTE)
    if(NOT EXISTS "${_extra_dir}/module.json" OR NOT EXISTS "${_extra_dir}/CMakeLists.txt")
        message(FATAL_ERROR "MIPK_EXTRA_MODULE_DIRS: ${_extra_dir} is not a module directory (needs module.json and CMakeLists.txt)")
    endif()
    get_filename_component(_extra_name "${_extra_dir}" NAME)
    add_subdirectory("${_extra_dir}" "${CMAKE_CURRENT_BINARY_DIR}/extra_modules/${_extra_name}")
endforeach()

if(MIPK_BACKEND_NULL)
    add_subdirectory(backends/null)
endif()
if(MIPK_BACKEND_ONNXRUNTIME)
    add_subdirectory(backends/onnxruntime)
endif()
if(MIPK_BACKEND_SVP_ACL)
    add_subdirectory(backends/svp_acl)
endif()

if(MIPK_PLATFORM_POSIX)
    add_subdirectory(platforms/posix)
endif()
if(MIPK_PLATFORM_HISILICON)
    add_subdirectory(platforms/hisilicon)
endif()

# The library applications link: core plus every enabled module plus the
# generated registry that wires them together.
# libmipk.a holds core, the generated registry and every enabled module and
# platform as one archive.  Core's registry calls each module's register
# function and the modules call core's API; inside a single archive the
# linker resolves that in either direction, which separate static libraries
# cannot promise.  Applications link mipk::mipk and nothing else.
mipk_generate_registry(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/mipk_modules_gen.c")
get_property(_mipk_modules GLOBAL PROPERTY MIPK_ENABLED_MODULES)
add_library(mipk STATIC "${CMAKE_CURRENT_BINARY_DIR}/mipk_modules_gen.c")
add_library(mipk::mipk ALIAS mipk)
target_link_libraries(mipk PUBLIC mipk::core ${_mipk_modules})
if(MIPK_PLATFORM_POSIX)
    target_link_libraries(mipk PUBLIC mipk::platform_posix)
endif()
if(MIPK_PLATFORM_HISILICON)
    target_link_libraries(mipk PUBLIC mipk::platform_hisilicon)
endif()
target_compile_options(mipk PRIVATE ${MIPK_WARNING_FLAGS})
set_target_properties(mipk PROPERTIES POSITION_INDEPENDENT_CODE ON)
mipk_write_capabilities(TARGET mipk OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/mipk_capabilities.json")

if(MIPK_BUILD_EXAMPLES)
    add_subdirectory(examples/host)
endif()
# The host tests are kept out of the repository; build them when a local
# tests/ directory is present.
if(MIPK_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
    enable_testing()
    add_subdirectory(tests)
endif()

message(STATUS "mipk modules: ${_mipk_modules}")
