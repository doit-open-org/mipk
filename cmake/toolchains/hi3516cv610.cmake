# Cross toolchain for Hi3516CV610 (arm32, musl, openEuler GCC).
# Pass -DMIPK_TOOLCHAIN_PATH=<bin dir> or put the compiler on PATH.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

set(_prefix "arm-openeuler-linux-musleabi-")
if(NOT MIPK_TOOLCHAIN_PATH AND DEFINED ENV{CV610_TOOLCHAIN_PATH})
    set(MIPK_TOOLCHAIN_PATH "$ENV{CV610_TOOLCHAIN_PATH}")
endif()
find_program(_mipk_cc NAMES "${_prefix}gcc" HINTS "${MIPK_TOOLCHAIN_PATH}")
find_program(_mipk_cxx NAMES "${_prefix}g++" HINTS "${MIPK_TOOLCHAIN_PATH}")
find_program(_mipk_ar NAMES "${_prefix}ar" HINTS "${MIPK_TOOLCHAIN_PATH}")
find_program(_mipk_strip NAMES "${_prefix}strip" HINTS "${MIPK_TOOLCHAIN_PATH}")
if(NOT _mipk_cc OR NOT _mipk_cxx OR NOT _mipk_ar)
    message(FATAL_ERROR "${_prefix}gcc not found; set MIPK_TOOLCHAIN_PATH or CV610_TOOLCHAIN_PATH")
endif()
# The compiler defaults to -mfloat-abi=soft and would emulate every float
# operation although the Cortex-A7 has VFPv4 and NEON.  softfp keeps the
# soft-float calling convention, so objects still link with the SDK, but
# uses the FPU inside functions.
set(_mipk_arm_flags "-mcpu=cortex-a7 -mfloat-abi=softfp -mfpu=neon-vfpv4")
set(CMAKE_C_FLAGS_INIT "${_mipk_arm_flags}")
set(CMAKE_CXX_FLAGS_INIT "${_mipk_arm_flags}")
set(CMAKE_ASM_FLAGS_INIT "${_mipk_arm_flags}")

set(CMAKE_C_COMPILER "${_mipk_cc}" CACHE FILEPATH "" FORCE)
set(CMAKE_CXX_COMPILER "${_mipk_cxx}" CACHE FILEPATH "" FORCE)
set(CMAKE_ASM_COMPILER "${_mipk_cc}" CACHE FILEPATH "" FORCE)
set(CMAKE_AR "${_mipk_ar}" CACHE FILEPATH "" FORCE)
if(_mipk_strip)
    set(CMAKE_STRIP "${_mipk_strip}" CACHE FILEPATH "" FORCE)
endif()
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)

# Everything that needs the SoC SDK takes it from here.
set(MIPK_PROFILE "hi3516cv610" CACHE STRING "Resource profile" FORCE)
