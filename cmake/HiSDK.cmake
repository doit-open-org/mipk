include_guard(GLOBAL)

# The HiSilicon SoC SDK is never vendored. MIPK_HI_SDK_ROOT (or the
# HI_SDK_ROOT environment variable) points at the unpacked SDK, whose
# soc/include and soc/libs/musl are used by the svp_acl backend and the
# hisilicon platform. mipk::hisdk is the interface target that carries them.
function(mipk_require_hisdk)
    if(TARGET mipk_hisdk)
        return()
    endif()
    if(NOT MIPK_HI_SDK_ROOT AND DEFINED ENV{HI_SDK_ROOT})
        set(MIPK_HI_SDK_ROOT "$ENV{HI_SDK_ROOT}" CACHE PATH "HiSilicon SDK root" FORCE)
    endif()
    if(NOT MIPK_HI_SDK_ROOT OR NOT EXISTS "${MIPK_HI_SDK_ROOT}/soc/include/svp_acl.h")
        message(FATAL_ERROR "HiSilicon SDK not found; set MIPK_HI_SDK_ROOT to a directory holding soc/include/svp_acl.h")
    endif()
    set(_libs "${MIPK_HI_SDK_ROOT}/soc/libs/musl")
    add_library(mipk_hisdk INTERFACE)
    add_library(mipk::hisdk ALIAS mipk_hisdk)
    target_include_directories(mipk_hisdk INTERFACE
        "${MIPK_HI_SDK_ROOT}/soc/include"
        "${MIPK_HI_SDK_ROOT}/soc/config/gcc")
    target_link_libraries(mipk_hisdk INTERFACE
        "${_libs}/libsvp_acl.so"
        "${_libs}/libprotobuf-c.so"
        "${_libs}/libsecurec.so"
        "${_libs}/libot_osal.so"
        "${_libs}/libot_base.so"
        "${_libs}/libot_irq.so"
        "${_libs}/libot_sys.so"
        "${_libs}/libss_mpi_sysmem.so"
        pthread dl m)
endfunction()
