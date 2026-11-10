#ifndef MIPK_PLATFORM_HISILICON_H
#define MIPK_PLATFORM_HISILICON_H

#include "mipk_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Platform for HiSilicon SoCs with MPP: physically contiguous buffers come
 * from MMZ, cache maintenance goes through the MPI, and one process-wide
 * arbiter serialises NPU submissions from every pipeline.  The MPP system
 * module must already be up (ss_mpi_sys_init) when buffers are requested;
 * a media framework normally does that.
 */
typedef struct mipk_hisilicon_platform mipk_hisilicon_platform;

typedef struct {
    uint32_t struct_size;
    /* Optional. NULL installs the platform's own NPU mutex. */
    const mipk_platform_arbiter *shared_arbiter;
    /* Optional MMZ zone name; NULL means the default zone. */
    const char *mmz_zone;
    /*
     * Optional hardware image preprocessing (VGS, IVE...).  It needs the
     * media framework's channels, so the application implements it and
     * injects it here; the platform itself stays independent of any media
     * stack.  Must outlive the platform.
     */
    const mipk_image_backend_ops *image_backend;
} mipk_hisilicon_platform_options;

mipk_status mipk_platform_hisilicon_create(
    const mipk_hisilicon_platform_options *options,
    mipk_hisilicon_platform **platform);
const mipk_platform_context *mipk_platform_hisilicon_context(
    const mipk_hisilicon_platform *platform);
mipk_status mipk_platform_hisilicon_destroy(mipk_hisilicon_platform *platform);

#ifdef __cplusplus
}
#endif
#endif
