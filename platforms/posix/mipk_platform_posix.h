#ifndef MIPK_PLATFORM_POSIX_H
#define MIPK_PLATFORM_POSIX_H

#include "mipk_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mipk_posix_platform mipk_posix_platform;

typedef struct {
    uint32_t struct_size;
    const mipk_platform_arbiter *shared_arbiter;
} mipk_posix_platform_options;

mipk_status mipk_platform_posix_create(
    const mipk_posix_platform_options *options,
    mipk_posix_platform **platform);
const mipk_platform_context *mipk_platform_posix_context(
    const mipk_posix_platform *platform);
mipk_status mipk_platform_posix_destroy(mipk_posix_platform *platform);

#ifdef __cplusplus
}
#endif
#endif
