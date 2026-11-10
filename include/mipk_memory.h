#ifndef MIPK_MEMORY_INTERNAL_H
#define MIPK_MEMORY_INTERNAL_H

#include <stddef.h>
#include <string.h>

#ifdef MIPK_USE_SECUREC
#include "securec.h"
#endif

static inline int mipk_memory_copy(void *destination, size_t destination_size,
    const void *source, size_t count)
{
    if (count > destination_size || (count != 0 &&
            (destination == NULL || source == NULL))) return -1;
#ifdef MIPK_USE_SECUREC
    return memcpy_s(destination, destination_size, source, count) == EOK ? 0 : -1;
#else
    if (count != 0) memcpy(destination, source, count);
    return 0;
#endif
}

static inline int mipk_memory_zero(void *destination, size_t destination_size)
{
    if (destination_size != 0 && destination == NULL) return -1;
#ifdef MIPK_USE_SECUREC
    return memset_s(destination, destination_size, 0, destination_size) == EOK ? 0 : -1;
#else
    if (destination_size != 0) memset(destination, 0, destination_size);
    return 0;
#endif
}

#endif
