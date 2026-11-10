#define _POSIX_C_SOURCE 200112L

#include "mipk_platform_hisilicon.h"

#include "ss_mpi_sys_mem.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

typedef struct hisilicon_allocation {
    void *memory;
    td_phys_addr_t physical;
    bool mmz;
    struct hisilicon_allocation *next;
} hisilicon_allocation;

struct mipk_hisilicon_platform {
    pthread_mutex_t lock;
    pthread_mutex_t npu_lock;
    uint32_t references;
    hisilicon_allocation *allocations;
    char *mmz_zone;
    mipk_platform_arbiter own_arbiter;
    mipk_platform_context public_context;
};

static bool valid_alignment(uint32_t alignment)
{
    return alignment >= sizeof(void *) &&
        (alignment & (alignment - 1U)) == 0;
}

static mipk_status hisilicon_retain(void *private_data)
{
    mipk_hisilicon_platform *platform = private_data;
    if (platform == NULL || pthread_mutex_lock(&platform->lock) != 0) {
        return MIPK_ERR_RUNTIME;
    }
    mipk_status status = MIPK_OK;
    if (platform->references == UINT32_MAX) status = MIPK_ERR_CAPACITY;
    else ++platform->references;
    (void)pthread_mutex_unlock(&platform->lock);
    return status;
}

static void hisilicon_release(void *private_data)
{
    mipk_hisilicon_platform *platform = private_data;
    if (platform == NULL || pthread_mutex_lock(&platform->lock) != 0) return;
    if (platform->references > 1U) --platform->references;
    (void)pthread_mutex_unlock(&platform->lock);
}

static mipk_status hisilicon_buffer_alloc(void *private_data, size_t size,
    uint32_t alignment, mipk_memory_type memory_type, mipk_buffer *buffer)
{
    mipk_hisilicon_platform *platform = private_data;
    if (platform == NULL || size == 0 || buffer == NULL) return MIPK_ERR_ARGUMENT;
    *buffer = (mipk_buffer) {0};
    if (alignment == 0) alignment = (uint32_t)sizeof(void *);
    if (!valid_alignment(alignment)) return MIPK_ERR_ARGUMENT;
    hisilicon_allocation *allocation = calloc(1, sizeof(*allocation));
    if (allocation == NULL) return MIPK_ERR_MEMORY;
    if (memory_type == MIPK_MEMORY_CPU) {
        if (posix_memalign(&allocation->memory, alignment, size) != 0) {
            free(allocation);
            return MIPK_ERR_MEMORY;
        }
    } else {
        /* MMZ blocks are page aligned, which satisfies any sane request. */
        if (size > UINT32_MAX || alignment > 4096U) {
            free(allocation);
            return MIPK_ERR_ARGUMENT;
        }
        const td_s32 result = memory_type == MIPK_MEMORY_PHYS_CACHED ?
            ss_mpi_sys_mmz_alloc_cached(&allocation->physical,
                &allocation->memory, "mipk", platform->mmz_zone, (td_u32)size) :
            ss_mpi_sys_mmz_alloc(&allocation->physical, &allocation->memory,
                "mipk", platform->mmz_zone, (td_u32)size);
        if (result != TD_SUCCESS || allocation->memory == NULL) {
            mipk_diag("hisilicon: MMZ allocation of %zu bytes failed (0x%x); "
                "is ss_mpi_sys_init() done and the zone large enough?", size,
                (unsigned)result);
            free(allocation);
            return MIPK_ERR_MEMORY;
        }
        allocation->mmz = true;
    }
    if (pthread_mutex_lock(&platform->lock) != 0) {
        if (allocation->mmz) {
            (void)ss_mpi_sys_mmz_free(allocation->physical, allocation->memory);
        } else {
            free(allocation->memory);
        }
        free(allocation);
        return MIPK_ERR_RUNTIME;
    }
    allocation->next = platform->allocations;
    platform->allocations = allocation;
    (void)pthread_mutex_unlock(&platform->lock);
    buffer->desc = (mipk_buffer_desc) {
        .virtual_address = allocation->memory,
        .physical_address = allocation->mmz ?
            (uint64_t)allocation->physical : MIPK_PHYSICAL_ADDRESS_INVALID,
        .size = size,
        .alignment = allocation->mmz ? 4096U : alignment,
        .memory_type = memory_type,
        .owner = MIPK_BUFFER_OWNER_BACKEND,
    };
    buffer->sync_state = MIPK_BUFFER_SYNCED;
    return MIPK_OK;
}

static void hisilicon_buffer_free(void *private_data, mipk_buffer *buffer)
{
    mipk_hisilicon_platform *platform = private_data;
    if (platform == NULL || buffer == NULL ||
        buffer->desc.owner != MIPK_BUFFER_OWNER_BACKEND) return;
    if (pthread_mutex_lock(&platform->lock) != 0) return;
    hisilicon_allocation **link = &platform->allocations;
    while (*link != NULL && (*link)->memory != buffer->desc.virtual_address) {
        link = &(*link)->next;
    }
    hisilicon_allocation *allocation = *link;
    if (allocation != NULL) *link = allocation->next;
    (void)pthread_mutex_unlock(&platform->lock);
    if (allocation == NULL) return;
    if (allocation->mmz) {
        (void)ss_mpi_sys_mmz_free(allocation->physical, allocation->memory);
    } else {
        free(allocation->memory);
    }
    free(allocation);
    *buffer = (mipk_buffer) {0};
}

/*
 * The MPI offers one cache operation, a clean-and-invalidate of the range,
 * which serves both directions: before the device reads, and before the CPU
 * reads what the device wrote.  CPU and uncached memory need nothing.
 */
static mipk_status hisilicon_sync(void *private_data, mipk_buffer *buffer,
    size_t offset, size_t size)
{
    if (private_data == NULL || buffer == NULL ||
        offset > buffer->desc.size || size > buffer->desc.size - offset) {
        return MIPK_ERR_ARGUMENT;
    }
    if (buffer->desc.memory_type == MIPK_MEMORY_PHYS_CACHED && size != 0 &&
        buffer->desc.physical_address != MIPK_PHYSICAL_ADDRESS_INVALID) {
        if (size > UINT32_MAX) return MIPK_ERR_ARGUMENT;
        if (ss_mpi_sys_flush_cache(
                (td_phys_addr_t)(buffer->desc.physical_address + offset),
                (uint8_t *)buffer->desc.virtual_address + offset,
                (td_u32)size) != TD_SUCCESS) {
            return MIPK_ERR_RUNTIME;
        }
    }
    buffer->sync_state = MIPK_BUFFER_SYNCED;
    return MIPK_OK;
}

static mipk_status npu_lock(void *context)
{
    return pthread_mutex_lock(context) == 0 ? MIPK_OK : MIPK_ERR_RUNTIME;
}

static void npu_unlock(void *context)
{
    (void)pthread_mutex_unlock(context);
}

mipk_status mipk_platform_hisilicon_create(
    const mipk_hisilicon_platform_options *options,
    mipk_hisilicon_platform **output)
{
    if (options == NULL || options->struct_size != sizeof(*options) ||
        output == NULL) return MIPK_ERR_ARGUMENT;
    *output = NULL;
    mipk_hisilicon_platform *platform = calloc(1, sizeof(*platform));
    if (platform == NULL) return MIPK_ERR_MEMORY;
    if (pthread_mutex_init(&platform->lock, NULL) != 0) {
        free(platform);
        return MIPK_ERR_RUNTIME;
    }
    if (pthread_mutex_init(&platform->npu_lock, NULL) != 0) {
        (void)pthread_mutex_destroy(&platform->lock);
        free(platform);
        return MIPK_ERR_RUNTIME;
    }
    if (options->mmz_zone != NULL) {
        platform->mmz_zone = strdup(options->mmz_zone);
        if (platform->mmz_zone == NULL) {
            (void)pthread_mutex_destroy(&platform->npu_lock);
            (void)pthread_mutex_destroy(&platform->lock);
            free(platform);
            return MIPK_ERR_MEMORY;
        }
    }
    platform->references = 1;
    platform->own_arbiter = (mipk_platform_arbiter) {
        .struct_size = sizeof(mipk_platform_arbiter),
        .context = &platform->npu_lock,
        .lock = npu_lock,
        .unlock = npu_unlock,
    };
    platform->public_context = (mipk_platform_context) {
        .struct_size = sizeof(mipk_platform_context),
        .context = platform,
        .retain = hisilicon_retain,
        .release = hisilicon_release,
        .buffer_alloc = hisilicon_buffer_alloc,
        .buffer_free = hisilicon_buffer_free,
        .sync_for_device = hisilicon_sync,
        .sync_for_cpu = hisilicon_sync,
        .image_backend = options->image_backend,
        .shared_arbiter = options->shared_arbiter != NULL ?
            options->shared_arbiter : &platform->own_arbiter,
    };
    *output = platform;
    return MIPK_OK;
}

const mipk_platform_context *mipk_platform_hisilicon_context(
    const mipk_hisilicon_platform *platform)
{
    return platform == NULL ? NULL : &platform->public_context;
}

mipk_status mipk_platform_hisilicon_destroy(mipk_hisilicon_platform *platform)
{
    if (platform == NULL) return MIPK_ERR_ARGUMENT;
    if (pthread_mutex_lock(&platform->lock) != 0) return MIPK_ERR_RUNTIME;
    if (platform->references != 1U || platform->allocations != NULL) {
        (void)pthread_mutex_unlock(&platform->lock);
        return MIPK_ERR_BUSY;
    }
    platform->references = 0;
    (void)pthread_mutex_unlock(&platform->lock);
    (void)pthread_mutex_destroy(&platform->npu_lock);
    (void)pthread_mutex_destroy(&platform->lock);
    free(platform->mmz_zone);
    free(platform);
    return MIPK_OK;
}
