#define _POSIX_C_SOURCE 200112L

#include "mipk_platform_posix.h"

#include <pthread.h>
#include <stdlib.h>

#ifndef mipk_aligned_free
#define mipk_aligned_free free
#endif

typedef struct posix_allocation {
    void *memory;
    struct posix_allocation *next;
} posix_allocation;

struct mipk_posix_platform {
    pthread_mutex_t lock;
    uint32_t references;
    posix_allocation *allocations;
    mipk_platform_context public_context;
};

static bool valid_alignment(uint32_t alignment)
{
    return alignment >= sizeof(void *) &&
        (alignment & (alignment - 1U)) == 0;
}

static mipk_status posix_retain(void *private_data)
{
    mipk_posix_platform *platform = private_data;
    if (platform == NULL || pthread_mutex_lock(&platform->lock) != 0) {
        return MIPK_ERR_RUNTIME;
    }
    mipk_status status = MIPK_OK;
    if (platform->references == UINT32_MAX) status = MIPK_ERR_CAPACITY;
    else ++platform->references;
    (void)pthread_mutex_unlock(&platform->lock);
    return status;
}

static void posix_release(void *private_data)
{
    mipk_posix_platform *platform = private_data;
    if (platform == NULL || pthread_mutex_lock(&platform->lock) != 0) return;
    if (platform->references > 1U) --platform->references;
    (void)pthread_mutex_unlock(&platform->lock);
}

static mipk_status posix_buffer_alloc(void *private_data, size_t size,
    uint32_t alignment, mipk_memory_type memory_type, mipk_buffer *buffer)
{
    if (private_data == NULL || size == 0 || buffer == NULL ||
        memory_type != MIPK_MEMORY_CPU) return MIPK_ERR_ARGUMENT;
    *buffer = (mipk_buffer) {0};
    if (alignment == 0) alignment = (uint32_t)sizeof(void *);
    if (!valid_alignment(alignment)) return MIPK_ERR_ARGUMENT;
    void *memory = NULL;
    if (posix_memalign(&memory, alignment, size) != 0) return MIPK_ERR_MEMORY;
    posix_allocation *allocation = malloc(sizeof(*allocation));
    if (allocation == NULL) {
        free(memory);
        return MIPK_ERR_MEMORY;
    }
    mipk_posix_platform *platform = private_data;
    if (pthread_mutex_lock(&platform->lock) != 0) {
        free(allocation);
        free(memory);
        return MIPK_ERR_RUNTIME;
    }
    allocation->memory = memory;
    allocation->next = platform->allocations;
    platform->allocations = allocation;
    (void)pthread_mutex_unlock(&platform->lock);
    buffer->desc = (mipk_buffer_desc) {
        .virtual_address = memory,
        .physical_address = MIPK_PHYSICAL_ADDRESS_INVALID,
        .size = size,
        .alignment = alignment,
        .memory_type = MIPK_MEMORY_CPU,
        .owner = MIPK_BUFFER_OWNER_BACKEND,
    };
    buffer->sync_state = MIPK_BUFFER_SYNCED;
    return MIPK_OK;
}

static void posix_buffer_free(void *private_data, mipk_buffer *buffer)
{
    if (private_data == NULL || buffer == NULL ||
        buffer->desc.owner != MIPK_BUFFER_OWNER_BACKEND ||
        buffer->desc.memory_type != MIPK_MEMORY_CPU) return;
    mipk_posix_platform *platform = private_data;
    if (pthread_mutex_lock(&platform->lock) != 0) return;
    posix_allocation **link = &platform->allocations;
    while (*link != NULL &&
        (*link)->memory != buffer->desc.virtual_address) {
        link = &(*link)->next;
    }
    if (*link == NULL) {
        (void)pthread_mutex_unlock(&platform->lock);
        return;
    }
    posix_allocation *allocation = *link;
    *link = allocation->next;
    (void)pthread_mutex_unlock(&platform->lock);
    mipk_aligned_free(allocation->memory);
    free(allocation);
    *buffer = (mipk_buffer) {0};
}

static mipk_status validate_sync_range(void *private_data,
    mipk_buffer *buffer, size_t offset, size_t size)
{
    if (private_data == NULL || buffer == NULL ||
        buffer->desc.memory_type != MIPK_MEMORY_CPU ||
        offset > buffer->desc.size || size > buffer->desc.size - offset) {
        return MIPK_ERR_ARGUMENT;
    }
    buffer->sync_state = MIPK_BUFFER_SYNCED;
    return MIPK_OK;
}

mipk_status mipk_platform_posix_create(
    const mipk_posix_platform_options *options,
    mipk_posix_platform **output)
{
    if (options == NULL || options->struct_size != sizeof(*options) ||
        output == NULL) return MIPK_ERR_ARGUMENT;
    *output = NULL;
    mipk_posix_platform *platform = calloc(1, sizeof(*platform));
    if (platform == NULL) return MIPK_ERR_MEMORY;
    if (pthread_mutex_init(&platform->lock, NULL) != 0) {
        free(platform);
        return MIPK_ERR_RUNTIME;
    }
    platform->references = 1;
    platform->public_context = (mipk_platform_context) {
        .struct_size = sizeof(mipk_platform_context),
        .context = platform,
        .retain = posix_retain,
        .release = posix_release,
        .buffer_alloc = posix_buffer_alloc,
        .buffer_free = posix_buffer_free,
        .sync_for_device = validate_sync_range,
        .sync_for_cpu = validate_sync_range,
        .shared_arbiter = options->shared_arbiter,
    };
    *output = platform;
    return MIPK_OK;
}

const mipk_platform_context *mipk_platform_posix_context(
    const mipk_posix_platform *platform)
{
    return platform == NULL ? NULL : &platform->public_context;
}

mipk_status mipk_platform_posix_destroy(mipk_posix_platform *platform)
{
    if (platform == NULL) return MIPK_ERR_ARGUMENT;
    if (pthread_mutex_lock(&platform->lock) != 0) return MIPK_ERR_RUNTIME;
    if (platform->references != 1U) {
        (void)pthread_mutex_unlock(&platform->lock);
        return MIPK_ERR_BUSY;
    }
    if (platform->allocations != NULL) {
        (void)pthread_mutex_unlock(&platform->lock);
        return MIPK_ERR_BUSY;
    }
    platform->references = 0;
    (void)pthread_mutex_unlock(&platform->lock);
    (void)pthread_mutex_destroy(&platform->lock);
    free(platform);
    return MIPK_OK;
}
