#ifndef MIPK_CONTAINER_H
#define MIPK_CONTAINER_H

#include "mipk_runtime.h"

#include <stdbool.h>
#include <stdint.h>

#define MIPK_CONTAINER_MAGIC_SIZE 8U

/*
 * A .mipk package is a flat, unencrypted archive: a fixed header, one entry
 * per model, the manifest JSON, then the model bytes back to back.  A
 * directory holding manifest.json and the model files is accepted in the
 * same places, so a recipe can be run straight from its source tree.
 */
typedef struct {
    const char *path;
    uint64_t offset;
    size_t size;
    uint32_t crc32;
} mipk_container_model_source;

/* Reads a packed .mipk file. */
mipk_status mipk_container_open(const char *path, char **manifest,
    mipk_container_model_source **model_sources, size_t *model_count);
/* Reads <directory>/manifest.json; models are resolved by the manifest. */
mipk_status mipk_container_open_directory(const char *directory,
    char **manifest);
void mipk_container_free_sources(mipk_container_model_source *model_sources);
mipk_status mipk_container_load_model(void *context, void *destination,
    size_t destination_size);
uint32_t mipk_container_crc32(uint32_t seed, const void *data, size_t size);

#endif
