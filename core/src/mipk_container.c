#include "mipk_container.h"

#include "mipk.h"
#include "mipk_config.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/*
 * Container v3 layout, all integers little endian:
 *
 *   header   32 bytes  magic "MIPK3\0\0\0", version u16, header_size u16,
 *                      flags u16, reserved u16, manifest_size u32,
 *                      entry_count u32, package_size u64
 *   entries  32 bytes  type u32, flags u32, offset u64, size u64, crc32 u32,
 *            each      index u32
 *   manifest manifest_size bytes of UTF-8 JSON
 *   payload  every model, in entry order, at the offsets the entries name
 */
#define MIPK_CONTAINER_HEADER_SIZE 32U
#define MIPK_CONTAINER_ENTRY_SIZE 32U
#define MIPK_CONTAINER_VERSION 3U
#define MIPK_ENTRY_MODEL 1U
#define MIPK_ENTRY_MAX_COUNT 64U

static const uint8_t g_container_magic[MIPK_CONTAINER_MAGIC_SIZE] =
    {'M', 'I', 'P', 'K', '3', 0, 0, 0};

static uint16_t read_u16_le(const uint8_t *value)
{
    return (uint16_t)value[0] | (uint16_t)((uint16_t)value[1] << 8U);
}

static uint32_t read_u32_le(const uint8_t *value)
{
    return (uint32_t)value[0] | ((uint32_t)value[1] << 8U) |
        ((uint32_t)value[2] << 16U) | ((uint32_t)value[3] << 24U);
}

static uint64_t read_u64_le(const uint8_t *value)
{
    return (uint64_t)read_u32_le(value) |
        ((uint64_t)read_u32_le(value + 4) << 32U);
}

static mipk_status read_exact(FILE *file, void *data, size_t size)
{
    return fread(data, 1, size, file) == size ? MIPK_OK : MIPK_ERR_IO;
}

uint32_t mipk_container_crc32(uint32_t seed, const void *data, size_t size)
{
    static uint32_t table[256];
    static bool ready;
    if (!ready) {
        for (uint32_t index = 0; index < 256U; ++index) {
            uint32_t value = index;
            for (int bit = 0; bit < 8; ++bit) {
                value = (value & 1U) != 0 ? (value >> 1U) ^ 0xEDB88320U :
                    value >> 1U;
            }
            table[index] = value;
        }
        ready = true;
    }
    uint32_t crc = seed ^ 0xFFFFFFFFU;
    const uint8_t *bytes = data;
    for (size_t index = 0; index < size; ++index) {
        crc = table[(crc ^ bytes[index]) & 0xFFU] ^ (crc >> 8U);
    }
    return crc ^ 0xFFFFFFFFU;
}

static mipk_status file_size_matches(const char *path, uint64_t expected)
{
    struct stat attributes;
    if (stat(path, &attributes) != 0 || !S_ISREG(attributes.st_mode) ||
        attributes.st_size < 0 || (uint64_t)attributes.st_size != expected) {
        return MIPK_ERR_MODEL;
    }
    return MIPK_OK;
}

mipk_status mipk_container_open(const char *path, char **manifest,
    mipk_container_model_source **model_sources, size_t *model_count)
{
    if (path == NULL || manifest == NULL || model_sources == NULL ||
        model_count == NULL) {
        return MIPK_ERR_ARGUMENT;
    }
    *manifest = NULL;
    *model_sources = NULL;
    *model_count = 0;
    FILE *file = fopen(path, "rb");
    if (file == NULL) return MIPK_ERR_IO;
    uint8_t header[MIPK_CONTAINER_HEADER_SIZE];
    uint8_t *entries = NULL;
    char *content = NULL;
    mipk_container_model_source *sources = NULL;
    mipk_status status = read_exact(file, header, sizeof(header));
    if (status != MIPK_OK) goto done;
    const uint16_t version = read_u16_le(header + 8);
    const uint16_t header_size = read_u16_le(header + 10);
    const uint16_t flags = read_u16_le(header + 12);
    const uint32_t manifest_size = read_u32_le(header + 16);
    const uint32_t entry_count = read_u32_le(header + 20);
    const uint64_t package_size = read_u64_le(header + 24);
    if (memcmp(header, g_container_magic, sizeof(g_container_magic)) != 0 ||
        version != MIPK_CONTAINER_VERSION || header_size != sizeof(header) ||
        flags != 0 || manifest_size == 0 ||
        manifest_size > MIPK_CONTAINER_MANIFEST_MAX_SIZE ||
        entry_count == 0 || entry_count > MIPK_ENTRY_MAX_COUNT) {
        status = MIPK_ERR_MODEL;
        goto done;
    }
    const size_t entries_size = (size_t)entry_count * MIPK_CONTAINER_ENTRY_SIZE;
    const uint64_t payload_start = sizeof(header) + entries_size +
        (uint64_t)manifest_size;
    if (package_size < payload_start) {
        status = MIPK_ERR_MODEL;
        goto done;
    }
    status = file_size_matches(path, package_size);
    if (status != MIPK_OK) goto done;

    entries = malloc(entries_size);
    content = malloc((size_t)manifest_size + 1U);
    sources = calloc(entry_count, sizeof(*sources));
    if (entries == NULL || content == NULL || sources == NULL) {
        status = MIPK_ERR_MEMORY;
        goto done;
    }
    status = read_exact(file, entries, entries_size);
    if (status == MIPK_OK) status = read_exact(file, content, manifest_size);
    if (status != MIPK_OK) goto done;
    content[manifest_size] = '\0';

    uint64_t expected_offset = payload_start;
    for (uint32_t index = 0; index < entry_count; ++index) {
        const uint8_t *entry = entries + (size_t)index *
            MIPK_CONTAINER_ENTRY_SIZE;
        const uint32_t type = read_u32_le(entry);
        const uint32_t entry_flags = read_u32_le(entry + 4);
        const uint64_t offset = read_u64_le(entry + 8);
        const uint64_t size = read_u64_le(entry + 16);
        const uint32_t crc = read_u32_le(entry + 24);
        const uint32_t logical_index = read_u32_le(entry + 28);
        if (type != MIPK_ENTRY_MODEL || entry_flags != 0 ||
            offset != expected_offset || size == 0 || size > SIZE_MAX ||
            offset > package_size || size > package_size - offset ||
            logical_index != index) {
            status = MIPK_ERR_MODEL;
            goto done;
        }
        sources[index].path = path;
        sources[index].offset = offset;
        sources[index].size = (size_t)size;
        sources[index].crc32 = crc;
        expected_offset += size;
    }
    if (expected_offset != package_size) {
        status = MIPK_ERR_MODEL;
        goto done;
    }
    *manifest = content;
    *model_sources = sources;
    *model_count = entry_count;
    content = NULL;
    sources = NULL;
done:
    free(entries);
    free(content);
    free(sources);
    fclose(file);
    return status;
}

mipk_status mipk_container_open_directory(const char *directory,
    char **manifest)
{
    if (directory == NULL || manifest == NULL) return MIPK_ERR_ARGUMENT;
    *manifest = NULL;
    const size_t directory_length = strlen(directory);
    static const char suffix[] = "/manifest.json";
    char *path = malloc(directory_length + sizeof(suffix));
    if (path == NULL) return MIPK_ERR_MEMORY;
    memcpy(path, directory, directory_length);
    memcpy(path + directory_length, suffix, sizeof(suffix));
    FILE *file = fopen(path, "rb");
    free(path);
    if (file == NULL) return MIPK_ERR_IO;
    mipk_status status = MIPK_OK;
    char *content = NULL;
    if (fseek(file, 0, SEEK_END) != 0) {
        status = MIPK_ERR_IO;
        goto done;
    }
    const long length = ftell(file);
    if (length <= 0 || (unsigned long)length > MIPK_CONTAINER_MANIFEST_MAX_SIZE ||
        fseek(file, 0, SEEK_SET) != 0) {
        status = MIPK_ERR_MODEL;
        goto done;
    }
    content = malloc((size_t)length + 1U);
    if (content == NULL) {
        status = MIPK_ERR_MEMORY;
        goto done;
    }
    status = read_exact(file, content, (size_t)length);
    if (status != MIPK_OK) goto done;
    content[length] = '\0';
    *manifest = content;
    content = NULL;
done:
    free(content);
    fclose(file);
    return status;
}

void mipk_container_free_sources(mipk_container_model_source *model_sources)
{
    free(model_sources);
}

mipk_status mipk_container_load_model(void *context, void *destination,
    size_t destination_size)
{
    mipk_container_model_source *source = context;
    if (source == NULL || source->path == NULL || destination == NULL ||
        destination_size != source->size) return MIPK_ERR_ARGUMENT;
    FILE *file = fopen(source->path, "rb");
    if (file == NULL) return MIPK_ERR_IO;
    mipk_status status = MIPK_OK;
    if (source->offset > LONG_MAX ||
        fseek(file, (long)source->offset, SEEK_SET) != 0) {
        status = MIPK_ERR_IO;
    } else {
        status = read_exact(file, destination, source->size);
    }
    fclose(file);
    if (status == MIPK_OK && mipk_container_crc32(0, destination,
            source->size) != source->crc32) {
        return MIPK_ERR_MODEL;
    }
    return status;
}
