#ifndef MIPK_PLATFORM_H
#define MIPK_PLATFORM_H

#include "mipk_capability.h"
#include "mipk_runtime.h"
#include "mipk_value.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * What image_preprocess asks a hardware image backend to do: take the
 * frame (or a view of it), letterbox it into a target of the given size and
 * write the model input, either as raw NV12 bytes or as planar float.
 * A backend that cannot honour every field says so in supports() and the
 * operator falls back to its software path for that request.
 */
typedef enum {
    MIPK_IMAGE_OUTPUT_NV12 = 0,
    MIPK_IMAGE_OUTPUT_FLOAT_NCHW,
} mipk_image_output_kind;

typedef struct mipk_image_request {
    const mipk_input *input;
    /* NULL for the whole frame; otherwise the region to sample. */
    const mipk_image_view *view;
    /* A mipk_data_format value (mipk.h is layered above this header). */
    int32_t source_format;
    uint32_t source_width;
    uint32_t source_height;
    uint32_t target_width;
    uint32_t target_height;
    /* Letterbox geometry inside the target; the rest is filled with pad. */
    uint32_t resized_width;
    uint32_t resized_height;
    uint32_t pad_x;
    uint32_t pad_y;
    uint8_t pad_value;
    mipk_image_output_kind output;
    /* Float output only: channel order and per-channel normalisation. */
    bool output_rgb;
    float mean[3];
    float scale[3];
} mipk_image_request;

typedef struct {
    const mipk_capability_descriptor *capability;
    void *context;
    bool (*supports)(void *context, const mipk_image_request *request);
    mipk_status (*process)(void *context,
        const mipk_image_request *request, mipk_mutable_tensor *output);
} mipk_image_backend_ops;

typedef struct {
    uint32_t struct_size;
    void *context;
    mipk_status (*lock)(void *context);
    void (*unlock)(void *context);
} mipk_platform_arbiter;

typedef struct mipk_platform_context {
    uint32_t struct_size;
    void *context;
    mipk_status (*retain)(void *context);
    void (*release)(void *context);
    const mipk_image_backend_ops *image_backend;
    mipk_status (*buffer_alloc)(void *context, size_t size,
        uint32_t alignment, mipk_memory_type memory_type,
        mipk_buffer *buffer);
    void (*buffer_free)(void *context, mipk_buffer *buffer);
    mipk_status (*sync_for_device)(void *context, mipk_buffer *buffer,
        size_t offset, size_t size);
    mipk_status (*sync_for_cpu)(void *context, mipk_buffer *buffer,
        size_t offset, size_t size);
    const mipk_platform_arbiter *shared_arbiter;
} mipk_platform_context;

#ifdef __cplusplus
}
#endif
#endif
