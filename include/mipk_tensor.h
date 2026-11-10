#ifndef MIPK_TENSOR_H
#define MIPK_TENSOR_H

#include "mipk_runtime.h"

static inline const void *mipk_tensor_data(const mipk_tensor *tensor)
{
    if (tensor == NULL || tensor->buffer == NULL ||
        tensor->offset > tensor->buffer->desc.size) return NULL;
    uint8_t *base = tensor->buffer->desc.virtual_address;
    return base == NULL ? NULL : base + tensor->offset;
}

static inline void *mipk_mutable_tensor_data(const mipk_mutable_tensor *tensor)
{
    return (void *)mipk_tensor_data(tensor);
}

static inline size_t mipk_tensor_row_stride(const mipk_tensor *tensor)
{
    return tensor == NULL || tensor->dim_count == 0 ? 0 :
        tensor->strides[tensor->dim_count - 1U];
}

static inline bool mipk_tensor_range_valid(const mipk_tensor *tensor)
{
    return tensor != NULL && tensor->buffer != NULL &&
        tensor->offset <= tensor->buffer->desc.size &&
        tensor->byte_size <= tensor->buffer->desc.size - tensor->offset;
}

mipk_status mipk_tensor_float_count(const mipk_tensor *tensor,
    size_t *value_count);
mipk_status mipk_tensor_to_float_buffer(const mipk_tensor *tensor,
    float *values, size_t capacity, size_t *value_count);

/*
 * The last two dimensions of a tensor, a row or a value at a time.
 *
 * A model's output is written by the NPU into memory the CPU does not cache,
 * so converting all of it costs a read of every byte -- on the 640x480 pose
 * model, 1.9 MB and 3.3 ms of a 55 ms frame.  A decoder that scans one row to
 * decide which anchors matter and then reads a handful of values from each
 * pays for what it looks at instead.  Neither supports NC1HWC0, whose
 * interleaving a caller cannot address this way; both refuse it.
 */
mipk_status mipk_tensor_row_to_float(const mipk_tensor *tensor, size_t row,
    float *values, size_t capacity);
mipk_status mipk_tensor_float_at(const mipk_tensor *tensor, size_t row,
    size_t column, float *value);
uint16_t mipk_float_to_half(float value);

/* "[1, 84, 8400] float32" for diagnostics; always NUL terminated. */
const char *mipk_tensor_describe(const mipk_tensor *tensor, char *buffer,
    size_t size);

#endif
