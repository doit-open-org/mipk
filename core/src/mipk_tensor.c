#include "mipk_tensor.h"

#include "mipk_memory.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


/*
 * IEEE half to single by moving bit fields.  Every half is exactly
 * representable as a single, so this is exact; it replaces an ldexpf() per
 * value, which on a soft-float build cost about 190 ns and made converting a
 * detector's output tensor dominate the frame.
 */
static float half_to_float(uint16_t value)
{
    const uint32_t sign = (uint32_t)(value & 0x8000U) << 16U;
    const uint32_t exponent = (value >> 10U) & 0x1fU;
    const uint32_t mantissa = value & 0x3ffU;
    uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            /* Subnormal: normalise the mantissa into single precision. */
            uint32_t shifted = mantissa;
            uint32_t single_exponent = 127U - 15U + 1U;
            while ((shifted & 0x400U) == 0) {
                shifted <<= 1U;
                --single_exponent;
            }
            bits = sign | (single_exponent << 23U) | ((shifted & 0x3ffU) << 13U);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000U | (mantissa << 13U);
    } else {
        bits = sign | ((exponent + 127U - 15U) << 23U) | (mantissa << 13U);
    }
    /* Union punning is defined in C11 and stays inline under securec. */
    const union {
        uint32_t bits;
        float value;
    } result = {.bits = bits};
    return result.value;
}

uint16_t mipk_float_to_half(float value)
{
    uint32_t bits = 0;
    if (mipk_memory_copy(&bits, sizeof(bits), &value, sizeof(value)) != 0) return 0;
    const uint32_t sign = (bits >> 16U) & 0x8000U;
    int32_t exponent = (int32_t)((bits >> 23U) & 0xffU) - 127 + 15;
    uint32_t mantissa = bits & 0x7fffffU;
    if (exponent <= 0) {
        if (exponent < -10) return (uint16_t)sign;
        mantissa = (mantissa | 0x800000U) >> (1 - exponent);
        return (uint16_t)(sign | ((mantissa + 0x1000U) >> 13U));
    }
    if (exponent >= 31) return (uint16_t)(sign | 0x7c00U);
    return (uint16_t)(sign | ((uint32_t)exponent << 10U) |
        ((mantissa + 0x1000U) >> 13U));
}

static size_t tensor_element_size(const mipk_tensor *tensor)
{
    if (tensor->data_type == MIPK_DTYPE_FLOAT16) return sizeof(uint16_t);
    if (tensor->data_type == MIPK_DTYPE_FLOAT32) return sizeof(float);
    return 0;
}

static float tensor_value(const mipk_tensor *tensor, size_t index)
{
    const void *data = mipk_tensor_data(tensor);
    return tensor->data_type == MIPK_DTYPE_FLOAT16 ?
        half_to_float(((const uint16_t *)data)[index]) :
        ((const float *)data)[index];
}

static bool multiply_size(size_t left, size_t right, size_t *result)
{
    if (left != 0 && right > SIZE_MAX / left) return false;
    *result = left * right;
    return true;
}

mipk_status mipk_tensor_float_count(const mipk_tensor *tensor,
    size_t *value_count)
{
    if (tensor == NULL || value_count == NULL) return MIPK_ERR_ARGUMENT;
    *value_count = 0;
    const size_t element_size = tensor_element_size(tensor);
    if (element_size == 0) return MIPK_ERR_UNSUPPORTED;
    if (!mipk_tensor_range_valid(tensor) || tensor->byte_size == 0 ||
        tensor->byte_size % element_size != 0) return MIPK_ERR_MODEL;
    const size_t source_count = tensor->byte_size / element_size;

    if (tensor->format == MIPK_LAYOUT_NC1HWC0) {
        if (tensor->dim_count != 4) return MIPK_ERR_MODEL;
        size_t packed_count = 1;
        for (size_t i = 0; i < 4; ++i) {
            if (tensor->dims[i] <= 0 || !multiply_size(packed_count,
                    (size_t)tensor->dims[i], &packed_count)) return MIPK_ERR_MODEL;
        }
        const size_t c = (size_t)tensor->dims[1];
        const size_t c1 = (c + 15U) / 16U;
        size_t storage_count = (size_t)tensor->dims[0];
        if (!multiply_size(storage_count, c1, &storage_count) ||
            !multiply_size(storage_count, (size_t)tensor->dims[2], &storage_count) ||
            !multiply_size(storage_count, (size_t)tensor->dims[3], &storage_count) ||
            !multiply_size(storage_count, 16U, &storage_count) ||
            source_count < storage_count) return MIPK_ERR_MODEL;
        *value_count = packed_count;
        return MIPK_OK;
    }

    if ((tensor->format == MIPK_LAYOUT_NCHW ||
            tensor->format == MIPK_LAYOUT_ND) && tensor->dim_count != 0) {
        const int64_t last_dim = tensor->dims[tensor->dim_count - 1];
        const size_t row_stride = mipk_tensor_row_stride(tensor);
        if (last_dim > 0 && row_stride % element_size == 0) {
            const size_t width = (size_t)last_dim;
            const size_t stride_values = row_stride / element_size;
            size_t rows = 1;
            bool valid_dims = true;
            for (size_t i = 0; i + 1 < tensor->dim_count; ++i) {
                if (tensor->dims[i] <= 0 || !multiply_size(rows,
                        (size_t)tensor->dims[i], &rows)) {
                    valid_dims = false;
                    break;
                }
            }
            size_t storage_count = 0;
            size_t packed_count = 0;
            if (valid_dims && stride_values > width &&
                multiply_size(rows, stride_values, &storage_count) &&
                multiply_size(rows, width, &packed_count) &&
                source_count >= storage_count) {
                *value_count = packed_count;
                return MIPK_OK;
            }
        }
    }
    *value_count = source_count;
    return MIPK_OK;
}

mipk_status mipk_tensor_to_float_buffer(const mipk_tensor *tensor,
    float *values, size_t capacity, size_t *value_count)
{
    if (tensor == NULL || mipk_tensor_data(tensor) == NULL || values == NULL ||
        value_count == NULL) return MIPK_ERR_ARGUMENT;
    size_t count = 0;
    mipk_status status = mipk_tensor_float_count(tensor, &count);
    if (status != MIPK_OK) return status;
    if (capacity < count) return MIPK_ERR_CAPACITY;

    const size_t element_size = tensor_element_size(tensor);
    if (tensor->format == MIPK_LAYOUT_NC1HWC0) {
        const size_t n = (size_t)tensor->dims[0];
        const size_t c = (size_t)tensor->dims[1];
        const size_t h = (size_t)tensor->dims[2];
        const size_t w = (size_t)tensor->dims[3];
        const size_t c1 = (c + 15U) / 16U;
        for (size_t nn = 0; nn < n; ++nn) {
            for (size_t cc = 0; cc < c; ++cc) {
                for (size_t yy = 0; yy < h; ++yy) {
                    for (size_t xx = 0; xx < w; ++xx) {
                        const size_t source =
                            ((((nn * c1 + cc / 16U) * h + yy) * w + xx) * 16U) +
                            cc % 16U;
                        values[((nn * c + cc) * h + yy) * w + xx] =
                            tensor_value(tensor, source);
                    }
                }
            }
        }
    } else {
        size_t rows = 1;
        const size_t width = tensor->dim_count == 0 ? 0 :
            (size_t)(tensor->dims[tensor->dim_count - 1] > 0 ?
                tensor->dims[tensor->dim_count - 1] : 0);
        for (size_t i = 0; i + 1 < tensor->dim_count; ++i) {
            if (tensor->dims[i] <= 0 || !multiply_size(rows,
                    (size_t)tensor->dims[i], &rows)) {
                rows = 0;
                break;
            }
        }
        const size_t row_stride = mipk_tensor_row_stride(tensor);
        const size_t stride_values = row_stride / element_size;
        size_t packed_count = 0;
        const bool padded = (tensor->format == MIPK_LAYOUT_NCHW ||
            tensor->format == MIPK_LAYOUT_ND) && width != 0 &&
            row_stride % element_size == 0 && stride_values > width &&
            multiply_size(rows, width, &packed_count) && packed_count == count;
        if (padded) {
            for (size_t row = 0; row < rows; ++row) {
                for (size_t column = 0; column < width; ++column) {
                    values[row * width + column] =
                        tensor_value(tensor, row * stride_values + column);
                }
            }
        } else if (tensor->data_type == MIPK_DTYPE_FLOAT32) {
            if (mipk_memory_copy(values, capacity * sizeof(*values),
                    mipk_tensor_data(tensor),
                    count * sizeof(*values)) != 0) return MIPK_ERR_RUNTIME;
        } else {
            for (size_t i = 0; i < count; ++i) values[i] = tensor_value(tensor, i);
        }
    }
    *value_count = count;
    return MIPK_OK;
}


/*
 * Where a row starts and how many values it holds, for the two accessors
 * below.  The shape is the last two dimensions: everything before them is
 * counted into the row index, the way the rest of this file treats it.
 */
static mipk_status tensor_row_shape(const mipk_tensor *tensor, size_t row,
    size_t *first_value, size_t *width)
{
    if (tensor == NULL || mipk_tensor_data(tensor) == NULL ||
        tensor->format == MIPK_LAYOUT_NC1HWC0 || tensor->dim_count < 2) {
        return MIPK_ERR_ARGUMENT;
    }
    const size_t last = (size_t)(tensor->dims[tensor->dim_count - 1U] > 0 ?
        tensor->dims[tensor->dim_count - 1U] : 0);
    if (last == 0) return MIPK_ERR_MODEL;
    size_t rows = 1;
    for (size_t index = 0; index + 1 < tensor->dim_count; ++index) {
        if (tensor->dims[index] <= 0 ||
            !multiply_size(rows, (size_t)tensor->dims[index], &rows)) {
            return MIPK_ERR_MODEL;
        }
    }
    if (row >= rows) return MIPK_ERR_ARGUMENT;
    const size_t element_size = tensor_element_size(tensor);
    if (element_size == 0) return MIPK_ERR_MODEL;
    const size_t row_stride = mipk_tensor_row_stride(tensor);
    const size_t stride_values = row_stride == 0 ? last :
        row_stride / element_size;
    if (stride_values < last) return MIPK_ERR_MODEL;
    size_t first = 0;
    if (!multiply_size(row, stride_values, &first)) return MIPK_ERR_MODEL;
    size_t needed = 0;
    if (!multiply_size(first + last, element_size, &needed) ||
        needed > tensor->byte_size) {
        return MIPK_ERR_MODEL;
    }
    *first_value = first;
    *width = last;
    return MIPK_OK;
}

mipk_status mipk_tensor_row_to_float(const mipk_tensor *tensor, size_t row,
    float *values, size_t capacity)
{
    size_t first = 0;
    size_t width = 0;
    if (values == NULL) return MIPK_ERR_ARGUMENT;
    const mipk_status status = tensor_row_shape(tensor, row, &first, &width);
    if (status != MIPK_OK) return status;
    if (capacity < width) return MIPK_ERR_CAPACITY;
    if (tensor->data_type == MIPK_DTYPE_FLOAT32) {
        /* One burst out of uncached memory rather than a read per value. */
        if (mipk_memory_copy(values, capacity * sizeof(*values),
                (const float *)mipk_tensor_data(tensor) + first,
                width * sizeof(*values)) != 0) {
            return MIPK_ERR_RUNTIME;
        }
        return MIPK_OK;
    }
    for (size_t index = 0; index < width; ++index) {
        values[index] = tensor_value(tensor, first + index);
    }
    return MIPK_OK;
}

mipk_status mipk_tensor_float_at(const mipk_tensor *tensor, size_t row,
    size_t column, float *value)
{
    size_t first = 0;
    size_t width = 0;
    if (value == NULL) return MIPK_ERR_ARGUMENT;
    const mipk_status status = tensor_row_shape(tensor, row, &first, &width);
    if (status != MIPK_OK) return status;
    if (column >= width) return MIPK_ERR_ARGUMENT;
    *value = tensor_value(tensor, first + column);
    return MIPK_OK;
}

const char *mipk_tensor_describe(const mipk_tensor *tensor, char *buffer,
    size_t size)
{
    static const char *const names[] = {
        "float32", "float16", "int8", "int32", "uint8", "?", "int16", "uint16",
        "uint32", "int64", "uint64", "double", "bool",
    };
    if (buffer == NULL || size == 0) return "";
    if (tensor == NULL) {
        snprintf(buffer, size, "(none)");
        return buffer;
    }
    size_t used = (size_t)snprintf(buffer, size, "[");
    for (size_t index = 0; index < tensor->dim_count && used < size; ++index) {
        used += (size_t)snprintf(buffer + used, size - used, "%s%lld",
            index == 0 ? "" : ", ", (long long)tensor->dims[index]);
    }
    if (used < size) {
        const int32_t type = tensor->data_type;
        const char *name = type >= 0 &&
            (size_t)type < sizeof(names) / sizeof(names[0]) ? names[type] : "?";
        snprintf(buffer + used, size - used, "] %s", name);
    }
    buffer[size - 1U] = '\0';
    return buffer;
}
