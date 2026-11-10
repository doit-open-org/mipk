#include "mipk_module.h"

#include "mipk_memory.h"
#include "mipk_quad_sampler.h"
#include "mipk_tensor.h"
#include "cJSON.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>


typedef enum {
    CHANNEL_ORDER_BGR = 0,
    CHANNEL_ORDER_RGB,
} channel_order;

typedef enum {
    OUTPUT_AUTO = 0,
    OUTPUT_NV12,
    OUTPUT_FLOAT,
} output_mode;

typedef struct {
    size_t source_size;
    size_t target_size;
    uint32_t source_width;
    uint32_t source_height;
    uint32_t width;
    uint32_t height;
    uint32_t resized_width;
    uint32_t resized_height;
    uint32_t pad_x;
    uint32_t pad_y;
    int32_t input_format;
    channel_order input_order;
    channel_order output_order;
    float mean[3];
    float scale[3];
    float letterbox_value;
    int model_data_type;
    size_t model_row_stride;
    /* The source frame is NV12/NV21 rather than packed RGB/BGR. */
    bool yuv_source;
    /* Requested output: raw NV12 bytes for a model with AIPP, or planar float. */
    output_mode mode;
    /* Resolved against the model input in validate. */
    bool direct;
    /* Target-sized NV12 staging for the YUV to float path. */
    uint8_t *scratch;
    size_t scratch_size;
    /* Hardware path offered by the platform, if any. */
    const mipk_image_backend_ops *image_backend;
    /* Float output samples bilinearly unless the manifest says nearest. */
    bool bilinear;
} image_preprocess_context;

static void image_preprocess_destroy(void *private_data)
{
    image_preprocess_context *context = private_data;
    if (context == NULL) return;
    free(context->scratch);
    free(context);
}

static const char *json_string(const cJSON *json, const char *name,
    const char *default_value)
{
    const cJSON *item = cJSON_GetObjectItemCaseSensitive(json, name);
    if (item == NULL) return default_value;
    return cJSON_IsString(item) && item->valuestring != NULL ?
        item->valuestring : NULL;
}

static bool parse_channel_order(const char *value, channel_order *order)
{
    if (value != NULL && strcmp(value, "bgr") == 0) {
        *order = CHANNEL_ORDER_BGR;
        return true;
    }
    if (value != NULL && strcmp(value, "rgb") == 0) {
        *order = CHANNEL_ORDER_RGB;
        return true;
    }
    return false;
}

static bool parse_triplet(const cJSON *json, const char *name,
    float default_value, float values[3])
{
    const cJSON *item = cJSON_GetObjectItemCaseSensitive(json, name);
    if (item == NULL) {
        values[0] = default_value;
        values[1] = default_value;
        values[2] = default_value;
        return true;
    }
    if (cJSON_IsNumber(item) && isfinite(item->valuedouble)) {
        values[0] = (float)item->valuedouble;
        values[1] = values[0];
        values[2] = values[0];
        return true;
    }
    if (!cJSON_IsArray(item) || cJSON_GetArraySize(item) != 3) return false;
    for (int index = 0; index < 3; ++index) {
        const cJSON *value = cJSON_GetArrayItem(item, index);
        if (!cJSON_IsNumber(value) || !isfinite(value->valuedouble)) return false;
        values[index] = (float)value->valuedouble;
    }
    return true;
}

static void *image_preprocess_create(const mipk_operator_config *config,
    const char *options_json, mipk_status *status)
{
    if (status == NULL) return NULL;
    *status = MIPK_ERR_ARGUMENT;
    if (config == NULL || config->source_width == 0 ||
        config->source_height == 0 || config->input_width == 0 ||
        config->input_height == 0 || config->input_size == 0 ||
        options_json == NULL) return NULL;
    if (config->input_width > SIZE_MAX / config->input_height) return NULL;
    const size_t pixels = (size_t)config->input_width * config->input_height;
    if (pixels > SIZE_MAX / 3U) return NULL;
    cJSON *json = cJSON_Parse(options_json);
    const bool yuv_source = config->input_format == MIPK_FORMAT_NV12 ||
        config->input_format == MIPK_FORMAT_NV21;
    const char *layout = json_string(json, "layout", "nchw");
    const char *format = json_string(json, "format", "auto");
    /*
     * A YUV frame is converted, so its channels have no order; the output
     * defaults to RGB, which is what most exported models expect.  Packed
     * input keeps its own order unless the manifest says otherwise.
     */
    const char *default_input_order =
        config->input_format == MIPK_FORMAT_RGB ? "rgb" : "bgr";
    const char *input_order = json_string(json, "input_order",
        default_input_order);
    const char *output_order = json_string(json, "output_order",
        yuv_source ? "rgb" : input_order);
    const cJSON *letterbox_value = json == NULL ? NULL :
        cJSON_GetObjectItemCaseSensitive(json, "letterbox_value");
    const char *interpolation = json_string(json, "interpolation", "bilinear");
    image_preprocess_context *context = calloc(1, sizeof(*context));
    bool valid = json != NULL && context != NULL && layout != NULL &&
        format != NULL && input_order != NULL && output_order != NULL &&
        interpolation != NULL;
    if (valid) {
        context->yuv_source = yuv_source;
        if (strcmp(interpolation, "bilinear") == 0) context->bilinear = true;
        else if (strcmp(interpolation, "nearest") == 0) context->bilinear = false;
        else valid = false;
        if (strcmp(format, "auto") == 0) context->mode = OUTPUT_AUTO;
        else if (strcmp(format, "nv12") == 0) context->mode = OUTPUT_NV12;
        else if (strcmp(format, "float") == 0) context->mode = OUTPUT_FLOAT;
        else valid = false;
    }
    valid = valid && strcmp(layout, "nchw") == 0 &&
        parse_channel_order(input_order, &context->input_order) &&
        parse_channel_order(output_order, &context->output_order) &&
        parse_triplet(json, "mean", 0.0F, context->mean) &&
        parse_triplet(json, "scale", 1.0F, context->scale) &&
        (letterbox_value == NULL || (cJSON_IsNumber(letterbox_value) &&
            letterbox_value->valuedouble >= 0.0 &&
            letterbox_value->valuedouble <= 255.0));
    if (valid) {
        context->letterbox_value = letterbox_value == NULL ? 114.0F :
            (float)letterbox_value->valuedouble;
        for (size_t index = 0; index < 3; ++index) {
            if (!isfinite(context->mean[index]) ||
                !isfinite(context->scale[index])) valid = false;
        }
        /* NV12 output only makes sense for a YUV frame. */
        if (context->mode == OUTPUT_NV12 && !yuv_source) valid = false;
    }
    cJSON_Delete(json);
    if (!valid) {
        free(context);
        return NULL;
    }
    if ((size_t)config->source_width > SIZE_MAX / config->source_height) {
        free(context);
        return NULL;
    }
    const size_t source_pixels =
        (size_t)config->source_width * config->source_height;
    context->source_size = context->yuv_source ?
        source_pixels + source_pixels / 2U : source_pixels * 3U;
    context->target_size = config->input_size;
    context->source_width = config->source_width;
    context->source_height = config->source_height;
    context->width = config->input_width;
    context->height = config->input_height;
    context->resized_width = config->resized_width;
    context->resized_height = config->resized_height;
    context->pad_x = config->pad_x;
    context->pad_y = config->pad_y;
    context->input_format = config->input_format;
    context->image_backend = config->platform != NULL ?
        config->platform->image_backend : NULL;
    *status = MIPK_OK;
    return context;
}

static mipk_status image_preprocess_validate(void *private_data,
    const mipk_mutable_tensor *inputs, size_t input_count,
    const mipk_tensor *outputs, size_t output_count)
{
    image_preprocess_context *context = private_data;
    (void)outputs;
    (void)output_count;
    if (context == NULL || inputs == NULL || input_count == 0 ||
        mipk_mutable_tensor_data(&inputs[0]) == NULL) return MIPK_ERR_MODEL;
    const bool float_input = inputs[0].data_type == MIPK_DTYPE_FLOAT32 ||
        inputs[0].data_type == MIPK_DTYPE_FLOAT16;
    /*
     * A model whose input is bytes takes the NV12 frame as it is (its AIPP
     * does the colour conversion); a float input gets the planar path.
     * "format" in the manifest pins one or the other; left on auto, an input
     * exactly the size of the NV12 frame is taken as wanting the frame, since
     * no float tensor of the same geometry can be that small.
     */
    const bool frame_sized = inputs[0].byte_size == context->target_size;
    switch (context->mode) {
        case OUTPUT_NV12:
            context->direct = true;
            break;
        case OUTPUT_FLOAT:
            context->direct = false;
            break;
        default:
            context->direct = context->yuv_source &&
                (!float_input || frame_sized);
            break;
    }
    char shape[96];
    if (context->direct && !context->yuv_source) {
        mipk_diag("image_preprocess: \"format\": \"nv12\" needs an NV12/NV21 "
            "pipeline input");
        return MIPK_ERR_MODEL;
    }
    if (context->direct) {
        if (inputs[0].byte_size < context->target_size) {
            mipk_diag("image_preprocess: model input %s holds %zu bytes but "
                "an NV12 frame of %ux%u needs %zu",
                mipk_tensor_describe(&inputs[0], shape, sizeof(shape)),
                inputs[0].byte_size, context->width, context->height,
                context->target_size);
            return MIPK_ERR_MODEL;
        }
        return MIPK_OK;
    }
    if (!float_input) {
        mipk_diag("image_preprocess: model input %s is neither float32 nor "
            "float16; set \"format\": \"nv12\" for a byte input with AIPP",
            mipk_tensor_describe(&inputs[0], shape, sizeof(shape)));
        return MIPK_ERR_MODEL;
    }
    if (context->yuv_source) {
        const size_t pixels = (size_t)context->width * context->height;
        const size_t needed = pixels + pixels / 2U;
        uint8_t *scratch = malloc(needed);
        if (scratch == NULL) return MIPK_ERR_MEMORY;
        free(context->scratch);
        context->scratch = scratch;
        context->scratch_size = needed;
    }
    const size_t pixels = (size_t)context->width * context->height;
    const size_t element_size = inputs[0].data_type == MIPK_DTYPE_FLOAT32 ?
        sizeof(float) : sizeof(uint16_t);
    const size_t tensor_stride = mipk_tensor_row_stride(&inputs[0]);
    if (pixels > SIZE_MAX / 3U || tensor_stride % element_size != 0) {
        return MIPK_ERR_MODEL;
    }
    size_t row_stride = tensor_stride / element_size;
    if (row_stride < context->width) row_stride = context->width;
    const size_t rows = (size_t)context->height * 3U;
    if (row_stride > SIZE_MAX / rows ||
        row_stride * rows > SIZE_MAX / element_size ||
        inputs[0].byte_size < row_stride * rows * element_size) {
        return MIPK_ERR_MODEL;
    }
    if (inputs[0].dim_count == 4 &&
        (inputs[0].dims[0] != 1 || inputs[0].dims[1] != 3 ||
         inputs[0].dims[2] != context->height ||
         inputs[0].dims[3] != context->width)) {
        mipk_diag("image_preprocess: model input %s does not match the "
            "%ux%u target; set target_width/target_height to the model's "
            "input size", mipk_tensor_describe(&inputs[0], shape, sizeof(shape)),
            context->width, context->height);
        return MIPK_ERR_MODEL;
    }
    context->model_data_type = inputs[0].data_type;
    context->model_row_stride = row_stride;
    return MIPK_OK;
}

/*
 * Copies one row, picking source samples by stepping rather than dividing.
 *
 * The obvious form, `source[column * source_count / target_count]`, costs a
 * 64 bit division for every pixel written -- around 460 thousand of them per
 * frame here, and this ARM has no instruction for it, so each one is a call
 * into libgcc.  Accumulating the remainder instead gives exactly the same
 * sample for every column, with no division at all: the inner loop advances
 * the source position a whole number of samples at a time, `source_count`
 * times over the row however wide the row is.
 *
 * `stride` is 1 for luma and 2 for the interleaved chroma pairs.
 */
/* One row of the widest source this operator is expected to be handed. */
#define MIPK_PREPROCESS_STAGING_BYTES 2048U

static void copy_row_stepped(uint8_t *target, const uint8_t *source,
    uint32_t target_count, uint32_t source_count, uint32_t stride)
{
    uint32_t source_index = 0;
    uint32_t remainder = 0;
    for (uint32_t index = 0; index < target_count; ++index) {
        for (uint32_t part = 0; part < stride; ++part) {
            target[index * stride + part] =
                source[source_index * stride + part];
        }
        remainder += source_count;
        while (remainder >= target_count) {
            remainder -= target_count;
            ++source_index;
        }
    }
}

/* copy_row_stepped, with the source row staged in cacheable memory first. */
static void copy_row_scaled(uint8_t *target, const uint8_t *source,
    uint32_t target_count, uint32_t source_count, uint32_t stride,
    uint8_t *staging, size_t staging_size)
{
    const size_t bytes = (size_t)source_count * stride;
    if (bytes <= staging_size) {
        memcpy(staging, source, bytes);
        source = staging;
    }
    copy_row_stepped(target, source, target_count, source_count, stride);
}

static mipk_status resize_yuv(const image_preprocess_context *context,
    const mipk_video_frame *frame, uint8_t *destination)
{
    if (frame->plane_count != 2 || destination == NULL ||
        context->resized_width == 0 || context->resized_height == 0) {
        return MIPK_ERR_ARGUMENT;
    }
    const uint8_t *source_y = frame->planes[0].buffer.virtual_address;
    const uint8_t *source_uv = frame->planes[1].buffer.virtual_address;
    if (source_y == NULL || source_uv == NULL) return MIPK_ERR_UNSUPPORTED;
    uint8_t *destination_uv = destination +
        (size_t)context->width * context->height;
    /*
     * The grey and neutral fill is only what shows around the frame.  When the
     * frame covers the whole input -- the usual case once the camera is set up
     * to match the model -- every byte is overwritten below, and filling first
     * would write the largest buffer in the pipeline for nothing.
     */
    if (context->pad_x != 0 || context->pad_y != 0 ||
        context->resized_width != context->width ||
        context->resized_height != context->height) {
        memset(destination, 114, (size_t)context->width * context->height);
        memset(destination_uv, 128,
            (size_t)context->width * context->height / 2U);
    }

    /*
     * The source frame lives in video memory the CPU does not cache, so
     * reading it a byte at a time -- which is exactly what subsampling does
     * -- pays a memory round trip per pixel.  Measured on the palm stage,
     * which takes 640x480 down to 192x144, that was 7.4 ms of a 22 ms frame,
     * for an output of 27 thousand pixels.  Pulling each source row in with
     * one memcpy lets the read burst, and the subsampling then works out of
     * cache.  A row wider than this buffer falls back to reading in place.
     */
    uint8_t staging[MIPK_PREPROCESS_STAGING_BYTES];
    const bool same_width = context->resized_width == context->source_width;
    for (uint32_t row = 0; row < context->resized_height; ++row) {
        const uint32_t source_row = (uint64_t)row * context->source_height /
            context->resized_height;
        uint8_t *target = destination +
            (size_t)(context->pad_y + row) * context->width + context->pad_x;
        const uint8_t *source = source_y +
            (size_t)source_row * frame->planes[0].stride;
        if (same_width) {
            memcpy(target, source, context->resized_width);
        } else {
            copy_row_scaled(target, source, context->resized_width,
                context->source_width, 1U, staging, sizeof(staging));
        }
    }

    const uint32_t resized_uv_rows = context->resized_height / 2U;
    const uint32_t source_uv_rows = context->source_height / 2U;
    const uint32_t resized_uv_pairs = context->resized_width / 2U;
    const uint32_t source_uv_pairs = context->source_width / 2U;
    for (uint32_t row = 0; row < resized_uv_rows; ++row) {
        const uint32_t source_row =
            (uint64_t)row * source_uv_rows / resized_uv_rows;
        uint8_t *target = destination_uv +
            (size_t)(context->pad_y / 2U + row) * context->width +
            context->pad_x;
        const uint8_t *source = source_uv +
            (size_t)source_row * frame->planes[1].stride;
        if (resized_uv_pairs == source_uv_pairs) {
            memcpy(target, source, (size_t)resized_uv_pairs * 2U);
        } else {
            copy_row_scaled(target, source, resized_uv_pairs,
                source_uv_pairs, 2U, staging, sizeof(staging));
        }
    }
    return MIPK_OK;
}

static bool view_corners(const mipk_image_view *view, float corners[8])
{
    if (view == NULL || view->root_input == NULL) return false;
    if (view->point_count == 2) {
        const float x1 = view->region[0];
        const float y1 = view->region[1];
        const float x2 = view->region[2];
        const float y2 = view->region[3];
        if (!isfinite(x1) || !isfinite(y1) || !isfinite(x2) ||
            !isfinite(y2) || x2 <= x1 || y2 <= y1) return false;
        const float rectangle[8] = {x1, y1, x2, y1, x2, y2, x1, y2};
        memcpy(corners, rectangle, sizeof(rectangle));
        return true;
    }
    if (view->point_count != 4) return false;
    for (size_t index = 0; index < 8; ++index) {
        if (!isfinite(view->region[index])) return false;
        corners[index] = view->region[index];
    }
    return true;
}

static mipk_status resize_yuv_view(const image_preprocess_context *context,
    const mipk_video_frame *frame, const mipk_image_view *view,
    uint8_t *destination)
{
    float corners[8];
    if (frame == NULL || frame->plane_count != 2 || destination == NULL ||
        context->resized_width == 0 || context->resized_height == 0 ||
        !view_corners(view, corners)) return MIPK_ERR_ARGUMENT;
    const uint8_t *source_y = frame->planes[0].buffer.virtual_address;
    const uint8_t *source_uv = frame->planes[1].buffer.virtual_address;
    if (source_y == NULL || source_uv == NULL || frame->width < 2 ||
        frame->height < 2) return MIPK_ERR_UNSUPPORTED;
    memset(destination, 114, (size_t)context->width * context->height);
    uint8_t *destination_uv = destination +
        (size_t)context->width * context->height;
    memset(destination_uv, 128,
        (size_t)context->width * context->height / 2U);
    const mipk_nv12_source source = {
        .luma = source_y,
        .chroma = source_uv,
        .width = frame->width,
        .height = frame->height,
        .luma_stride = frame->planes[0].stride,
        .chroma_stride = frame->planes[1].stride,
    };
    mipk_quad_sample_luma(&source, corners,
        destination + (size_t)context->pad_y * context->width + context->pad_x,
        context->width, context->resized_width, context->resized_height);
    mipk_quad_sample_chroma(&source, corners,
        destination_uv +
            (size_t)(context->pad_y / 2U) * context->width + context->pad_x,
        context->width, context->resized_width / 2U,
        context->resized_height / 2U);
    return MIPK_OK;
}

/* BT.601 limited range, the convention of camera NV12 on these SoCs. */
static void yuv_to_rgb(uint8_t y, uint8_t u, uint8_t v, float rgb[3])
{
    const float luma = 1.164F * ((float)y - 16.0F);
    const float cb = (float)u - 128.0F;
    const float cr = (float)v - 128.0F;
    rgb[0] = luma + 1.596F * cr;
    rgb[1] = luma - 0.392F * cb - 0.813F * cr;
    rgb[2] = luma + 2.017F * cb;
    for (size_t index = 0; index < 3; ++index) {
        rgb[index] = fminf(fmaxf(rgb[index], 0.0F), 255.0F);
    }
}

static void store_pixel(const image_preprocess_context *context,
    void *destination, size_t row, size_t column, const float rgb[3])
{
    for (size_t channel = 0; channel < 3; ++channel) {
        /* rgb[] is semantic RGB; pick the value the output order wants. */
        const size_t source = context->output_order == CHANNEL_ORDER_RGB ?
            channel : 2U - channel;
        const float value = (rgb[source] - context->mean[channel]) *
            context->scale[channel];
        const size_t index = (channel * context->height + row) *
            context->model_row_stride + column;
        if (context->model_data_type == MIPK_DTYPE_FLOAT32) {
            ((float *)destination)[index] = value;
        } else {
            ((uint16_t *)destination)[index] = mipk_float_to_half(value);
        }
    }
}

/* Luma and chroma of an NV12/NV21 input, whether it came as a frame or bytes. */
typedef struct {
    const uint8_t *luma;
    const uint8_t *chroma;
    size_t luma_stride;
    size_t chroma_stride;
} yuv_source;

static bool yuv_source_from_input(const image_preprocess_context *context,
    const mipk_input *input, yuv_source *source)
{
    if (input->type == MIPK_INPUT_VIDEO_FRAME) {
        const mipk_video_frame *frame = &input->value.video;
        if (frame->plane_count != 2) return false;
        source->luma = frame->planes[0].buffer.virtual_address;
        source->chroma = frame->planes[1].buffer.virtual_address;
        source->luma_stride = frame->planes[0].stride;
        source->chroma_stride = frame->planes[1].stride;
    } else {
        const uint8_t *bytes = mipk_input_data(input);
        if (bytes == NULL ||
            mipk_input_data_size(input) != context->source_size) return false;
        source->luma = bytes;
        source->chroma = bytes +
            (size_t)context->source_width * context->source_height;
        source->luma_stride = context->source_width;
        source->chroma_stride = context->source_width;
    }
    return source->luma != NULL && source->chroma != NULL;
}

/*
 * Maps an output column or row back to a source coordinate for bilinear
 * sampling: pixel centres align, and the result never leaves the source.
 */
static float source_coordinate(uint32_t target, uint32_t target_extent,
    uint32_t source_extent)
{
    float value = ((float)target + 0.5F) * (float)source_extent /
        (float)target_extent - 0.5F;
    if (value < 0.0F) value = 0.0F;
    if (value > (float)source_extent - 1.0F) value = (float)source_extent - 1.0F;
    return value;
}

static float sample_bilinear(const uint8_t *plane, size_t stride,
    uint32_t width, uint32_t height, float x, float y, size_t step)
{
    const uint32_t x0 = (uint32_t)x;
    const uint32_t y0 = (uint32_t)y;
    const uint32_t x1 = x0 + 1U < width ? x0 + 1U : x0;
    const uint32_t y1 = y0 + 1U < height ? y0 + 1U : y0;
    const float fx = x - (float)x0;
    const float fy = y - (float)y0;
    const float top = (float)plane[(size_t)y0 * stride + x0 * step] * (1.0F - fx) +
        (float)plane[(size_t)y0 * stride + x1 * step] * fx;
    const float bottom = (float)plane[(size_t)y1 * stride + x0 * step] * (1.0F - fx) +
        (float)plane[(size_t)y1 * stride + x1 * step] * fx;
    return top * (1.0F - fy) + bottom * fy;
}

/* Bilinear letterbox straight from the NV12 planes into planar float. */
static mipk_status bilinear_from_yuv(const image_preprocess_context *context,
    mipk_operator_io *execution, void *destination)
{
    yuv_source source;
    if (!yuv_source_from_input(context, execution->input, &source) ||
        context->resized_width == 0 || context->resized_height == 0 ||
        context->source_width < 2 || context->source_height < 2) {
        return MIPK_ERR_ARGUMENT;
    }
    const bool nv21 = context->input_format == MIPK_FORMAT_NV21;
    const uint32_t chroma_width = context->source_width / 2U;
    const uint32_t chroma_height = context->source_height / 2U;
    const float pad_luma = context->letterbox_value;
    float pad_rgb[3];
    yuv_to_rgb((uint8_t)pad_luma, 128, 128, pad_rgb);
    for (uint32_t row = 0; row < context->height; ++row) {
        const bool inside_rows = row >= context->pad_y &&
            row < context->pad_y + context->resized_height;
        const float sy = inside_rows ? source_coordinate(row - context->pad_y,
            context->resized_height, context->source_height) : 0.0F;
        for (uint32_t column = 0; column < context->width; ++column) {
            const bool inside = inside_rows && column >= context->pad_x &&
                column < context->pad_x + context->resized_width;
            if (!inside) {
                store_pixel(context, destination, row, column, pad_rgb);
                continue;
            }
            const float sx = source_coordinate(column - context->pad_x,
                context->resized_width, context->source_width);
            const float luma = sample_bilinear(source.luma, source.luma_stride,
                context->source_width, context->source_height, sx, sy, 1);
            const float cx = sx * 0.5F;
            const float cy = sy * 0.5F;
            const float u = sample_bilinear(source.chroma + (nv21 ? 1 : 0),
                source.chroma_stride, chroma_width, chroma_height, cx, cy, 2);
            const float v = sample_bilinear(source.chroma + (nv21 ? 0 : 1),
                source.chroma_stride, chroma_width, chroma_height, cx, cy, 2);
            float rgb[3];
            yuv_to_rgb((uint8_t)(luma + 0.5F), (uint8_t)(u + 0.5F),
                (uint8_t)(v + 0.5F), rgb);
            store_pixel(context, destination, row, column, rgb);
        }
    }
    return MIPK_OK;
}

/* Stage the frame as target-sized NV12, then convert plane by plane. */
static mipk_status float_from_yuv(const image_preprocess_context *context,
    mipk_operator_io *execution, void *destination)
{
    mipk_status status;
    const uint8_t *source = mipk_input_data(execution->input);
    if (context->scratch == NULL) return MIPK_ERR_RUNTIME;
    if (execution->input->type == MIPK_INPUT_VIDEO_FRAME) {
        status = execution->image_view != NULL ?
            resize_yuv_view(context, &execution->input->value.video,
                execution->image_view, context->scratch) :
            resize_yuv(context, &execution->input->value.video,
                context->scratch);
    } else if (source != NULL &&
        mipk_input_data_size(execution->input) == context->source_size) {
        const size_t source_pixels =
            (size_t)context->source_width * context->source_height;
        const mipk_video_frame packed = {
            .format = (uint32_t)context->input_format,
            .width = context->source_width,
            .height = context->source_height,
            .plane_count = 2,
            .planes = {
                {
                    .buffer = {
                        .virtual_address = (void *)source,
                        .size = source_pixels,
                    },
                    .stride = context->source_width,
                },
                {
                    .buffer = {
                        .virtual_address = (void *)(source + source_pixels),
                        .size = source_pixels / 2U,
                    },
                    .stride = context->source_width,
                },
            },
        };
        status = resize_yuv(context, &packed, context->scratch);
    } else {
        return MIPK_ERR_ARGUMENT;
    }
    if (status != MIPK_OK) return status;
    const uint8_t *luma = context->scratch;
    const uint8_t *chroma = context->scratch +
        (size_t)context->width * context->height;
    const bool nv21 = context->input_format == MIPK_FORMAT_NV21;
    for (size_t row = 0; row < context->height; ++row) {
        const uint8_t *chroma_row = chroma + (row / 2U) * context->width;
        for (size_t column = 0; column < context->width; ++column) {
            const uint8_t *pair = chroma_row + (column & ~(size_t)1U);
            float rgb[3];
            yuv_to_rgb(luma[row * context->width + column],
                nv21 ? pair[1] : pair[0], nv21 ? pair[0] : pair[1], rgb);
            store_pixel(context, destination, row, column, rgb);
        }
    }
    return MIPK_OK;
}

/* Nearest-neighbour letterbox of packed RGB or BGR bytes. */
static mipk_status float_from_packed(const image_preprocess_context *context,
    mipk_operator_io *execution, void *destination)
{
    const uint8_t *source = mipk_input_data(execution->input);
    if (source == NULL || execution->image_view != NULL ||
        mipk_input_data_size(execution->input) != context->source_size ||
        context->resized_width == 0 || context->resized_height == 0) {
        return MIPK_ERR_ARGUMENT;
    }
    const float fill[3] = {context->letterbox_value, context->letterbox_value,
        context->letterbox_value};
    for (size_t row = 0; row < context->height; ++row) {
        const bool inside_rows = row >= context->pad_y &&
            row < (size_t)context->pad_y + context->resized_height;
        const size_t source_row = inside_rows ?
            (row - context->pad_y) * context->source_height /
                context->resized_height : 0;
        for (size_t column = 0; column < context->width; ++column) {
            const bool inside = inside_rows && column >= context->pad_x &&
                column < (size_t)context->pad_x + context->resized_width;
            if (!inside) {
                store_pixel(context, destination, row, column, fill);
                continue;
            }
            float rgb[3];
            if (context->bilinear) {
                const float sx = source_coordinate(
                    (uint32_t)(column - context->pad_x),
                    context->resized_width, context->source_width);
                const float sy = source_coordinate(
                    (uint32_t)(row - context->pad_y),
                    context->resized_height, context->source_height);
                for (size_t channel = 0; channel < 3; ++channel) {
                    rgb[channel] = sample_bilinear(source + channel,
                        (size_t)context->source_width * 3U,
                        context->source_width, context->source_height, sx, sy,
                        3);
                }
            } else {
                const size_t source_column = (column - context->pad_x) *
                    context->source_width / context->resized_width;
                const uint8_t *pixel = source +
                    (source_row * context->source_width + source_column) * 3U;
                rgb[0] = pixel[0];
                rgb[1] = pixel[1];
                rgb[2] = pixel[2];
            }
            if (context->input_order == CHANNEL_ORDER_BGR) {
                const float red = rgb[2];
                rgb[2] = rgb[0];
                rgb[0] = red;
            }
            store_pixel(context, destination, row, column, rgb);
        }
    }
    return MIPK_OK;
}

static mipk_status process_float(const image_preprocess_context *context,
    mipk_operator_io *execution, void *destination)
{
    /*
     * Rows are written only as far as the image is wide, so any stride
     * padding beyond it has to start clear.
     */
    if (mipk_memory_zero(destination,
            execution->model_inputs[0].byte_size) != 0) {
        return MIPK_ERR_RUNTIME;
    }
    if (context->yuv_source) {
        /* A rotated view goes through the quad sampler; the rest is direct. */
        if (context->bilinear && execution->image_view == NULL) {
            return bilinear_from_yuv(context, execution, destination);
        }
        return float_from_yuv(context, execution, destination);
    }
    return float_from_packed(context, execution, destination);
}

static mipk_status image_preprocess_process(void *private_data,
    mipk_operator_io *execution)
{
    const image_preprocess_context *context = private_data;
    if (context == NULL || execution == NULL || execution->input == NULL ||
        (execution->input->type != MIPK_INPUT_BYTES &&
            execution->input->type != MIPK_INPUT_VIDEO_FRAME) ||
        execution->model_inputs == NULL || execution->model_input_count == 0 ||
        mipk_mutable_tensor_data(&execution->model_inputs[0]) == NULL) {
        return MIPK_ERR_ARGUMENT;
    }
    void *destination = mipk_mutable_tensor_data(&execution->model_inputs[0]);
    /*
     * A platform with a hardware scaler gets first refusal.  It sees the whole
     * request and may decline (a rotated view, a float output it cannot
     * produce), in which case the software path below does the work.
     */
    if (context->image_backend != NULL &&
        context->image_backend->supports != NULL &&
        context->image_backend->process != NULL) {
        const mipk_image_request request = {
            .input = execution->input,
            .view = execution->image_view,
            .source_format = context->input_format,
            .source_width = context->source_width,
            .source_height = context->source_height,
            .target_width = context->width,
            .target_height = context->height,
            .resized_width = context->resized_width,
            .resized_height = context->resized_height,
            .pad_x = context->pad_x,
            .pad_y = context->pad_y,
            .pad_value = (uint8_t)context->letterbox_value,
            .output = context->direct ? MIPK_IMAGE_OUTPUT_NV12 :
                MIPK_IMAGE_OUTPUT_FLOAT_NCHW,
            .output_rgb = context->output_order == CHANNEL_ORDER_RGB,
            .mean = {context->mean[0], context->mean[1], context->mean[2]},
            .scale = {context->scale[0], context->scale[1], context->scale[2]},
        };
        if (context->image_backend->supports(context->image_backend->context,
                &request)) {
            return context->image_backend->process(
                context->image_backend->context, &request,
                &execution->model_inputs[0]);
        }
    }
    if (context->direct) {
        if (execution->model_inputs[0].byte_size < context->target_size) {
            return MIPK_ERR_MODEL;
        }
        if (execution->input->type == MIPK_INPUT_VIDEO_FRAME &&
            (context->input_format == MIPK_FORMAT_NV12 ||
                context->input_format == MIPK_FORMAT_NV21)) {
            /*
             * Both of these fill the whole model input themselves, padding
             * included, so clearing it first would write the pipeline's
             * largest buffer twice for no gain.
             */
            if (execution->image_view != NULL) {
                return resize_yuv_view(context,
                    &execution->input->value.video, execution->image_view,
                    destination);
            }
            return resize_yuv(context, &execution->input->value.video,
                destination);
        }
        if (mipk_memory_zero(destination,
                execution->model_inputs[0].byte_size) != 0) {
            return MIPK_ERR_RUNTIME;
        }
        const uint8_t *source = mipk_input_data(execution->input);
        if (source == NULL ||
            mipk_input_data_size(execution->input) != context->source_size) {
            return MIPK_ERR_ARGUMENT;
        }
        if (context->source_size != context->target_size ||
            context->source_width != context->width ||
            context->source_height != context->height) {
            const size_t source_pixels =
                (size_t)context->source_width * context->source_height;
            const mipk_video_frame packed = {
                .format = (uint32_t)context->input_format,
                .width = context->source_width,
                .height = context->source_height,
                .plane_count = 2,
                .planes = {
                    {
                        .buffer = {
                            .virtual_address = (void *)source,
                            .size = source_pixels,
                        },
                        .stride = context->source_width,
                    },
                    {
                        .buffer = {
                            .virtual_address = (void *)(source + source_pixels),
                            .size = source_pixels / 2U,
                        },
                        .stride = context->source_width,
                    },
                },
            };
            return resize_yuv(context, &packed, destination);
        }
        return mipk_memory_copy(destination,
            execution->model_inputs[0].byte_size, source,
            context->target_size) == 0 ? MIPK_OK : MIPK_ERR_RUNTIME;
    }
    return process_float(context, execution, destination);
}

static const mipk_capability_descriptor g_capability =
    MIPK_OPERATOR_CAPABILITY("image_preprocess", "2.0.0");

static const char *const g_options[] = {
    "mode",
    "format",
    "layout",
    "input_order",
    "output_order",
    "mean",
    "scale",
    "letterbox_value",
    "interpolation",
    "target_width",
    "target_height",
};

static const mipk_port_schema g_inputs[] = {
    {
        .port = "input",
        .accepted_value_mask = MIPK_VALUE_MASK(MIPK_VALUE_INPUT) |
            MIPK_VALUE_MASK(MIPK_VALUE_IMAGE_VIEW),
        .accepted_domain_mask = MIPK_DOMAIN_MASK(MIPK_DOMAIN_IMAGE),
        .required = true,
    },
};

static const mipk_node_schema g_schema = {
    .inputs = g_inputs,
    .input_count = sizeof(g_inputs) / sizeof(g_inputs[0]),
    .option_names = g_options,
    .option_count = sizeof(g_options) / sizeof(g_options[0]),
    .output_type = MIPK_VALUE_TENSORS,
    .output_domain = MIPK_DOMAIN_IMAGE,
};

const mipk_operator_ops *mipk_image_preprocess_operator(void)
{
    static const mipk_operator_ops operations = {
        .capability = &g_capability,
        .schema = &g_schema,
        .name = "image_preprocess",
        .create = image_preprocess_create,
        .destroy = image_preprocess_destroy,
        .process = image_preprocess_process,
        .validate = image_preprocess_validate,
    };
    return &operations;
}

mipk_status mipk_image_preprocess_register(void)
{
    return mipk_operator_register(mipk_image_preprocess_operator());
}
