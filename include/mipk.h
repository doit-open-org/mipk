#ifndef MIPK_H
#define MIPK_H

#include "mipk_runtime.h"
#include "mipk_platform.h"
#include "mipk_task.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MIPK_FORMAT_VERSION 2U
#define MIPK_SCHEMA_REVISION 1U
#define MIPK_MAX_OUTPUT_HANDLES 8U

typedef enum {
    MIPK_FORMAT_NV12 = 0,
    MIPK_FORMAT_NV21,
    MIPK_FORMAT_BGR,
    MIPK_FORMAT_RGB,
    MIPK_FORMAT_PCM_S16LE,
} mipk_data_format;

typedef struct {
    const char *name;
    const char *version;
    const char *task;
    mipk_data_format input_format;
    uint32_t input_width;
    uint32_t input_height;
    size_t input_size;
    uint32_t input_sample_rate;
    uint32_t input_channels;
    uint32_t input_frame_samples;
} mipk_pipeline_info;

typedef enum { MIPK_ATTACHMENT_TEXT = 0 } mipk_attachment_type;

typedef struct mipk_pipeline mipk_pipeline;
typedef struct mipk_output mipk_output;
typedef struct {
    uint32_t struct_size;
    uint32_t max_output_handles;
    /* Optional. NULL lets each backend allocate ordinary CPU memory. */
    const mipk_platform_context *platform;
    /*
     * Optional. When set, every failure reason from open() and from later
     * calls on this pipeline is written here (newline separated, NUL
     * terminated, truncated to fit) instead of stderr.  Each call clears it
     * first, so after a failed call it describes that call.  The buffer must
     * outlive the pipeline.
     */
    char *diagnostic;
    size_t diagnostic_size;
} mipk_pipeline_open_options;

#define MIPK_RUNTIME_VERSION "2.0.0"

typedef struct {
    uint32_t struct_size;
    size_t model_source_bytes;
    size_t model_io_bytes;
    size_t persistent_cpu_bytes;
    size_t scratch_cpu_bytes;
    size_t collection_bytes;
    size_t output_pool_bytes;
    size_t peak_cpu_bytes;
    size_t peak_mmz_bytes;
    size_t fusion_group_count;
} mipk_memory_plan;

typedef struct {
    uint32_t struct_size;
    const char *node_id;
    const char *node_type;
    const char *node_name;
    const char *option_name;
    mipk_runtime_option_type option_type;
    const char *unit;
    const char *description;
} mipk_runtime_option_info;

mipk_status mipk_pipeline_open(const char *package_path,
    const mipk_pipeline_open_options *options, mipk_pipeline **pipeline);
mipk_status mipk_pipeline_close(mipk_pipeline *pipeline);

const mipk_pipeline_info *mipk_pipeline_get_info(
    const mipk_pipeline *pipeline);
mipk_status mipk_pipeline_get_memory_plan(const mipk_pipeline *pipeline,
    mipk_memory_plan *plan);
size_t mipk_pipeline_runtime_option_count(const mipk_pipeline *pipeline);
mipk_status mipk_pipeline_runtime_option_at(const mipk_pipeline *pipeline,
    size_t index, mipk_runtime_option_info *info);
mipk_status mipk_pipeline_runtime_option_value_at(
    const mipk_pipeline *pipeline, size_t index, char *buffer,
    size_t capacity, size_t *required_size);
const char *mipk_runtime_option_type_name(mipk_runtime_option_type type);
mipk_status mipk_pipeline_label_set_id(const mipk_pipeline *pipeline,
    const char *name, uint32_t *label_set_id);
const char *mipk_pipeline_label_set_name(const mipk_pipeline *pipeline,
    uint32_t label_set_id);
const char *mipk_pipeline_label_name(const mipk_pipeline *pipeline,
    uint32_t label_set_id, int32_t class_id);

mipk_status mipk_pipeline_output_create(mipk_pipeline *pipeline,
    mipk_output **output);
void mipk_output_destroy(mipk_output *output);
mipk_status mipk_pipeline_run_output(mipk_pipeline *pipeline,
    const mipk_input *input, mipk_output *output);
mipk_status mipk_pipeline_reset(mipk_pipeline *pipeline);

/* Runtime option updates are transactional and never replace model assets. */
mipk_status mipk_pipeline_update_node_options(mipk_pipeline *pipeline,
    const char *node_id, const char *options_json);
/*
 * Same, but refuses unless the node is the named operator or postprocess.
 * Modules build their typed update helpers on it (see mipk_nms.h).
 */
mipk_status mipk_pipeline_update_stage_options(mipk_pipeline *pipeline,
    const char *node_id, mipk_stage_type stage_type, const char *stage_name,
    const char *options_json);
mipk_status mipk_pipeline_apply_overrides_json(mipk_pipeline *pipeline,
    const char *overrides_json);
mipk_status mipk_pipeline_load_overrides(mipk_pipeline *pipeline,
    const char *path);

uint64_t mipk_output_pts_us(const mipk_output *output);
size_t mipk_output_result_count(const mipk_output *output);
const mipk_result *mipk_output_result_at(const mipk_output *output,
    size_t index);
size_t mipk_output_attachment_count(const mipk_output *output);
mipk_status mipk_output_attachment_at(const mipk_output *output,
    size_t index, mipk_attachment_type *type, const void **data,
    size_t *size);

const char *mipk_data_format_string(mipk_data_format format);

/*
 * Inputs for the common case: ordinary CPU memory the application owns and
 * keeps valid for the duration of the run.  Zero-copy from device memory
 * fills mipk_input by hand instead.
 */
static inline mipk_buffer_desc mipk_cpu_buffer_desc(const void *data,
    size_t size)
{
    /* A named initializer rather than a compound literal, which C++ lacks. */
    const mipk_buffer_desc desc = {
        .virtual_address = (void *)data,
        .physical_address = MIPK_PHYSICAL_ADDRESS_INVALID,
        .size = size,
        .alignment = 1,
        .memory_type = MIPK_MEMORY_CPU,
        .owner = MIPK_BUFFER_OWNER_APPLICATION,
    };
    return desc;
}

/* Packed bytes in the manifest's input format: RGB/BGR pixels or raw NV12. */
static inline mipk_input mipk_input_bytes(const void *data, size_t size,
    uint64_t pts_us)
{
    mipk_input input = {.type = MIPK_INPUT_BYTES, .pts_us = pts_us};
    input.value.bytes.buffer = mipk_cpu_buffer_desc(data, size);
    input.value.bytes.initial_sync_state = MIPK_BUFFER_SYNCED;
    return input;
}

/* Packed RGB or BGR, three bytes per pixel, rows without padding. */
static inline mipk_input mipk_input_packed_image(const void *pixels,
    uint32_t width, uint32_t height, uint64_t pts_us)
{
    return mipk_input_bytes(pixels, (size_t)width * height * 3U, pts_us);
}

/* NV12 or NV21 as two planes; strides in bytes, 0 for the frame width. */
static inline mipk_input mipk_input_nv12(mipk_data_format format,
    const void *luma, uint32_t luma_stride, const void *chroma,
    uint32_t chroma_stride, uint32_t width, uint32_t height, uint64_t pts_us)
{
    if (luma_stride == 0) luma_stride = width;
    if (chroma_stride == 0) chroma_stride = width;
    mipk_input input = {.type = MIPK_INPUT_VIDEO_FRAME, .pts_us = pts_us};
    input.value.video.format = (uint32_t)format;
    input.value.video.width = width;
    input.value.video.height = height;
    input.value.video.plane_count = 2;
    input.value.video.planes[0].buffer = mipk_cpu_buffer_desc(luma,
        (size_t)luma_stride * height);
    input.value.video.planes[0].initial_sync_state = MIPK_BUFFER_SYNCED;
    input.value.video.planes[0].stride = luma_stride;
    input.value.video.planes[1].buffer = mipk_cpu_buffer_desc(chroma,
        (size_t)chroma_stride * (height / 2U));
    input.value.video.planes[1].initial_sync_state = MIPK_BUFFER_SYNCED;
    input.value.video.planes[1].stride = chroma_stride;
    return input;
}

/* One frame of mono 16-bit PCM; pts_us must advance with real time. */
static inline mipk_input mipk_input_pcm_s16(const int16_t *samples,
    uint32_t sample_count, uint32_t sample_rate, uint64_t pts_us)
{
    mipk_input input = {.type = MIPK_INPUT_AUDIO, .pts_us = pts_us};
    input.value.audio.buffer = mipk_cpu_buffer_desc(samples,
        (size_t)sample_count * sizeof(*samples));
    input.value.audio.initial_sync_state = MIPK_BUFFER_SYNCED;
    input.value.audio.sample_rate = sample_rate;
    input.value.audio.channels = 1;
    input.value.audio.sample_count = sample_count;
    return input;
}

#ifdef __cplusplus
}
#endif
#endif
