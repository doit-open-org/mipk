#define _XOPEN_SOURCE 700

#include "mipk_pipeline_private.h"

#include "mipk_config.h"
#include "cJSON.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define MIPK_MAX_OVERRIDES_BYTES (64U * 1024U)

static void pipeline_free(mipk_pipeline *pipeline)
{
    if (pipeline == NULL) return;
    mipk_pipeline_runtime_destroy(pipeline->runtime);
    mipk_manifest_free(pipeline);
    mipk_container_free_sources(pipeline->model_sources);
    free(pipeline->output_results);
    free(pipeline->outputs);
    free(pipeline->package_path);
    if (pipeline->platform_retained) {
        pipeline->platform->release(pipeline->platform->context);
    }
    free(pipeline);
}


static bool platform_context_valid(const mipk_platform_context *platform)
{
    return platform == NULL ||
        (platform->struct_size == sizeof(*platform) &&
         platform->context != NULL && platform->retain != NULL &&
         platform->release != NULL && platform->buffer_alloc != NULL &&
         platform->buffer_free != NULL && platform->sync_for_device != NULL &&
         platform->sync_for_cpu != NULL &&
         (platform->shared_arbiter == NULL ||
            (platform->shared_arbiter->struct_size ==
                sizeof(*platform->shared_arbiter) &&
             platform->shared_arbiter->context != NULL &&
             platform->shared_arbiter->lock != NULL &&
             platform->shared_arbiter->unlock != NULL)));
}

mipk_status mipk_pipeline_open(const char *package_path,
    const mipk_pipeline_open_options *options, mipk_pipeline **output)
{
    if (package_path == NULL || package_path[0] == '\0' || options == NULL ||
        options->struct_size != sizeof(*options) ||
        options->max_output_handles == 0 ||
        options->max_output_handles > MIPK_MAX_OUTPUT_HANDLES ||
        !platform_context_valid(options->platform) || output == NULL) {
        return MIPK_ERR_ARGUMENT;
    }
    *output = NULL;
    mipk_diag_begin(options->diagnostic, options->diagnostic_size);
    mipk_status status = mipk_init();
    if (status != MIPK_OK) {
        mipk_diag("module registration failed: %s", mipk_status_string(status));
        mipk_diag_end();
        return status;
    }
    mipk_pipeline *pipeline = calloc(1, sizeof(*pipeline));
    if (pipeline == NULL) {
        mipk_diag_end();
        return MIPK_ERR_MEMORY;
    }
    pipeline->diagnostic = options->diagnostic;
    pipeline->diagnostic_size = options->diagnostic_size;
    pipeline->platform = options->platform;
    pipeline->config.platform = options->platform;
    if (pipeline->platform != NULL) {
        status = pipeline->platform->retain(pipeline->platform->context);
        if (status != MIPK_OK) goto failed;
        pipeline->platform_retained = true;
    }
    pipeline->max_output_handles = options->max_output_handles;
    pipeline->package_path = realpath(package_path, NULL);
    struct stat attributes;
    status = MIPK_OK;
    if (pipeline->package_path == NULL ||
        stat(pipeline->package_path, &attributes) != 0 ||
        (!S_ISREG(attributes.st_mode) && !S_ISDIR(attributes.st_mode))) {
        mipk_diag("package '%s' is not a readable file or directory",
            package_path);
        status = MIPK_ERR_IO;
        goto failed;
    }
    char *manifest = NULL;
    if (S_ISDIR(attributes.st_mode)) {
        status = mipk_container_open_directory(pipeline->package_path,
            &manifest);
    } else {
        status = mipk_container_open(pipeline->package_path, &manifest,
            &pipeline->model_sources, &pipeline->model_source_count);
    }
    if (status != MIPK_OK) {
        mipk_diag("cannot read package '%s': %s", package_path,
            mipk_status_string(status));
    }
    if (status == MIPK_OK) status = mipk_manifest_parse(manifest, pipeline);
    free(manifest);
    if (status != MIPK_OK) goto failed;
    status = mipk_pipeline_runtime_create(&pipeline->config,
        &pipeline->runtime);
    if (status != MIPK_OK) goto failed;
    if (pipeline->config.result_capacity >
        SIZE_MAX / pipeline->max_output_handles ||
        pipeline->config.result_capacity * pipeline->max_output_handles >
            SIZE_MAX / sizeof(*pipeline->output_results) ||
        pipeline->config.result_capacity * pipeline->max_output_handles *
            sizeof(*pipeline->output_results) > MIPK_MAX_OUTPUT_BYTES) {
        status = MIPK_ERR_CAPACITY;
        goto failed;
    }
    pipeline->outputs = calloc(pipeline->max_output_handles,
        sizeof(*pipeline->outputs));
    pipeline->output_results = calloc(
        pipeline->config.result_capacity * pipeline->max_output_handles,
        sizeof(*pipeline->output_results));
    if (pipeline->outputs == NULL || pipeline->output_results == NULL) {
        status = MIPK_ERR_MEMORY;
        goto failed;
    }
    for (uint32_t index = 0; index < pipeline->max_output_handles; ++index) {
        pipeline->outputs[index].owner = pipeline;
        pipeline->outputs[index].pts_us = MIPK_PTS_UNKNOWN;
        pipeline->outputs[index].results.items = pipeline->output_results +
            (size_t)index * pipeline->config.result_capacity;
        pipeline->outputs[index].results.capacity =
            pipeline->config.result_capacity;
    }
    *output = pipeline;
    mipk_diag_end();
    return MIPK_OK;

failed:
    pipeline_free(pipeline);
    mipk_diag_end();
    return status;
}

/* Every later call reports into the buffer open() was given. */
static void arm_diagnostics(const mipk_pipeline *pipeline)
{
    mipk_diag_begin(pipeline->diagnostic, pipeline->diagnostic_size);
}

mipk_status mipk_pipeline_close(mipk_pipeline *pipeline)
{
    if (pipeline == NULL) return MIPK_ERR_ARGUMENT;
    if (pipeline->active_output_handles != 0 ||
        mipk_pipeline_runtime_busy(pipeline->runtime)) return MIPK_ERR_BUSY;
    pipeline_free(pipeline);
    return MIPK_OK;
}

const mipk_pipeline_info *mipk_pipeline_get_info(
    const mipk_pipeline *pipeline)
{
    return pipeline == NULL ? NULL : &pipeline->info;
}

mipk_status mipk_pipeline_get_memory_plan(const mipk_pipeline *pipeline,
    mipk_memory_plan *plan)
{
    if (pipeline == NULL || plan == NULL ||
        plan->struct_size != sizeof(*plan)) return MIPK_ERR_ARGUMENT;
    mipk_status status = mipk_pipeline_runtime_get_memory_plan(
        pipeline->runtime, plan);
    if (status != MIPK_OK) return status;
    if (pipeline->config.result_capacity > SIZE_MAX /
            pipeline->max_output_handles ||
        pipeline->config.result_capacity * pipeline->max_output_handles >
            SIZE_MAX / sizeof(*pipeline->output_results)) {
        return MIPK_ERR_CAPACITY;
    }
    plan->output_pool_bytes = pipeline->config.result_capacity *
        pipeline->max_output_handles * sizeof(*pipeline->output_results);
    if (plan->output_pool_bytes > SIZE_MAX - plan->peak_cpu_bytes) {
        return MIPK_ERR_CAPACITY;
    }
    plan->peak_cpu_bytes += plan->output_pool_bytes;
    return MIPK_OK;
}

size_t mipk_pipeline_runtime_option_count(const mipk_pipeline *pipeline)
{
    return pipeline == NULL ? 0 :
        mipk_pipeline_runtime_get_option_count(pipeline->runtime);
}

mipk_status mipk_pipeline_runtime_option_at(const mipk_pipeline *pipeline,
    size_t index, mipk_runtime_option_info *info)
{
    if (pipeline == NULL) return MIPK_ERR_ARGUMENT;
    return mipk_pipeline_runtime_get_option_at(pipeline->runtime, index,
        info);
}

mipk_status mipk_pipeline_runtime_option_value_at(
    const mipk_pipeline *pipeline, size_t index, char *buffer,
    size_t capacity, size_t *required_size)
{
    if (pipeline == NULL) return MIPK_ERR_ARGUMENT;
    return mipk_pipeline_runtime_get_option_value_at(pipeline->runtime,
        index, buffer, capacity, required_size);
}

const char *mipk_runtime_option_type_name(mipk_runtime_option_type type)
{
    switch (type) {
        case MIPK_RUNTIME_OPTION_BOOL: return "bool";
        case MIPK_RUNTIME_OPTION_UINT32: return "uint32";
        case MIPK_RUNTIME_OPTION_SIZE: return "size";
        case MIPK_RUNTIME_OPTION_FLOAT: return "float";
        case MIPK_RUNTIME_OPTION_STRING: return "string";
        case MIPK_RUNTIME_OPTION_STRING_ARRAY: return "string_array";
        case MIPK_RUNTIME_OPTION_JSON: return "json";
        default: return "unknown";
    }
}

mipk_status mipk_pipeline_label_set_id(const mipk_pipeline *pipeline,
    const char *name, uint32_t *label_set_id)
{
    if (pipeline == NULL || name == NULL || label_set_id == NULL) {
        return MIPK_ERR_ARGUMENT;
    }
    return mipk_manifest_find_label_set(pipeline, name, label_set_id) ? MIPK_OK :
        MIPK_ERR_ARGUMENT;
}

const char *mipk_pipeline_label_set_name(const mipk_pipeline *pipeline,
    uint32_t label_set_id)
{
    return pipeline == NULL || label_set_id >= pipeline->label_set_count ? NULL :
        pipeline->label_sets[label_set_id].name;
}

const char *mipk_pipeline_label_name(const mipk_pipeline *pipeline,
    uint32_t label_set_id, int32_t class_id)
{
    if (pipeline == NULL || label_set_id >= pipeline->label_set_count ||
        class_id < 0 || (size_t)class_id >=
            pipeline->label_sets[label_set_id].class_count) return NULL;
    return pipeline->label_sets[label_set_id].classes[class_id].name;
}

mipk_status mipk_pipeline_output_create(mipk_pipeline *pipeline,
    mipk_output **output)
{
    if (pipeline == NULL || output == NULL) return MIPK_ERR_ARGUMENT;
    *output = NULL;
    for (uint32_t index = 0; index < pipeline->max_output_handles; ++index) {
        if (pipeline->outputs[index].leased) continue;
        pipeline->outputs[index].leased = true;
        pipeline->outputs[index].results.count = 0;
        pipeline->outputs[index].pts_us = MIPK_PTS_UNKNOWN;
        ++pipeline->active_output_handles;
        *output = &pipeline->outputs[index];
        return MIPK_OK;
    }
    return MIPK_ERR_CAPACITY;
}

void mipk_output_destroy(mipk_output *output)
{
    if (output == NULL || !output->leased || output->owner == NULL) return;
    output->results.count = 0;
    output->pts_us = MIPK_PTS_UNKNOWN;
    output->leased = false;
    --output->owner->active_output_handles;
}

static mipk_status validate_input(const mipk_pipeline *pipeline,
    const mipk_input *input)
{
    if (pipeline == NULL || input == NULL) return MIPK_ERR_ARGUMENT;
    const bool audio_pipeline =
        pipeline->audio;
    if (input->type < MIPK_INPUT_BYTES || input->type > MIPK_INPUT_AUDIO ||
        (audio_pipeline && input->type != MIPK_INPUT_AUDIO) ||
        (!audio_pipeline && input->type == MIPK_INPUT_AUDIO)) {
        return MIPK_ERR_ARGUMENT;
    }

    if (input->type == MIPK_INPUT_BYTES) {
        const mipk_buffer_desc *buffer = &input->value.bytes.buffer;
        if (buffer->owner != MIPK_BUFFER_OWNER_APPLICATION ||
            buffer->memory_type != MIPK_MEMORY_CPU ||
            buffer->virtual_address == NULL ||
            buffer->physical_address != MIPK_PHYSICAL_ADDRESS_INVALID ||
            buffer->size != pipeline->info.input_size) {
            return MIPK_ERR_ARGUMENT;
        }
        return MIPK_OK;
    }

    if (input->type == MIPK_INPUT_AUDIO) {
        const mipk_audio_frame *audio = &input->value.audio;
        const mipk_buffer_desc *buffer = &audio->buffer;
        if (input->pts_us == MIPK_PTS_UNKNOWN ||
            audio->sample_rate != pipeline->info.input_sample_rate ||
            audio->channels != pipeline->info.input_channels ||
            audio->sample_count != pipeline->info.input_frame_samples ||
            buffer->owner != MIPK_BUFFER_OWNER_APPLICATION ||
            buffer->memory_type != MIPK_MEMORY_CPU ||
            buffer->virtual_address == NULL ||
            buffer->physical_address != MIPK_PHYSICAL_ADDRESS_INVALID ||
            buffer->size != pipeline->info.input_size) {
            return MIPK_ERR_ARGUMENT;
        }
        return MIPK_OK;
    }

    const mipk_video_frame *video = &input->value.video;
    if (video->format != (uint32_t)pipeline->info.input_format ||
        video->width != pipeline->info.input_width ||
        video->height != pipeline->info.input_height) {
        return MIPK_ERR_ARGUMENT;
    }
    const bool yuv = video->format == MIPK_FORMAT_NV12 ||
        video->format == MIPK_FORMAT_NV21;
    const uint32_t expected_planes = yuv ? 2U : 1U;
    if (video->plane_count != expected_planes) return MIPK_ERR_ARGUMENT;
    for (uint32_t index = 0; index < expected_planes; ++index) {
        const mipk_video_plane *plane = &video->planes[index];
        const mipk_buffer_desc *buffer = &plane->buffer;
        const size_t row_bytes = yuv ? video->width :
            (size_t)video->width * 3U;
        const size_t rows = yuv && index == 1 ? video->height / 2U :
            video->height;
        if (plane->stride < row_bytes ||
            plane->stride > SIZE_MAX / rows ||
            buffer->size < (size_t)plane->stride * rows ||
            buffer->owner != MIPK_BUFFER_OWNER_APPLICATION ||
            buffer->memory_type < MIPK_MEMORY_CPU ||
            buffer->memory_type > MIPK_MEMORY_PHYS_UNCACHED ||
            (buffer->virtual_address == NULL &&
                buffer->physical_address == MIPK_PHYSICAL_ADDRESS_INVALID) ||
            (buffer->memory_type == MIPK_MEMORY_CPU &&
                (buffer->virtual_address == NULL || buffer->physical_address !=
                    MIPK_PHYSICAL_ADDRESS_INVALID)) ||
            (buffer->memory_type != MIPK_MEMORY_CPU &&
                buffer->physical_address == MIPK_PHYSICAL_ADDRESS_INVALID) ||
            (buffer->memory_type == MIPK_MEMORY_PHYS_CACHED &&
                plane->initial_sync_state == MIPK_BUFFER_SYNC_UNKNOWN)) {
            return MIPK_ERR_ARGUMENT;
        }
    }
    return MIPK_OK;
}

mipk_status mipk_pipeline_run_output(mipk_pipeline *pipeline,
    const mipk_input *input, mipk_output *output)
{
    if (pipeline == NULL || output == NULL || output->owner != pipeline ||
        !output->leased) return MIPK_ERR_ARGUMENT;
    arm_diagnostics(pipeline);
    output->results.count = 0;
    output->pts_us = MIPK_PTS_UNKNOWN;
    mipk_status status = validate_input(pipeline, input);
    if (status != MIPK_OK) {
        mipk_diag("input rejected: %s", mipk_status_string(status));
    }
    if (status == MIPK_OK) {
        if (pipeline->audio &&
            pipeline->has_last_audio_pts &&
            (input->pts_us <= pipeline->last_audio_pts_us ||
                input->pts_us - pipeline->last_audio_pts_us >
                    pipeline->audio_reset_gap_us)) {
            status = mipk_pipeline_runtime_reset(pipeline->runtime);
        }
    }
    if (status == MIPK_OK) {
        status = mipk_pipeline_runtime_run(pipeline->runtime, input,
            &output->results);
    }
    if (status == MIPK_OK) {
        for (size_t index = 0; index < output->results.count; ++index) {
            output->results.items[index].result_id = index + 1U;
        }
    } else {
        output->results.count = 0;
    }
    if (status == MIPK_OK || status == MIPK_AGAIN) {
        output->pts_us = input->pts_us;
        if (pipeline->audio) {
            pipeline->last_audio_pts_us = input->pts_us;
            pipeline->has_last_audio_pts = true;
        }
    }
    if (status != MIPK_OK && status != MIPK_AGAIN) {
        mipk_diag("run failed: %s", mipk_status_string(status));
    }
    mipk_diag_end();
    return status;
}

mipk_status mipk_pipeline_reset(mipk_pipeline *pipeline)
{
    if (pipeline == NULL) return MIPK_ERR_ARGUMENT;
    mipk_status status = mipk_pipeline_runtime_reset(pipeline->runtime);
    if (status == MIPK_OK) pipeline->has_last_audio_pts = false;
    return status;
}

mipk_status mipk_pipeline_update_node_options(mipk_pipeline *pipeline,
    const char *node_id, const char *options_json)
{
    if (pipeline == NULL) return MIPK_ERR_ARGUMENT;
    arm_diagnostics(pipeline);
    const mipk_status status = mipk_pipeline_runtime_update_node_options(
        pipeline->runtime, node_id, options_json);
    mipk_diag_end();
    return status;
}

mipk_status mipk_pipeline_update_stage_options(mipk_pipeline *pipeline,
    const char *node_id, mipk_stage_type stage_type, const char *stage_name,
    const char *options_json)
{
    if (pipeline == NULL) return MIPK_ERR_ARGUMENT;
    arm_diagnostics(pipeline);
    const mipk_status status =
        mipk_pipeline_runtime_update_node_options_checked(pipeline->runtime,
            node_id, stage_type, stage_name, options_json);
    if (status == MIPK_ERR_ARGUMENT) {
        mipk_diag("node '%s' is not a %s stage or the options are invalid",
            node_id == NULL ? "" : node_id,
            stage_name == NULL ? "" : stage_name);
    }
    mipk_diag_end();
    return status;
}

mipk_status mipk_pipeline_apply_overrides_json(mipk_pipeline *pipeline,
    const char *overrides_json)
{
    if (pipeline == NULL) return MIPK_ERR_ARGUMENT;
    arm_diagnostics(pipeline);
    const mipk_status status = mipk_pipeline_runtime_apply_overrides_json(
        pipeline->runtime, overrides_json);
    mipk_diag_end();
    return status;
}

mipk_status mipk_pipeline_load_overrides(mipk_pipeline *pipeline,
    const char *path)
{
    if (pipeline == NULL || path == NULL || path[0] == '\0') {
        return MIPK_ERR_ARGUMENT;
    }
    struct stat attributes;
    if (stat(path, &attributes) != 0 || !S_ISREG(attributes.st_mode) ||
        attributes.st_size <= 0 ||
        (uint64_t)attributes.st_size > MIPK_MAX_OVERRIDES_BYTES) {
        return MIPK_ERR_IO;
    }
    const size_t size = (size_t)attributes.st_size;
    char *content = malloc(size + 1);
    if (content == NULL) return MIPK_ERR_MEMORY;
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        free(content);
        return MIPK_ERR_IO;
    }
    const size_t read_size = fread(content, 1, size, file);
    const bool stream_ok = read_size == size && ferror(file) == 0;
    const bool close_ok = fclose(file) == 0;
    const bool read_ok = stream_ok && close_ok;
    if (!read_ok) {
        free(content);
        return MIPK_ERR_IO;
    }
    content[size] = '\0';
    const mipk_status status = mipk_pipeline_apply_overrides_json(pipeline,
        content);
    free(content);
    return status;
}

uint64_t mipk_output_pts_us(const mipk_output *output)
{
    return output == NULL || !output->leased ? MIPK_PTS_UNKNOWN :
        output->pts_us;
}

size_t mipk_output_result_count(const mipk_output *output)
{
    return output == NULL || !output->leased ? 0 : output->results.count;
}

const mipk_result *mipk_output_result_at(const mipk_output *output,
    size_t index)
{
    return output == NULL || !output->leased ||
        index >= output->results.count ? NULL : &output->results.items[index];
}

size_t mipk_output_attachment_count(const mipk_output *output)
{
    (void)output;
    return 0;
}

mipk_status mipk_output_attachment_at(const mipk_output *output,
    size_t index, mipk_attachment_type *type, const void **data, size_t *size)
{
    (void)output;
    (void)index;
    (void)type;
    (void)data;
    (void)size;
    return MIPK_ERR_ARGUMENT;
}

const char *mipk_data_format_string(mipk_data_format format)
{
    switch (format) {
        case MIPK_FORMAT_NV12: return "nv12";
        case MIPK_FORMAT_NV21: return "nv21";
        case MIPK_FORMAT_BGR: return "bgr";
        case MIPK_FORMAT_RGB: return "rgb";
        case MIPK_FORMAT_PCM_S16LE: return "pcm_s16le";
        default: return "unknown";
    }
}

const char *mipk_status_string(mipk_status status)
{
    switch (status) {
        case MIPK_OK: return "success";
        case MIPK_AGAIN: return "input pending";
        case MIPK_ERR_ARGUMENT: return "invalid argument";
        case MIPK_ERR_MEMORY: return "out of memory";
        case MIPK_ERR_IO: return "I/O error";
        case MIPK_ERR_RUNTIME: return "runtime error";
        case MIPK_ERR_MODEL: return "invalid or unsupported model";
        case MIPK_ERR_OPERATOR: return "pipeline operator error";
        case MIPK_ERR_CAPACITY: return "result capacity exceeded";
        case MIPK_ERR_UNSUPPORTED: return "unsupported operation";
        case MIPK_ERR_SECURITY: return "package authentication failed";
        case MIPK_ERR_BUSY: return "resource busy";
        default: return "unknown error";
    }
}
