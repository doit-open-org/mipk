#include "mipk_module.h"

#include "mipk_memory.h"
#include "mipk_microfrontend.h"
#include "mipk_tensor.h"
#include "cJSON.h"

#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    FRONTEND_TIME_MAJOR,
    FRONTEND_FEATURE_MAJOR,
} frontend_layout;

typedef struct {
    mipk_microfrontend_config frontend_config;
    mipk_microfrontend *frontend;
    frontend_layout layout;
    size_t feature_frames;
    size_t inference_stride;
    size_t input_index;
    size_t input_size;
    float *frames;
    float *scratch;
    size_t next_frame;
    size_t populated_frames;
    size_t frames_since_inference;
} frontend_context;

#define FRONTEND_CHECKPOINT_BYTES (256U * 1024U)

typedef struct {
    size_t micro_size;
    size_t next_frame;
    size_t populated_frames;
    size_t frames_since_inference;
} frontend_checkpoint_header;

static bool checked_product(size_t first, size_t second, size_t *value)
{
    if (first != 0 && second > SIZE_MAX / first) return false;
    *value = first * second;
    return true;
}

static bool json_size(const cJSON *json, const char *name,
    size_t default_value, size_t *value)
{
    const cJSON *item = cJSON_GetObjectItemCaseSensitive(json, name);
    if (item == NULL) {
        *value = default_value;
        return true;
    }
    if (!cJSON_IsNumber(item) || !isfinite(item->valuedouble) ||
        item->valuedouble < 0.0 || item->valuedouble > (double)SIZE_MAX ||
        floor(item->valuedouble) != item->valuedouble) return false;
    *value = (size_t)item->valuedouble;
    return true;
}

static bool json_int(const cJSON *json, const char *name,
    int default_value, int *value)
{
    const cJSON *item = cJSON_GetObjectItemCaseSensitive(json, name);
    if (item == NULL) {
        *value = default_value;
        return true;
    }
    if (!cJSON_IsNumber(item) || !isfinite(item->valuedouble) ||
        item->valuedouble < INT_MIN || item->valuedouble > INT_MAX ||
        floor(item->valuedouble) != item->valuedouble) return false;
    *value = (int)item->valuedouble;
    return true;
}

static bool json_float(const cJSON *json, const char *name,
    float default_value, float *value)
{
    const cJSON *item = cJSON_GetObjectItemCaseSensitive(json, name);
    if (item == NULL) {
        *value = default_value;
        return true;
    }
    if (!cJSON_IsNumber(item) || !isfinite(item->valuedouble)) return false;
    *value = (float)item->valuedouble;
    return true;
}

static bool json_bool(const cJSON *json, const char *name,
    bool default_value, bool *value)
{
    const cJSON *item = cJSON_GetObjectItemCaseSensitive(json, name);
    if (item == NULL) {
        *value = default_value;
        return true;
    }
    if (!cJSON_IsBool(item)) return false;
    *value = cJSON_IsTrue(item);
    return true;
}

static const char *json_string(const cJSON *json, const char *name,
    const char *default_value)
{
    const cJSON *item = cJSON_GetObjectItemCaseSensitive(json, name);
    if (item == NULL) return default_value;
    return cJSON_IsString(item) && item->valuestring != NULL ?
        item->valuestring : NULL;
}

static bool parse_options(frontend_context *context,
    const mipk_operator_config *config, const char *options_json)
{
    cJSON *json = cJSON_Parse(options_json);
    const char *layout = json == NULL ? NULL :
        json_string(json, "layout", "time_major");
    size_t window_ms = 0;
    size_t step_ms = 0;
    size_t feature_channels = 0;
    bool valid = json != NULL && config->input_sample_rate > 0 &&
        config->input_channels == 1 && config->input_frame_samples > 0 &&
        json_size(json, "window_ms", 30, &window_ms) &&
        json_size(json, "step_ms", 10, &step_ms) &&
        json_size(json, "feature_channels", 0, &feature_channels) &&
        json_size(json, "feature_frames", 0, &context->feature_frames) &&
        json_size(json, "inference_stride", 1, &context->inference_stride) &&
        json_size(json, "input_index", 0, &context->input_index) &&
        json_float(json, "lower_band_hz", 125.0F,
            &context->frontend_config.lower_band_limit) &&
        json_float(json, "upper_band_hz", 7500.0F,
            &context->frontend_config.upper_band_limit) &&
        json_int(json, "noise_smoothing_bits", 10,
            &context->frontend_config.noise_smoothing_bits) &&
        json_float(json, "noise_even_smoothing", 0.025F,
            &context->frontend_config.noise_even_smoothing) &&
        json_float(json, "noise_odd_smoothing", 0.06F,
            &context->frontend_config.noise_odd_smoothing) &&
        json_float(json, "noise_min_signal_remaining", 0.05F,
            &context->frontend_config.noise_min_signal_remaining) &&
        json_bool(json, "pcan", true,
            &context->frontend_config.enable_pcan) &&
        json_float(json, "pcan_strength", 0.95F,
            &context->frontend_config.pcan_strength) &&
        json_float(json, "pcan_offset", 80.0F,
            &context->frontend_config.pcan_offset) &&
        json_int(json, "pcan_gain_bits", 21,
            &context->frontend_config.pcan_gain_bits) &&
        json_bool(json, "log_scale", true,
            &context->frontend_config.enable_log) &&
        json_int(json, "log_scale_shift", 6,
            &context->frontend_config.log_scale_shift) &&
        json_float(json, "output_scale", 0.0390625F,
            &context->frontend_config.output_scale) &&
        window_ms > 0 && window_ms <= INT_MAX && step_ms > 0 &&
        step_ms <= INT_MAX && feature_channels > 0 &&
        context->feature_frames > 0 && context->inference_stride > 0;
    if (valid && strcmp(layout, "time_major") == 0) {
        context->layout = FRONTEND_TIME_MAJOR;
    } else if (valid && strcmp(layout, "feature_major") == 0) {
        context->layout = FRONTEND_FEATURE_MAJOR;
    } else {
        valid = false;
    }
    if (valid) {
        context->frontend_config.sample_rate = config->input_sample_rate;
        context->frontend_config.audio_frame_samples =
            config->input_frame_samples;
        context->frontend_config.feature_count = feature_channels;
        context->frontend_config.window_size_ms = (int)window_ms;
        context->frontend_config.window_step_ms = (int)step_ms;
    }
    cJSON_Delete(json);
    return valid;
}

static void frontend_destroy(void *private_data)
{
    frontend_context *context = private_data;
    if (context == NULL) return;
    mipk_microfrontend_destroy(context->frontend);
    free(context->scratch);
    free(context->frames);
    free(context);
}

static void *frontend_create(const mipk_operator_config *config,
    const char *options_json, mipk_status *status)
{
    if (status == NULL) return NULL;
    *status = MIPK_ERR_ARGUMENT;
    if (config == NULL || options_json == NULL ||
        config->input_size == 0 || config->input_frame_samples == 0 ||
        (uint64_t)config->input_size !=
            (uint64_t)config->input_frame_samples * sizeof(int16_t)) {
        return NULL;
    }
    frontend_context *context = calloc(1, sizeof(*context));
    if (context == NULL) {
        *status = MIPK_ERR_MEMORY;
        return NULL;
    }
    context->input_size = config->input_size;
    if (!parse_options(context, config, options_json)) goto failed;
    size_t value_count = 0;
    if (!checked_product(context->feature_frames,
            context->frontend_config.feature_count, &value_count)) goto failed;
    context->frames = calloc(value_count, sizeof(*context->frames));
    context->scratch = calloc(context->frontend_config.feature_count,
        sizeof(*context->scratch));
    context->frontend = mipk_microfrontend_create(&context->frontend_config);
    if (context->frames == NULL || context->scratch == NULL ||
        context->frontend == NULL) {
        *status = MIPK_ERR_MEMORY;
        frontend_destroy(context);
        return NULL;
    }
    *status = MIPK_OK;
    return context;

failed:
    frontend_destroy(context);
    return NULL;
}

static bool expected_shape(const frontend_context *context,
    const mipk_mutable_tensor *input)
{
    const int64_t first = context->layout == FRONTEND_TIME_MAJOR ?
        (int64_t)context->feature_frames :
        (int64_t)context->frontend_config.feature_count;
    const int64_t second = context->layout == FRONTEND_TIME_MAJOR ?
        (int64_t)context->frontend_config.feature_count :
        (int64_t)context->feature_frames;
    return (input->dim_count == 2 && input->dims[0] == first &&
            input->dims[1] == second) ||
        (input->dim_count == 3 && input->dims[0] == 1 &&
            input->dims[1] == first && input->dims[2] == second);
}

static mipk_status frontend_validate(void *private_data,
    const mipk_mutable_tensor *inputs, size_t input_count,
    const mipk_tensor *outputs, size_t output_count)
{
    const frontend_context *context = private_data;
    (void)outputs;
    (void)output_count;
    if (context == NULL || inputs == NULL ||
        context->input_index >= input_count) return MIPK_ERR_MODEL;
    size_t value_count = 0;
    if (!checked_product(context->feature_frames,
            context->frontend_config.feature_count, &value_count) ||
        value_count > SIZE_MAX / sizeof(float)) return MIPK_ERR_MODEL;
    const mipk_mutable_tensor *input = &inputs[context->input_index];
    return mipk_mutable_tensor_data(input) != NULL && input->data_type == MIPK_DTYPE_FLOAT32 &&
        input->byte_size >= value_count * sizeof(float) &&
        expected_shape(context, input) ? MIPK_OK : MIPK_ERR_MODEL;
}

static void copy_model_input(const frontend_context *context,
    mipk_mutable_tensor *input)
{
    float *destination = mipk_mutable_tensor_data(input);
    for (size_t frame = 0; frame < context->feature_frames; ++frame) {
        const size_t source_frame = (context->next_frame + frame) %
            context->feature_frames;
        const float *source = context->frames + source_frame *
            context->frontend_config.feature_count;
        if (context->layout == FRONTEND_TIME_MAJOR) {
            memcpy(destination + frame * context->frontend_config.feature_count,
                source, context->frontend_config.feature_count * sizeof(*source));
        } else {
            for (size_t feature = 0;
                feature < context->frontend_config.feature_count; ++feature) {
                destination[feature * context->feature_frames + frame] =
                    source[feature];
            }
        }
    }
}

static mipk_status frontend_process(void *private_data,
    mipk_operator_io *execution)
{
    frontend_context *context = private_data;
    if (context == NULL || execution == NULL || execution->input == NULL ||
        execution->model_inputs == NULL ||
        context->input_index >= execution->model_input_count ||
        mipk_mutable_tensor_data(
            &execution->model_inputs[context->input_index]) == NULL ||
        (execution->input->type != MIPK_INPUT_AUDIO &&
            execution->input->type != MIPK_INPUT_BYTES) ||
        mipk_input_data(execution->input) == NULL ||
        mipk_input_data_size(execution->input) != context->input_size) {
        return MIPK_ERR_ARGUMENT;
    }
    const int frontend_status = mipk_microfrontend_process(context->frontend,
        mipk_input_data(execution->input),
        mipk_input_data_size(execution->input) / sizeof(int16_t), context->scratch,
        context->frontend_config.feature_count);
    if (frontend_status < 0) return MIPK_ERR_RUNTIME;
    if (frontend_status == 0) return MIPK_AGAIN;

    float *destination = context->frames + context->next_frame *
        context->frontend_config.feature_count;
    memcpy(destination, context->scratch,
        context->frontend_config.feature_count * sizeof(*destination));
    context->next_frame = (context->next_frame + 1) % context->feature_frames;
    bool inference_due = false;
    if (context->populated_frames < context->feature_frames) {
        ++context->populated_frames;
        inference_due = context->populated_frames == context->feature_frames;
    } else if (++context->frames_since_inference >= context->inference_stride) {
        inference_due = true;
    }
    if (!inference_due) return MIPK_AGAIN;
    context->frames_since_inference = 0;
    mipk_mutable_tensor *model_input =
        &execution->model_inputs[context->input_index];
    if (mipk_memory_zero(mipk_mutable_tensor_data(model_input),
            model_input->byte_size) != 0) {
        return MIPK_ERR_RUNTIME;
    }
    copy_model_input(context, model_input);
    return MIPK_OK;
}

static mipk_status frontend_reset(void *private_data)
{
    frontend_context *context = private_data;
    if (context == NULL || context->frontend == NULL) return MIPK_ERR_ARGUMENT;
    mipk_microfrontend_reset(context->frontend);
    memset(context->frames, 0, context->feature_frames *
        context->frontend_config.feature_count * sizeof(*context->frames));
    memset(context->scratch, 0, context->frontend_config.feature_count *
        sizeof(*context->scratch));
    context->next_frame = 0;
    context->populated_frames = 0;
    context->frames_since_inference = 0;
    return MIPK_OK;
}

static size_t frontend_frame_bytes(const frontend_context *context)
{
    return context->feature_frames * context->frontend_config.feature_count *
        sizeof(*context->frames);
}

static mipk_status frontend_checkpoint(void *private_data, void *buffer,
    size_t size)
{
    frontend_context *context = private_data;
    if (context == NULL || buffer == NULL || size != FRONTEND_CHECKPOINT_BYTES) {
        return MIPK_ERR_ARGUMENT;
    }
    const size_t frames_size = frontend_frame_bytes(context);
    const size_t scratch_size = context->frontend_config.feature_count *
        sizeof(*context->scratch);
    const size_t micro_size = mipk_microfrontend_checkpoint_size(
        context->frontend);
    if (frames_size > size - sizeof(frontend_checkpoint_header) ||
        scratch_size > size - sizeof(frontend_checkpoint_header) - frames_size ||
        micro_size > size - sizeof(frontend_checkpoint_header) - frames_size -
            scratch_size) return MIPK_ERR_CAPACITY;
    frontend_checkpoint_header header = {
        .micro_size = micro_size,
        .next_frame = context->next_frame,
        .populated_frames = context->populated_frames,
        .frames_since_inference = context->frames_since_inference,
    };
    uint8_t *bytes = buffer;
    memcpy(bytes, &header, sizeof(header));
    memcpy(bytes + sizeof(header), context->frames, frames_size);
    memcpy(bytes + sizeof(header) + frames_size, context->scratch,
        scratch_size);
    return mipk_microfrontend_checkpoint(context->frontend,
        bytes + sizeof(header) + frames_size + scratch_size, micro_size) ?
        MIPK_OK : MIPK_ERR_RUNTIME;
}

static mipk_status frontend_restore(void *private_data, const void *buffer,
    size_t size)
{
    frontend_context *context = private_data;
    if (context == NULL || buffer == NULL || size != FRONTEND_CHECKPOINT_BYTES) {
        return MIPK_ERR_ARGUMENT;
    }
    frontend_checkpoint_header header;
    memcpy(&header, buffer, sizeof(header));
    const size_t frames_size = frontend_frame_bytes(context);
    const size_t scratch_size = context->frontend_config.feature_count *
        sizeof(*context->scratch);
    if (header.micro_size != mipk_microfrontend_checkpoint_size(
            context->frontend) || frames_size > size - sizeof(header) ||
        scratch_size > size - sizeof(header) - frames_size ||
        header.micro_size > size - sizeof(header) - frames_size - scratch_size) {
        return MIPK_ERR_RUNTIME;
    }
    const uint8_t *bytes = buffer;
    memcpy(context->frames, bytes + sizeof(header), frames_size);
    memcpy(context->scratch, bytes + sizeof(header) + frames_size,
        scratch_size);
    if (!mipk_microfrontend_restore(context->frontend,
            bytes + sizeof(header) + frames_size + scratch_size,
            header.micro_size)) return MIPK_ERR_RUNTIME;
    context->next_frame = header.next_frame;
    context->populated_frames = header.populated_frames;
    context->frames_since_inference = header.frames_since_inference;
    return MIPK_OK;
}

static const mipk_capability_descriptor g_capability =
    MIPK_OPERATOR_CAPABILITY("tfmicrofrontend", "2.0.0");

static const char *const g_options[] = {
    "window_ms",
    "step_ms",
    "feature_channels",
    "feature_frames",
    "inference_stride",
    "input_index",
    "lower_band_hz",
    "upper_band_hz",
    "noise_smoothing_bits",
    "noise_even_smoothing",
    "noise_odd_smoothing",
    "noise_min_signal_remaining",
    "pcan",
    "pcan_strength",
    "pcan_offset",
    "pcan_gain_bits",
    "log_scale",
    "log_scale_shift",
    "output_scale",
    "layout",
};

static const mipk_port_schema g_inputs[] = {
    {
        .port = "audio",
        .accepted_value_mask = MIPK_VALUE_MASK(MIPK_VALUE_INPUT),
        .accepted_domain_mask = MIPK_DOMAIN_MASK(MIPK_DOMAIN_AUDIO),
        .required = true,
    },
};

static const mipk_node_schema g_schema = {
    .inputs = g_inputs,
    .input_count = sizeof(g_inputs) / sizeof(g_inputs[0]),
    .option_names = g_options,
    .option_count = sizeof(g_options) / sizeof(g_options[0]),
    .output_type = MIPK_VALUE_TENSORS,
    .output_domain = MIPK_DOMAIN_AUDIO,
    .may_return_again = true,
};

const mipk_operator_ops *mipk_tfmicrofrontend_operator(void)
{
    static const mipk_operator_ops operations = {
        .capability = &g_capability,
        .schema = &g_schema,
        .name = "tfmicrofrontend",
        .create = frontend_create,
        .destroy = frontend_destroy,
        .process = frontend_process,
        .validate = frontend_validate,
        .stateful = true,
        .may_return_again = true,
        .reset = frontend_reset,
        .checkpoint_size = FRONTEND_CHECKPOINT_BYTES,
        .checkpoint = frontend_checkpoint,
        .restore = frontend_restore,
    };
    return &operations;
}

mipk_status mipk_tfmicrofrontend_register(void)
{
    return mipk_operator_register(mipk_tfmicrofrontend_operator());
}
