#include "mipk_module.h"

#include "mipk_tensor.h"
#include "cJSON.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    KEYPOINT_COORDINATES_NORMALIZED = 0,
    KEYPOINT_COORDINATES_PIXELS,
} keypoint_coordinates;

typedef struct {
    mipk_postprocess_config config;
    size_t output_index;
    size_t point_count;
    size_t values_per_keypoint;
    size_t class_id;
    size_t instance_id;
    keypoint_coordinates coordinates;
    float *values;
    size_t value_capacity;
    mipk_result *keypoints;
} keypoints_context;

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

static const char *json_string(const cJSON *json, const char *name,
    const char *default_value)
{
    const cJSON *item = cJSON_GetObjectItemCaseSensitive(json, name);
    if (item == NULL) return default_value;
    return cJSON_IsString(item) && item->valuestring != NULL ?
        item->valuestring : NULL;
}

static void keypoints_destroy(void *private_data)
{
    keypoints_context *context = private_data;
    if (context == NULL) return;
    free(context->keypoints);
    free(context->values);
    free(context);
}

static void *keypoints_create(const mipk_postprocess_config *config,
    const char *options_json, mipk_status *status)
{
    if (status == NULL) return NULL;
    *status = MIPK_ERR_ARGUMENT;
    if (config == NULL || config->classes == NULL || config->class_count == 0 ||
        config->input_width == 0 || config->input_height == 0 ||
        config->result_capacity == 0 || options_json == NULL) return NULL;
    keypoints_context *context = calloc(1, sizeof(*context));
    if (context == NULL) {
        *status = MIPK_ERR_MEMORY;
        return NULL;
    }
    context->config = *config;
    cJSON *json = cJSON_Parse(options_json);
    const char *coordinates = json == NULL ? NULL :
        json_string(json, "coordinates", "normalized");
    bool valid = json != NULL && coordinates != NULL &&
        json_size(json, "output_index", 0, &context->output_index) &&
        json_size(json, "point_count", 0, &context->point_count) &&
        json_size(json, "values_per_keypoint", 2,
            &context->values_per_keypoint) &&
        json_size(json, "class_id", 0, &context->class_id) &&
        json_size(json, "instance_id", 0, &context->instance_id) &&
        context->point_count != 0 &&
        context->point_count <= config->result_capacity &&
        (context->values_per_keypoint == 2 ||
            context->values_per_keypoint == 3) &&
        context->class_id < config->class_count &&
        context->instance_id <= INT32_MAX && context->point_count <= INT32_MAX;
    if (valid && strcmp(coordinates, "normalized") == 0) {
        context->coordinates = KEYPOINT_COORDINATES_NORMALIZED;
    } else if (valid && strcmp(coordinates, "pixels") == 0) {
        context->coordinates = KEYPOINT_COORDINATES_PIXELS;
    } else {
        valid = false;
    }
    cJSON_Delete(json);
    if (!valid || context->point_count > SIZE_MAX / sizeof(*context->keypoints)) {
        free(context);
        return NULL;
    }
    context->keypoints = calloc(context->point_count,
        sizeof(*context->keypoints));
    if (context->keypoints == NULL) {
        *status = MIPK_ERR_MEMORY;
        free(context);
        return NULL;
    }
    *status = MIPK_OK;
    return context;
}

static mipk_status keypoints_validate(void *private_data,
    const mipk_mutable_tensor *inputs, size_t input_count,
    const mipk_tensor *outputs, size_t output_count)
{
    keypoints_context *context = private_data;
    (void)inputs;
    (void)input_count;
    if (context == NULL || outputs == NULL ||
        context->output_index >= output_count ||
        context->point_count > SIZE_MAX / context->values_per_keypoint) {
        return MIPK_ERR_MODEL;
    }
    size_t value_count = 0;
    mipk_status status = mipk_tensor_float_count(
        &outputs[context->output_index], &value_count);
    const size_t required = context->point_count * context->values_per_keypoint;
    if (status != MIPK_OK || value_count < required ||
        value_count > SIZE_MAX / sizeof(float)) {
        mipk_diag("keypoints: output %zu has %zu values but %zu points x %zu "
            "values need %zu", context->output_index, value_count,
            context->point_count, context->values_per_keypoint, required);
        return MIPK_ERR_MODEL;
    }
    context->values = malloc(value_count * sizeof(*context->values));
    if (context->values == NULL) return MIPK_ERR_MEMORY;
    context->value_capacity = value_count;
    return MIPK_OK;
}

static mipk_status keypoints_process(void *private_data,
    const mipk_tensor *outputs, size_t output_count,
    mipk_postprocess_output *output)
{
    keypoints_context *context = private_data;
    if (context == NULL || outputs == NULL || output == NULL ||
        context->output_index >= output_count) return MIPK_ERR_ARGUMENT;
    *output = (mipk_postprocess_output) {0};
    size_t value_count = 0;
    mipk_status status = mipk_tensor_to_float_buffer(
        &outputs[context->output_index], context->values,
        context->value_capacity, &value_count);
    if (status != MIPK_OK) return status;
    const size_t required = context->point_count * context->values_per_keypoint;
    if (value_count < required) return MIPK_ERR_MODEL;
    for (size_t index = 0; index < context->point_count; ++index) {
        const size_t offset = index * context->values_per_keypoint;
        float x = context->values[offset];
        float y = context->values[offset + 1U];
        float visibility = context->values_per_keypoint == 3 ?
            context->values[offset + 2U] : 1.0F;
        if (!isfinite(x) || !isfinite(y) || !isfinite(visibility)) {
            return MIPK_ERR_MODEL;
        }
        if (context->coordinates == KEYPOINT_COORDINATES_NORMALIZED) {
            x *= (float)context->config.input_width;
            y *= (float)context->config.input_height;
        }
        const float denominator = context->config.to_source[6] * x +
            context->config.to_source[7] * y +
            context->config.to_source[8];
        if (!isfinite(denominator) || fabsf(denominator) < 1.0e-6F) {
            return MIPK_ERR_MODEL;
        }
        const float source_x = (context->config.to_source[0] * x +
            context->config.to_source[1] * y +
            context->config.to_source[2]) / denominator;
        const float source_y = (context->config.to_source[3] * x +
            context->config.to_source[4] * y +
            context->config.to_source[5]) / denominator;
        context->keypoints[index] = (mipk_result) {
            .type = MIPK_RESULT_KEYPOINT,
            .class_id = (int32_t)context->class_id,
            .label_set_id = context->config.label_set_id,
            .track_id = MIPK_INVALID_TRACK_ID,
            .attachment_index = MIPK_INVALID_ATTACHMENT_INDEX,
            .score = visibility,
            .value.keypoint = {
                .x = fminf(fmaxf(source_x, 0.0F),
                    context->config.source_width),
                .y = fminf(fmaxf(source_y, 0.0F),
                    context->config.source_height),
                .visibility = visibility,
                .keypoint_id = (int32_t)index,
            },
        };
    }
    output->items = context->keypoints;
    output->item_count = context->point_count;
    output->item_capacity = context->point_count;
    return MIPK_OK;
}

static const mipk_capability_descriptor g_capability =
    MIPK_POSTPROCESS_CAPABILITY("keypoints", "2.0.0");

static const char *const g_options[] = {
    "output_index",
    "point_count",
    "values_per_keypoint",
    "coordinates",
    "class_id",
    "instance_id",
};

static const mipk_port_schema g_inputs[] = {
    MIPK_TENSOR_INPUT_PORT("tensors"),
};

static const mipk_node_schema g_schema = {
    .inputs = g_inputs,
    .input_count = sizeof(g_inputs) / sizeof(g_inputs[0]),
    .option_names = g_options,
    .option_count = sizeof(g_options) / sizeof(g_options[0]),
    .output_type = MIPK_VALUE_RESULTS,
    .output_domain = MIPK_DOMAIN_IMAGE,
    .output_results = {
        .result_type_mask = MIPK_RESULT_MASK(MIPK_RESULT_KEYPOINT),
        .label_set_rule = MIPK_LABEL_SET_FIXED,
    },
    .result_bound_option = "point_count",
};

const mipk_postprocess_ops *mipk_keypoints_postprocess(void)
{
    static const mipk_postprocess_ops operations = {
        .capability = &g_capability,
        .schema = &g_schema,
        .name = "keypoints",
        .create = keypoints_create,
        .destroy = keypoints_destroy,
        .validate = keypoints_validate,
        .process = keypoints_process,
    };
    return &operations;
}

mipk_status mipk_keypoints_register(void)
{
    return mipk_postprocess_register(mipk_keypoints_postprocess());
}
