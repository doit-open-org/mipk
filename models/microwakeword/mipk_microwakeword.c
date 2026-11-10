#include "mipk_module.h"

#include "mipk_tensor.h"
#include "cJSON.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    TRANSFORM_PROBABILITY = 0,
    TRANSFORM_SIGMOID,
    TRANSFORM_SOFTMAX,
} output_transform;

typedef struct {
    mipk_postprocess_config config;
    output_transform transform;
    size_t output_index;
    size_t *class_output_indices;
    float *probabilities;
    float *output_values;
    size_t output_value_capacity;
} microwakeword_context;

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

static bool parse_class_indices(microwakeword_context *context,
    const cJSON *json)
{
    const cJSON *indices = cJSON_GetObjectItemCaseSensitive(json,
        "class_output_indices");
    if (indices == NULL) {
        for (size_t index = 0; index < context->config.class_count; ++index) {
            context->class_output_indices[index] = index;
        }
        return true;
    }
    const int count = cJSON_GetArraySize(indices);
    if (!cJSON_IsArray(indices) || count < 0 ||
        (size_t)count != context->config.class_count) return false;
    for (size_t index = 0; index < context->config.class_count; ++index) {
        const cJSON *item = cJSON_GetArrayItem(indices, (int)index);
        if (!cJSON_IsNumber(item) || !isfinite(item->valuedouble) ||
            item->valuedouble < 0.0 || item->valuedouble > (double)SIZE_MAX ||
            floor(item->valuedouble) != item->valuedouble) return false;
        context->class_output_indices[index] = (size_t)item->valuedouble;
        for (size_t previous = 0; previous < index; ++previous) {
            if (context->class_output_indices[previous] ==
                context->class_output_indices[index]) return false;
        }
    }
    return true;
}

static bool parse_options(microwakeword_context *context,
    const char *options_json)
{
    cJSON *json = cJSON_Parse(options_json);
    const char *transform = json == NULL ? NULL :
        json_string(json, "transform", "probability");
    bool valid = json != NULL && transform != NULL &&
        json_size(json, "output_index", 0, &context->output_index) &&
        parse_class_indices(context, json);
    if (valid && strcmp(transform, "probability") == 0) {
        context->transform = TRANSFORM_PROBABILITY;
    } else if (valid && strcmp(transform, "sigmoid") == 0) {
        context->transform = TRANSFORM_SIGMOID;
    } else if (valid && strcmp(transform, "softmax") == 0) {
        context->transform = TRANSFORM_SOFTMAX;
    } else {
        valid = false;
    }
    cJSON_Delete(json);
    return valid;
}

static void microwakeword_destroy(void *private_data)
{
    microwakeword_context *context = private_data;
    if (context == NULL) return;
    free(context->output_values);
    free(context->probabilities);
    free(context->class_output_indices);
    free(context);
}

static void *microwakeword_create(const mipk_postprocess_config *config,
    const char *options_json, mipk_status *status)
{
    if (status == NULL) return NULL;
    *status = MIPK_ERR_ARGUMENT;
    if (config == NULL || config->classes == NULL || config->class_count == 0 ||
        options_json == NULL) return NULL;
    microwakeword_context *context = calloc(1, sizeof(*context));
    if (context == NULL) {
        *status = MIPK_ERR_MEMORY;
        return NULL;
    }
    context->config = *config;
    context->class_output_indices = calloc(config->class_count,
        sizeof(*context->class_output_indices));
    context->probabilities = calloc(config->class_count,
        sizeof(*context->probabilities));
    if (context->class_output_indices == NULL ||
        context->probabilities == NULL || !parse_options(context, options_json)) {
        microwakeword_destroy(context);
        return NULL;
    }
    *status = MIPK_OK;
    return context;
}

static mipk_status microwakeword_validate(void *private_data,
    const mipk_mutable_tensor *inputs, size_t input_count,
    const mipk_tensor *outputs, size_t output_count)
{
    microwakeword_context *context = private_data;
    (void)inputs;
    (void)input_count;
    if (context == NULL || outputs == NULL ||
        context->output_index >= output_count) return MIPK_ERR_MODEL;
    size_t value_count = 0;
    mipk_status status = mipk_tensor_float_count(
        &outputs[context->output_index], &value_count);
    if (status != MIPK_OK || value_count == 0 ||
        value_count > SIZE_MAX / sizeof(float)) return MIPK_ERR_MODEL;
    for (size_t index = 0; index < context->config.class_count; ++index) {
        if (context->class_output_indices[index] >= value_count) {
            return MIPK_ERR_MODEL;
        }
    }
    context->output_values = malloc(value_count * sizeof(*context->output_values));
    if (context->output_values == NULL) return MIPK_ERR_MEMORY;
    context->output_value_capacity = value_count;
    return MIPK_OK;
}

static bool adapt_output(microwakeword_context *context, size_t value_count)
{
    float maximum = -FLT_MAX;
    float sum = 0.0F;
    if (context->transform == TRANSFORM_SOFTMAX) {
        for (size_t index = 0; index < value_count; ++index) {
            if (!isfinite(context->output_values[index])) return false;
            if (context->output_values[index] > maximum) {
                maximum = context->output_values[index];
            }
        }
        for (size_t index = 0; index < value_count; ++index) {
            sum += expf(context->output_values[index] - maximum);
        }
        if (!(sum > 0.0F) || !isfinite(sum)) return false;
    }
    for (size_t index = 0; index < context->config.class_count; ++index) {
        const float value =
            context->output_values[context->class_output_indices[index]];
        float probability = value;
        if (context->transform == TRANSFORM_SIGMOID) {
            if (!isfinite(value)) return false;
            probability = value >= 0.0F ?
                1.0F / (1.0F + expf(-value)) :
                expf(value) / (1.0F + expf(value));
        } else if (context->transform == TRANSFORM_SOFTMAX) {
            probability = expf(value - maximum) / sum;
        }
        if (!(probability >= 0.0F && probability <= 1.0F)) return false;
        context->probabilities[index] = probability;
    }
    return true;
}

static mipk_status microwakeword_process(void *private_data,
    const mipk_tensor *outputs, size_t output_count,
    mipk_postprocess_output *output)
{
    microwakeword_context *context = private_data;
    if (context == NULL || outputs == NULL || output == NULL ||
        context->output_index >= output_count) {
        return MIPK_ERR_ARGUMENT;
    }
    *output = (mipk_postprocess_output) {0};
    size_t value_count = 0;
    mipk_status status = mipk_tensor_to_float_buffer(
        &outputs[context->output_index],
        context->output_values, context->output_value_capacity, &value_count);
    if (status != MIPK_OK || !adapt_output(context, value_count)) {
        return status == MIPK_OK ? MIPK_ERR_MODEL : status;
    }
    output->scores = context->probabilities;
    output->score_count = context->config.class_count;
    return MIPK_OK;
}

static const mipk_capability_descriptor g_capability =
    MIPK_POSTPROCESS_CAPABILITY("microwakeword", "2.0.0");

static const char *const g_options[] = {
    "output_index",
    "transform",
    "class_output_indices",
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
    .output_domain = MIPK_DOMAIN_AUDIO,
    .output_results = {
        .result_type_mask = MIPK_RESULT_MASK(MIPK_RESULT_CLASSIFICATION),
        .label_set_rule = MIPK_LABEL_SET_FIXED,
    },
};

const mipk_postprocess_ops *mipk_microwakeword_postprocess(void)
{
    static const mipk_postprocess_ops operations = {
        .capability = &g_capability,
        .schema = &g_schema,
        .name = "microwakeword",
        .create = microwakeword_create,
        .destroy = microwakeword_destroy,
        .process = microwakeword_process,
        .validate = microwakeword_validate,
    };
    return &operations;
}

mipk_status mipk_microwakeword_register(void)
{
    return mipk_postprocess_register(mipk_microwakeword_postprocess());
}
