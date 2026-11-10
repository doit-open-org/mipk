#include "mipk_module.h"

#include "mipk_tensor.h"
#include "cJSON.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    CLASSIFICATION_PROBABILITY = 0,
    CLASSIFICATION_SIGMOID,
    CLASSIFICATION_SOFTMAX,
} classification_transform;

typedef struct {
    mipk_postprocess_config config;
    classification_transform transform;
    size_t output_index;
    size_t top_k;
    float score_threshold;
    size_t *class_output_indices;
    float *values;
    size_t value_capacity;
    mipk_result *candidates;
    size_t candidate_capacity;
} classification_context;

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

static bool parse_class_indices(classification_context *context,
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
    if (!cJSON_IsArray(indices) ||
        cJSON_GetArraySize(indices) != (int)context->config.class_count) return false;
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

static void classification_destroy(void *private_data)
{
    classification_context *context = private_data;
    if (context == NULL) return;
    free(context->candidates);
    free(context->values);
    free(context->class_output_indices);
    free(context);
}

static void *classification_create(const mipk_postprocess_config *config,
    const char *options_json, mipk_status *status)
{
    if (status == NULL) return NULL;
    *status = MIPK_ERR_ARGUMENT;
    if (config == NULL || config->classes == NULL || config->class_count == 0 ||
        config->result_capacity == 0 || options_json == NULL) return NULL;
    classification_context *context = calloc(1, sizeof(*context));
    if (context == NULL) {
        *status = MIPK_ERR_MEMORY;
        return NULL;
    }
    context->config = *config;
    context->class_output_indices = calloc(config->class_count,
        sizeof(*context->class_output_indices));
    cJSON *json = cJSON_Parse(options_json);
    const char *transform = json == NULL ? NULL :
        json_string(json, "transform", "probability");
    const cJSON *threshold = json == NULL ? NULL :
        cJSON_GetObjectItemCaseSensitive(json, "score_threshold");
    bool valid = json != NULL && transform != NULL &&
        context->class_output_indices != NULL &&
        json_size(json, "output_index", 0, &context->output_index) &&
        json_size(json, "top_k", config->result_capacity, &context->top_k) &&
        context->top_k != 0 && parse_class_indices(context, json) &&
        (threshold == NULL || (cJSON_IsNumber(threshold) &&
            isfinite(threshold->valuedouble) && threshold->valuedouble >= 0.0 &&
            threshold->valuedouble <= 1.0));
    context->score_threshold = threshold == NULL ? 0.0F :
        (float)threshold->valuedouble;
    if (valid && strcmp(transform, "probability") == 0) {
        context->transform = CLASSIFICATION_PROBABILITY;
    } else if (valid && strcmp(transform, "sigmoid") == 0) {
        context->transform = CLASSIFICATION_SIGMOID;
    } else if (valid && strcmp(transform, "softmax") == 0) {
        context->transform = CLASSIFICATION_SOFTMAX;
    } else {
        valid = false;
    }
    cJSON_Delete(json);
    if (!valid) {
        classification_destroy(context);
        return NULL;
    }
    context->candidate_capacity = context->top_k;
    if (context->candidate_capacity > config->class_count) {
        context->candidate_capacity = config->class_count;
    }
    if (context->candidate_capacity > config->result_capacity) {
        context->candidate_capacity = config->result_capacity;
    }
    context->candidates = calloc(context->candidate_capacity,
        sizeof(*context->candidates));
    if (context->candidates == NULL) {
        *status = MIPK_ERR_MEMORY;
        classification_destroy(context);
        return NULL;
    }
    *status = MIPK_OK;
    return context;
}

static mipk_status classification_validate(void *private_data,
    const mipk_mutable_tensor *inputs, size_t input_count,
    const mipk_tensor *outputs, size_t output_count)
{
    classification_context *context = private_data;
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
            mipk_diag("classification: class %zu reads output value %zu but "
                "output %zu has only %zu values", index,
                context->class_output_indices[index], context->output_index,
                value_count);
            return MIPK_ERR_MODEL;
        }
    }
    context->values = malloc(value_count * sizeof(*context->values));
    if (context->values == NULL) return MIPK_ERR_MEMORY;
    context->value_capacity = value_count;
    return MIPK_OK;
}

static float class_threshold(const classification_context *context,
    size_t class_id)
{
    const float threshold = context->config.classes[class_id].score_threshold;
    return threshold < 0.0F ? context->score_threshold : threshold;
}

static void retain_candidate(classification_context *context, size_t *count,
    const mipk_result *candidate)
{
    size_t position = *count;
    if (position < context->candidate_capacity) ++*count;
    else if (context->candidate_capacity != 0 &&
        candidate->score > context->candidates[position - 1].score) {
        position = context->candidate_capacity - 1;
    } else {
        return;
    }
    while (position != 0 &&
        context->candidates[position - 1].score < candidate->score) {
        if (position < context->candidate_capacity) {
            context->candidates[position] = context->candidates[position - 1];
        }
        --position;
    }
    context->candidates[position] = *candidate;
}

static mipk_status classification_process(void *private_data,
    const mipk_tensor *outputs, size_t output_count,
    mipk_postprocess_output *output)
{
    classification_context *context = private_data;
    if (context == NULL || outputs == NULL || output == NULL ||
        context->output_index >= output_count) return MIPK_ERR_ARGUMENT;
    *output = (mipk_postprocess_output) {0};
    size_t value_count = 0;
    mipk_status status = mipk_tensor_to_float_buffer(
        &outputs[context->output_index], context->values,
        context->value_capacity, &value_count);
    if (status != MIPK_OK) return status;
    float maximum = -FLT_MAX;
    float sum = 0.0F;
    if (context->transform == CLASSIFICATION_SOFTMAX) {
        for (size_t class_id = 0; class_id < context->config.class_count;
            ++class_id) {
            const float value =
                context->values[context->class_output_indices[class_id]];
            if (!isfinite(value)) return MIPK_ERR_MODEL;
            if (value > maximum) maximum = value;
        }
        for (size_t class_id = 0; class_id < context->config.class_count;
            ++class_id) {
            sum += expf(context->values[context->class_output_indices[class_id]] -
                maximum);
        }
        if (!(sum > 0.0F) || !isfinite(sum)) return MIPK_ERR_MODEL;
    }
    size_t count = 0;
    for (size_t class_id = 0; class_id < context->config.class_count; ++class_id) {
        if (!context->config.classes[class_id].enabled) continue;
        const float value = context->values[context->class_output_indices[class_id]];
        float score = value;
        if (context->transform == CLASSIFICATION_SIGMOID) {
            if (!isfinite(value)) return MIPK_ERR_MODEL;
            score = value >= 0.0F ? 1.0F / (1.0F + expf(-value)) :
                expf(value) / (1.0F + expf(value));
        } else if (context->transform == CLASSIFICATION_SOFTMAX) {
            score = expf(value - maximum) / sum;
        }
        if (!isfinite(score) || score < 0.0F || score > 1.0F) {
            return MIPK_ERR_MODEL;
        }
        if (score < class_threshold(context, class_id)) continue;
        const mipk_result candidate = {
            .type = MIPK_RESULT_CLASSIFICATION,
            .class_id = (int32_t)class_id,
            .label_set_id = context->config.label_set_id,
            .track_id = MIPK_INVALID_TRACK_ID,
            .attachment_index = MIPK_INVALID_ATTACHMENT_INDEX,
            .score = score,
        };
        retain_candidate(context, &count, &candidate);
    }
    output->items = context->candidates;
    output->item_count = count;
    output->item_capacity = context->candidate_capacity;
    return MIPK_OK;
}

static const mipk_capability_descriptor g_capability =
    MIPK_POSTPROCESS_CAPABILITY("classification", "2.0.0");

static const char *const g_options[] = {
    "output_index",
    "transform",
    "class_output_indices",
    "score_threshold",
    "top_k",
    "max_items",
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
        .result_type_mask = MIPK_RESULT_MASK(MIPK_RESULT_CLASSIFICATION),
        .label_set_rule = MIPK_LABEL_SET_FIXED,
    },
    .result_bound_option = "top_k",
};

const mipk_postprocess_ops *mipk_classification_postprocess(void)
{
    static const mipk_runtime_option runtime_options[] = {
        {.struct_size = sizeof(mipk_runtime_option),
            .name = "score_threshold", .type = MIPK_RUNTIME_OPTION_FLOAT,
            .description = "Minimum classification score",
            .default_json = "0"},
    };
    static const mipk_postprocess_ops operations = {
        .capability = &g_capability,
        .schema = &g_schema,
        .name = "classification",
        .create = classification_create,
        .destroy = classification_destroy,
        .validate = classification_validate,
        .process = classification_process,
        .runtime_options = runtime_options,
        .runtime_option_count = sizeof(runtime_options) /
            sizeof(runtime_options[0]),
    };
    return &operations;
}

mipk_status mipk_classification_register(void)
{
    return mipk_postprocess_register(mipk_classification_postprocess());
}
