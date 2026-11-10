#include "mipk_module.h"

#include "cJSON.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    mipk_result *items;
    int32_t *allowlist;
    size_t allowlist_count;
    size_t top_k;
    size_t max_items;
    float score_threshold;
} result_filter_context;

static void result_filter_destroy(void *private_data)
{
    result_filter_context *context = private_data;
    if (context == NULL) return;
    free(context->allowlist);
    free(context->items);
    free(context);
}

static int32_t class_id_by_name(const mipk_operator_config *config,
    const char *name)
{
    for (size_t index = 0; index < config->class_count; ++index) {
        if (strcmp(config->classes[index].name, name) == 0) return (int32_t)index;
    }
    return -1;
}

static void *result_filter_create(const mipk_operator_config *config,
    const char *options_json, mipk_status *status)
{
    if (status == NULL) return NULL;
    *status = MIPK_ERR_ARGUMENT;
    if (config == NULL || config->classes == NULL || options_json == NULL) {
        return NULL;
    }
    cJSON *json = cJSON_Parse(options_json);
    const cJSON *top_k = cJSON_GetObjectItemCaseSensitive(json, "top_k");
    const cJSON *max_items = cJSON_GetObjectItemCaseSensitive(json, "max_items");
    const cJSON *threshold = cJSON_GetObjectItemCaseSensitive(json,
        "score_threshold");
    const cJSON *allowlist = cJSON_GetObjectItemCaseSensitive(json,
        "class_allowlist");
    if (!cJSON_IsObject(json) || !cJSON_IsNumber(top_k) ||
        !cJSON_IsNumber(max_items) || top_k->valuedouble < 1.0 ||
        max_items->valuedouble < 1.0 || floor(top_k->valuedouble) !=
            top_k->valuedouble || floor(max_items->valuedouble) !=
            max_items->valuedouble || top_k->valuedouble > max_items->valuedouble ||
        max_items->valuedouble > (double)config->result_capacity ||
        (threshold != NULL && (!cJSON_IsNumber(threshold) ||
            !isfinite(threshold->valuedouble) || threshold->valuedouble < 0.0 ||
            threshold->valuedouble > 1.0)) ||
        (allowlist != NULL && !cJSON_IsArray(allowlist))) {
        cJSON_Delete(json);
        return NULL;
    }
    result_filter_context *context = calloc(1, sizeof(*context));
    if (context == NULL) {
        cJSON_Delete(json);
        *status = MIPK_ERR_MEMORY;
        return NULL;
    }
    context->top_k = (size_t)top_k->valuedouble;
    context->max_items = (size_t)max_items->valuedouble;
    context->score_threshold = threshold == NULL ? 0.0F :
        (float)threshold->valuedouble;
    context->items = calloc(context->max_items, sizeof(*context->items));
    const int allow_count = cJSON_GetArraySize(allowlist);
    if (allow_count > 0) {
        context->allowlist = calloc((size_t)allow_count,
            sizeof(*context->allowlist));
    }
    bool valid = context->items != NULL &&
        (allow_count == 0 || context->allowlist != NULL);
    for (int index = 0; valid && index < allow_count; ++index) {
        const cJSON *item = cJSON_GetArrayItem(allowlist, index);
        if (!cJSON_IsString(item) || item->valuestring == NULL) {
            valid = false;
            break;
        }
        const int32_t class_id = class_id_by_name(config, item->valuestring);
        if (class_id < 0) {
            valid = false;
            break;
        }
        for (int previous = 0; previous < index; ++previous) {
            if (context->allowlist[previous] == class_id) valid = false;
        }
        context->allowlist[index] = class_id;
    }
    context->allowlist_count = valid ? (size_t)allow_count : 0;
    cJSON_Delete(json);
    if (!valid) {
        result_filter_destroy(context);
        return NULL;
    }
    *status = MIPK_OK;
    return context;
}

static bool class_allowed(const result_filter_context *context,
    int32_t class_id)
{
    if (context->allowlist_count == 0) return true;
    for (size_t index = 0; index < context->allowlist_count; ++index) {
        if (context->allowlist[index] == class_id) return true;
    }
    return false;
}

static mipk_status result_filter_process(void *private_data,
    mipk_operator_io *execution)
{
    result_filter_context *context = private_data;
    if (context == NULL || execution == NULL ||
        (execution->item_count != 0 && execution->items == NULL)) {
        return MIPK_ERR_ARGUMENT;
    }
    size_t count = 0;
    for (size_t index = 0; index < execution->item_count; ++index) {
        const mipk_result candidate = execution->items[index];
        if (!isfinite(candidate.score) ||
            candidate.score < context->score_threshold ||
            !class_allowed(context, candidate.class_id)) continue;
        size_t insert = count;
        while (insert > 0 && context->items[insert - 1].score < candidate.score) {
            if (insert < context->max_items) {
                context->items[insert] = context->items[insert - 1];
            }
            --insert;
        }
        if (insert >= context->max_items) continue;
        context->items[insert] = candidate;
        if (count < context->max_items) ++count;
    }
    if (count > context->top_k) count = context->top_k;
    execution->items = context->items;
    execution->item_count = count;
    execution->item_capacity = context->max_items;
    return MIPK_OK;
}

static mipk_status result_filter_validate(void *context,
    const mipk_mutable_tensor *inputs, size_t input_count,
    const mipk_tensor *outputs, size_t output_count)
{
    (void)inputs;
    (void)input_count;
    (void)outputs;
    (void)output_count;
    return context == NULL ? MIPK_ERR_ARGUMENT : MIPK_OK;
}

static const mipk_capability_descriptor g_capability =
    MIPK_OPERATOR_CAPABILITY("result_filter", "2.0.0");

static const char *const g_options[] = {
    "class_allowlist",
    "score_threshold",
    "top_k",
    "max_items",
};

static const mipk_port_schema g_inputs[] = {
    {
        .port = "results",
        .accepted_value_mask = MIPK_VALUE_MASK(MIPK_VALUE_RESULTS),
        .accepted_domain_mask = MIPK_DOMAIN_MASK(MIPK_DOMAIN_IMAGE),
        .results = {
            .label_set_rule = MIPK_LABEL_SET_HOMOGENEOUS,
        },
        .required = true,
    },
};

static const mipk_node_schema g_schema = {
    .inputs = g_inputs,
    .input_count = sizeof(g_inputs) / sizeof(g_inputs[0]),
    .option_names = g_options,
    .option_count = sizeof(g_options) / sizeof(g_options[0]),
    .output_type = MIPK_VALUE_RESULTS,
    .output_domain = MIPK_DOMAIN_IMAGE,
    .output_results = {
        .label_set_rule = MIPK_LABEL_SET_INHERIT,
        .label_set_from_port = "results",
    },
};

const mipk_operator_ops *mipk_result_filter_operator(void)
{
    static const mipk_runtime_option runtime_options[] = {
        {.struct_size = sizeof(mipk_runtime_option),
            .name = "score_threshold", .type = MIPK_RUNTIME_OPTION_FLOAT,
            .description = "Minimum result score to retain",
            .default_json = "0"},
        {.struct_size = sizeof(mipk_runtime_option), .name = "top_k",
            .type = MIPK_RUNTIME_OPTION_SIZE,
            .description = "Maximum number of highest-score results"},
        {.struct_size = sizeof(mipk_runtime_option),
            .name = "class_allowlist",
            .type = MIPK_RUNTIME_OPTION_STRING_ARRAY,
            .description = "Class names allowed to pass the filter",
            .default_json = "[]"},
    };
    static const mipk_operator_ops operations = {
        .capability = &g_capability,
        .schema = &g_schema,
        .name = "result_filter",
        .create = result_filter_create,
        .destroy = result_filter_destroy,
        .process = result_filter_process,
        .validate = result_filter_validate,
        .runtime_options = runtime_options,
        .runtime_option_count = sizeof(runtime_options) /
            sizeof(runtime_options[0]),
    };
    return &operations;
}

mipk_status mipk_result_filter_register(void)
{
    return mipk_operator_register(mipk_result_filter_operator());
}
