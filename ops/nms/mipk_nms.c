#include "mipk_module.h"
#include "mipk_nms.h"

#include "mipk_detection.h"
#include "cJSON.h"

#include <math.h>
#include <stdlib.h>

typedef struct {
    float threshold;
    bool class_agnostic;
    mipk_result *selected;
    size_t capacity;
} nms_context;

static void nms_destroy(void *private_data)
{
    nms_context *context = private_data;
    if (context == NULL) return;
    free(context->selected);
    free(context);
}

static void *nms_create(const mipk_operator_config *config,
    const char *options_json, mipk_status *status)
{
    if (status == NULL) return NULL;
    *status = MIPK_ERR_ARGUMENT;
    if (config == NULL || config->result_capacity == 0 || options_json == NULL) {
        return NULL;
    }
    cJSON *json = cJSON_Parse(options_json);
    const cJSON *threshold = json == NULL ? NULL :
        cJSON_GetObjectItemCaseSensitive(json, "iou_threshold");
    const cJSON *class_agnostic = json == NULL ? NULL :
        cJSON_GetObjectItemCaseSensitive(json, "class_agnostic");
    const cJSON *max_results = json == NULL ? NULL :
        cJSON_GetObjectItemCaseSensitive(json, "max_results");
    bool valid = json != NULL &&
        (threshold == NULL || (cJSON_IsNumber(threshold) &&
            isfinite(threshold->valuedouble) && threshold->valuedouble >= 0.0 &&
            threshold->valuedouble <= 1.0)) &&
        (class_agnostic == NULL || cJSON_IsBool(class_agnostic));
    valid = valid && cJSON_IsNumber(max_results) &&
        isfinite(max_results->valuedouble) && max_results->valuedouble >= 1.0 &&
        max_results->valuedouble <= (double)config->result_capacity &&
        floor(max_results->valuedouble) == max_results->valuedouble;
    nms_context *context = valid ? calloc(1, sizeof(*context)) : NULL;
    if (context != NULL) {
        context->threshold = threshold == NULL ? 0.45F :
            (float)threshold->valuedouble;
        context->class_agnostic = class_agnostic != NULL &&
            cJSON_IsTrue(class_agnostic);
        context->capacity = (size_t)max_results->valuedouble;
        context->selected = calloc(context->capacity, sizeof(*context->selected));
        if (context->selected == NULL) {
            nms_destroy(context);
            context = NULL;
            *status = MIPK_ERR_MEMORY;
        } else {
            *status = MIPK_OK;
        }
    } else if (valid) {
        *status = MIPK_ERR_MEMORY;
    }
    cJSON_Delete(json);
    return context;
}

static mipk_status nms_validate(void *private_data,
    const mipk_mutable_tensor *inputs, size_t input_count,
    const mipk_tensor *outputs, size_t output_count)
{
    (void)inputs;
    (void)input_count;
    (void)outputs;
    (void)output_count;
    return private_data == NULL ? MIPK_ERR_OPERATOR : MIPK_OK;
}

static mipk_status nms_process(void *private_data,
    mipk_operator_io *execution)
{
    nms_context *context = private_data;
    if (context == NULL || execution == NULL ||
        (execution->item_count != 0 && execution->items == NULL)) {
        return MIPK_ERR_ARGUMENT;
    }
    mipk_result_buffer selected = {
        .items = context->selected,
        .capacity = context->capacity,
    };
    mipk_status status = mipk_detection_nms(execution->items,
        execution->item_count, true, context->class_agnostic,
        context->threshold, &selected);
    if (status != MIPK_OK) return status;
    execution->items = context->selected;
    execution->item_count = selected.count;
    execution->item_capacity = context->capacity;
    return MIPK_OK;
}

static const mipk_capability_descriptor g_capability =
    MIPK_OPERATOR_CAPABILITY("nms", "2.0.0");

static const char *const g_options[] = {
    "iou_threshold",
    "class_agnostic",
    "max_results",
};

static const mipk_port_schema g_inputs[] = {
    {
        .port = "results",
        .accepted_value_mask = MIPK_VALUE_MASK(MIPK_VALUE_RESULTS),
        .accepted_domain_mask = MIPK_DOMAIN_MASK(MIPK_DOMAIN_IMAGE),
        .results = {
            .result_type_mask = MIPK_RESULT_MASK(MIPK_RESULT_DETECTION),
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

const mipk_operator_ops *mipk_nms_operator(void)
{
    static const mipk_runtime_option runtime_options[] = {
        {.struct_size = sizeof(mipk_runtime_option), .name = "iou_threshold",
            .type = MIPK_RUNTIME_OPTION_FLOAT,
            .description = "IoU threshold used to suppress overlapping boxes",
            .default_json = "0.45"},
        {.struct_size = sizeof(mipk_runtime_option), .name = "class_agnostic",
            .type = MIPK_RUNTIME_OPTION_BOOL,
            .description = "Suppress overlapping boxes across classes",
            .default_json = "false"},
    };
    static const mipk_operator_ops operations = {
        .capability = &g_capability,
        .schema = &g_schema,
        .name = "nms",
        .create = nms_create,
        .destroy = nms_destroy,
        .process = nms_process,
        .validate = nms_validate,
        .runtime_options = runtime_options,
        .runtime_option_count = sizeof(runtime_options) /
            sizeof(runtime_options[0]),
    };
    return &operations;
}

mipk_status mipk_nms_register(void)
{
    return mipk_operator_register(mipk_nms_operator());
}

mipk_status mipk_pipeline_update_nms(mipk_pipeline *pipeline,
    const char *node_id, const mipk_nms_runtime_options *options)
{
    if (pipeline == NULL || node_id == NULL || options == NULL ||
        options->struct_size != sizeof(*options) || options->field_mask == 0 ||
        (options->field_mask & ~MIPK_NMS_UPDATE_ALL) != 0) {
        return MIPK_ERR_ARGUMENT;
    }
    cJSON *json = cJSON_CreateObject();
    mipk_status status = json == NULL ? MIPK_ERR_MEMORY : MIPK_OK;
    if (status == MIPK_OK &&
        (options->field_mask & MIPK_NMS_UPDATE_IOU_THRESHOLD) != 0 &&
        cJSON_AddNumberToObject(json, "iou_threshold",
            options->iou_threshold) == NULL) status = MIPK_ERR_MEMORY;
    if (status == MIPK_OK &&
        (options->field_mask & MIPK_NMS_UPDATE_CLASS_AGNOSTIC) != 0 &&
        cJSON_AddBoolToObject(json, "class_agnostic",
            options->class_agnostic) == NULL) status = MIPK_ERR_MEMORY;
    char *options_json = status == MIPK_OK ? cJSON_PrintUnformatted(json) : NULL;
    if (status == MIPK_OK && options_json == NULL) status = MIPK_ERR_MEMORY;
    if (status == MIPK_OK) {
        status = mipk_pipeline_update_stage_options(pipeline, node_id,
            MIPK_STAGE_OPERATOR, "nms", options_json);
    }
    free(options_json);
    cJSON_Delete(json);
    return status;
}
