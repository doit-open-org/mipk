#include "mipk_module.h"

#include "cJSON.h"

#include <math.h>
#include <stdlib.h>

typedef struct {
    uint32_t source_width;
    uint32_t source_height;
    size_t max_items;
    float scale;
    float min_size;
    bool square;
} roi_extract_context;

static bool json_size(const cJSON *json, const char *name, size_t *value)
{
    const cJSON *item = cJSON_GetObjectItemCaseSensitive(json, name);
    if (!cJSON_IsNumber(item) || !isfinite(item->valuedouble) ||
        item->valuedouble < 1.0 || item->valuedouble > (double)SIZE_MAX ||
        floor(item->valuedouble) != item->valuedouble) return false;
    *value = (size_t)item->valuedouble;
    return true;
}

static bool json_float(const cJSON *json, const char *name,
    float fallback, float *value)
{
    const cJSON *item = cJSON_GetObjectItemCaseSensitive(json, name);
    if (item == NULL) {
        *value = fallback;
        return true;
    }
    if (!cJSON_IsNumber(item) || !isfinite(item->valuedouble)) return false;
    *value = (float)item->valuedouble;
    return true;
}

static void *roi_extract_create(const mipk_operator_config *config,
    const char *options_json, mipk_status *status)
{
    if (status == NULL) return NULL;
    *status = MIPK_ERR_ARGUMENT;
    if (config == NULL || config->source_width == 0 ||
        config->source_height == 0 || options_json == NULL) return NULL;
    roi_extract_context *context = calloc(1, sizeof(*context));
    cJSON *json = cJSON_Parse(options_json);
    const cJSON *square = json == NULL ? NULL :
        cJSON_GetObjectItemCaseSensitive(json, "square");
    bool valid = context != NULL && json != NULL &&
        json_size(json, "max_items", &context->max_items) &&
        json_float(json, "scale", 1.0F, &context->scale) &&
        json_float(json, "min_size", 1.0F, &context->min_size) &&
        (square == NULL || cJSON_IsBool(square)) &&
        context->scale >= 1.0F && context->min_size > 0.0F;
    if (valid) {
        context->source_width = config->source_width;
        context->source_height = config->source_height;
        context->square = square != NULL && cJSON_IsTrue(square);
    }
    cJSON_Delete(json);
    if (!valid) {
        free(context);
        return NULL;
    }
    *status = MIPK_OK;
    return context;
}

static mipk_status roi_extract_validate(void *private_data,
    const mipk_mutable_tensor *inputs, size_t input_count,
    const mipk_tensor *outputs, size_t output_count)
{
    (void)inputs;
    (void)input_count;
    (void)outputs;
    (void)output_count;
    return private_data == NULL ? MIPK_ERR_ARGUMENT : MIPK_OK;
}

static const mipk_bbox *result_box(const mipk_result *result)
{
    if (result->type == MIPK_RESULT_DETECTION) return &result->value.bbox;
    if (result->type == MIPK_RESULT_TRACK) return &result->value.track.bbox;
    return NULL;
}

static bool expand_box(const roi_extract_context *context,
    const mipk_bbox *input, float box[4])
{
    if (!isfinite(input->x1) || !isfinite(input->y1) ||
        !isfinite(input->x2) || !isfinite(input->y2) ||
        input->x2 <= input->x1 || input->y2 <= input->y1) return false;
    const float center_x = (input->x1 + input->x2) * 0.5F;
    const float center_y = (input->y1 + input->y2) * 0.5F;
    float width = (input->x2 - input->x1) * context->scale;
    float height = (input->y2 - input->y1) * context->scale;
    if (context->square) width = height = fmaxf(width, height);
    box[0] = fmaxf(0.0F, center_x - width * 0.5F);
    box[1] = fmaxf(0.0F, center_y - height * 0.5F);
    box[2] = fminf((float)context->source_width,
        center_x + width * 0.5F);
    box[3] = fminf((float)context->source_height,
        center_y + height * 0.5F);
    return box[2] - box[0] >= context->min_size &&
        box[3] - box[1] >= context->min_size;
}

static mipk_status roi_extract_process(void *private_data,
    mipk_operator_io *execution)
{
    const roi_extract_context *context = private_data;
    if (context == NULL || execution == NULL ||
        execution->stage_output == NULL || execution->collection_items == NULL ||
        execution->collection_capacity < context->max_items) {
        return MIPK_ERR_ARGUMENT;
    }
    const mipk_value *frame = mipk_named_value_find(execution->stage_inputs,
        execution->stage_input_count, "frame");
    const mipk_value *results = mipk_named_value_find(execution->stage_inputs,
        execution->stage_input_count, "results");
    if (frame == NULL || frame->type != MIPK_VALUE_INPUT ||
        frame->value.input == NULL || results == NULL ||
        results->type != MIPK_VALUE_RESULTS) return MIPK_ERR_ARGUMENT;

    size_t count = 0;
    for (size_t index = 0;
        index < results->value.results.item_count; ++index) {
        const mipk_result *result = &results->value.results.items[index];
        const mipk_bbox *bbox = result_box(result);
        float region[4];
        if (bbox == NULL || !expand_box(context, bbox, region)) continue;
        if (count == context->max_items) break;
        uint64_t parent_id = result->result_id;
        if (parent_id == MIPK_INVALID_RESULT_ID) parent_id = index + 1U;
        execution->collection_items[count] = (mipk_value) {
            .type = MIPK_VALUE_IMAGE_VIEW,
            .metadata = {
                .item_id = parent_id,
                .parent_item_id = parent_id,
            },
            .value.image_view = {
                .root_input = frame->value.input,
                .region = {region[0], region[1], region[2], region[3]},
                .point_count = 2,
            },
        };
        ++count;
    }
    *execution->stage_output = (mipk_value) {
        .type = MIPK_VALUE_COLLECTION,
        .value.collection = {
            .items = execution->collection_items,
            .count = count,
            .capacity = execution->collection_capacity,
            .item_type = MIPK_VALUE_IMAGE_VIEW,
        },
    };
    return MIPK_OK;
}

static const mipk_capability_descriptor g_capability =
    MIPK_OPERATOR_CAPABILITY("roi_extract", "2.0.0");

static const char *const g_options[] = {
    "max_items",
    "scale",
    "square",
    "min_size",
};

static const mipk_port_schema g_inputs[] = {
    {
        .port = "frame",
        .accepted_value_mask = MIPK_VALUE_MASK(MIPK_VALUE_INPUT),
        .accepted_domain_mask = MIPK_DOMAIN_MASK(MIPK_DOMAIN_IMAGE),
        .required = true,
    },
    {
        .port = "results",
        .accepted_value_mask = MIPK_VALUE_MASK(MIPK_VALUE_RESULTS),
        .accepted_domain_mask = MIPK_DOMAIN_MASK(MIPK_DOMAIN_IMAGE),
        .results = {
            .result_type_mask = MIPK_RESULT_MASK(MIPK_RESULT_DETECTION) |
                MIPK_RESULT_MASK(MIPK_RESULT_TRACK),
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
    .output_type = MIPK_VALUE_COLLECTION,
    .collection_item_type = MIPK_VALUE_IMAGE_VIEW,
    .output_domain = MIPK_DOMAIN_IMAGE,
};

const mipk_operator_ops *mipk_roi_extract_operator(void)
{
    static const mipk_operator_ops operations = {
        .capability = &g_capability,
        .schema = &g_schema,
        .name = "roi_extract",
        .create = roi_extract_create,
        .destroy = free,
        .process = roi_extract_process,
        .validate = roi_extract_validate,
    };
    return &operations;
}

mipk_status mipk_roi_extract_register(void)
{
    return mipk_operator_register(mipk_roi_extract_operator());
}
