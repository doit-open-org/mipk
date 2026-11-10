#include "mipk_module.h"

#include "cJSON.h"

#include <math.h>
#include <stdlib.h>

typedef struct {
    uint32_t output_width;
    uint32_t output_height;
    int32_t point_ids[3];
    float target_points[6];
    float min_score;
} image_align_context;

static bool json_u32(const cJSON *json, const char *name, uint32_t *value)
{
    const cJSON *item = cJSON_GetObjectItemCaseSensitive(json, name);
    if (!cJSON_IsNumber(item) || !isfinite(item->valuedouble) ||
        item->valuedouble < 1.0 || item->valuedouble > UINT32_MAX ||
        floor(item->valuedouble) != item->valuedouble) return false;
    *value = (uint32_t)item->valuedouble;
    return true;
}

static bool parse_point_ids(const cJSON *json, int32_t ids[3])
{
    const cJSON *items = cJSON_GetObjectItemCaseSensitive(json, "point_ids");
    if (!cJSON_IsArray(items) || cJSON_GetArraySize(items) != 3) return false;
    for (int index = 0; index < 3; ++index) {
        const cJSON *item = cJSON_GetArrayItem(items, index);
        if (!cJSON_IsNumber(item) || !isfinite(item->valuedouble) ||
            item->valuedouble < 0.0 || item->valuedouble > INT32_MAX ||
            floor(item->valuedouble) != item->valuedouble) return false;
        ids[index] = (int32_t)item->valuedouble;
        for (int previous = 0; previous < index; ++previous) {
            if (ids[previous] == ids[index]) return false;
        }
    }
    return true;
}

static bool parse_target_points(const cJSON *json, float points[6])
{
    const cJSON *items = cJSON_GetObjectItemCaseSensitive(json,
        "target_points");
    if (!cJSON_IsArray(items) || cJSON_GetArraySize(items) != 3) return false;
    for (int index = 0; index < 3; ++index) {
        const cJSON *point = cJSON_GetArrayItem(items, index);
        if (!cJSON_IsArray(point) || cJSON_GetArraySize(point) != 2) {
            return false;
        }
        for (int axis = 0; axis < 2; ++axis) {
            const cJSON *coordinate = cJSON_GetArrayItem(point, axis);
            if (!cJSON_IsNumber(coordinate) ||
                !isfinite(coordinate->valuedouble)) return false;
            points[index * 2 + axis] = (float)coordinate->valuedouble;
        }
    }
    return true;
}

static void *image_align_create(const mipk_operator_config *config,
    const char *options_json, mipk_status *status)
{
    (void)config;
    if (status == NULL) return NULL;
    *status = MIPK_ERR_ARGUMENT;
    if (options_json == NULL) return NULL;
    image_align_context *context = calloc(1, sizeof(*context));
    cJSON *json = cJSON_Parse(options_json);
    const cJSON *score = json == NULL ? NULL :
        cJSON_GetObjectItemCaseSensitive(json, "min_score");
    bool valid = context != NULL && json != NULL &&
        json_u32(json, "output_width", &context->output_width) &&
        json_u32(json, "output_height", &context->output_height) &&
        parse_point_ids(json, context->point_ids) &&
        parse_target_points(json, context->target_points) &&
        (score == NULL || (cJSON_IsNumber(score) &&
            isfinite(score->valuedouble) && score->valuedouble >= 0.0 &&
            score->valuedouble <= 1.0));
    context->min_score = score == NULL ? 0.0F : (float)score->valuedouble;
    for (size_t index = 0; valid && index < 3; ++index) {
        const float x = context->target_points[index * 2];
        const float y = context->target_points[index * 2 + 1];
        valid = x >= 0.0F && x <= context->output_width &&
            y >= 0.0F && y <= context->output_height;
    }
    cJSON_Delete(json);
    if (!valid) {
        free(context);
        return NULL;
    }
    *status = MIPK_OK;
    return context;
}

static mipk_status image_align_validate(void *private_data,
    const mipk_mutable_tensor *inputs, size_t input_count,
    const mipk_tensor *outputs, size_t output_count)
{
    (void)inputs;
    (void)input_count;
    (void)outputs;
    (void)output_count;
    return private_data == NULL ? MIPK_ERR_ARGUMENT : MIPK_OK;
}

static const mipk_result *find_keypoint(const mipk_value *value, int32_t id,
    float minimum_score)
{
    for (size_t index = 0; index < value->value.results.item_count; ++index) {
        const mipk_result *result = &value->value.results.items[index];
        if (result->type == MIPK_RESULT_KEYPOINT &&
            result->value.keypoint.keypoint_id == id &&
            result->score >= minimum_score) return result;
    }
    return NULL;
}

static bool solve_affine(const float destination[6], const float source[6],
    float matrix[6])
{
    const double x0 = destination[0], y0 = destination[1];
    const double x1 = destination[2], y1 = destination[3];
    const double x2 = destination[4], y2 = destination[5];
    const double determinant = x0 * (y1 - y2) - y0 * (x1 - x2) +
        x1 * y2 - x2 * y1;
    if (!isfinite(determinant) || fabs(determinant) < 1.0e-9) return false;
    const double inverse[9] = {
        (y1 - y2) / determinant, (x2 - x1) / determinant,
            (x1 * y2 - x2 * y1) / determinant,
        (y2 - y0) / determinant, (x0 - x2) / determinant,
            (x2 * y0 - x0 * y2) / determinant,
        (y0 - y1) / determinant, (x1 - x0) / determinant,
            (x0 * y1 - x1 * y0) / determinant,
    };
    for (size_t axis = 0; axis < 2; ++axis) {
        for (size_t column = 0; column < 3; ++column) {
            double value = 0.0;
            for (size_t row = 0; row < 3; ++row) {
                value += inverse[row * 3 + column] * source[row * 2 + axis];
            }
            if (!isfinite(value)) return false;
            matrix[axis * 3 + column] = (float)value;
        }
    }
    return true;
}

static mipk_status image_align_process(void *private_data,
    mipk_operator_io *execution)
{
    const image_align_context *context = private_data;
    if (context == NULL || execution == NULL ||
        execution->stage_output == NULL) return MIPK_ERR_ARGUMENT;
    const mipk_value *image = mipk_named_value_find(execution->stage_inputs,
        execution->stage_input_count, "image");
    const mipk_value *landmarks = mipk_named_value_find(
        execution->stage_inputs, execution->stage_input_count,
        "landmarks");
    if (image == NULL || image->type != MIPK_VALUE_IMAGE_VIEW ||
        image->value.image_view.root_input == NULL || landmarks == NULL ||
        landmarks->type != MIPK_VALUE_RESULTS) return MIPK_ERR_ARGUMENT;
    float source[6];
    for (size_t index = 0; index < 3; ++index) {
        const mipk_result *point = find_keypoint(landmarks,
            context->point_ids[index], context->min_score);
        if (point == NULL) return MIPK_ERR_OPERATOR;
        source[index * 2] = point->value.keypoint.x;
        source[index * 2 + 1] = point->value.keypoint.y;
    }
    float affine[6];
    if (!solve_affine(context->target_points, source, affine)) {
        return MIPK_ERR_OPERATOR;
    }
    const float corners[8] = {
        0.0F, 0.0F, (float)context->output_width, 0.0F,
        (float)context->output_width, (float)context->output_height,
        0.0F, (float)context->output_height,
    };
    mipk_value output = {
        .type = MIPK_VALUE_IMAGE_VIEW,
        .metadata = image->metadata,
        .value.image_view = {
            .root_input = image->value.image_view.root_input,
            .point_count = 4,
        },
    };
    for (size_t index = 0; index < 4; ++index) {
        const float x = corners[index * 2];
        const float y = corners[index * 2 + 1];
        output.value.image_view.region[index * 2] =
            affine[0] * x + affine[1] * y + affine[2];
        output.value.image_view.region[index * 2 + 1] =
            affine[3] * x + affine[4] * y + affine[5];
        if (!isfinite(output.value.image_view.region[index * 2]) ||
            !isfinite(output.value.image_view.region[index * 2 + 1])) {
            return MIPK_ERR_OPERATOR;
        }
    }
    *execution->stage_output = output;
    return MIPK_OK;
}

static const mipk_capability_descriptor g_capability =
    MIPK_OPERATOR_CAPABILITY("image_align", "2.0.0");

static const char *const g_options[] = {
    "max_items",
    "output_width",
    "output_height",
    "point_ids",
    "target_points",
    "min_score",
};

static const mipk_port_schema g_inputs[] = {
    {
        .port = "image",
        .accepted_value_mask = MIPK_VALUE_MASK(MIPK_VALUE_IMAGE_VIEW),
        .accepted_domain_mask = MIPK_DOMAIN_MASK(MIPK_DOMAIN_IMAGE),
        .required = true,
    },
    {
        .port = "landmarks",
        .accepted_value_mask = MIPK_VALUE_MASK(MIPK_VALUE_RESULTS),
        .accepted_domain_mask = MIPK_DOMAIN_MASK(MIPK_DOMAIN_IMAGE),
        .results = {
            .result_type_mask = MIPK_RESULT_MASK(MIPK_RESULT_KEYPOINT),
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
    .output_type = MIPK_VALUE_IMAGE_VIEW,
    .output_domain = MIPK_DOMAIN_IMAGE,
};

const mipk_operator_ops *mipk_image_align_operator(void)
{
    static const mipk_operator_ops operations = {
        .capability = &g_capability,
        .schema = &g_schema,
        .name = "image_align",
        .create = image_align_create,
        .destroy = free,
        .process = image_align_process,
        .validate = image_align_validate,
    };
    return &operations;
}

mipk_status mipk_image_align_register(void)
{
    return mipk_operator_register(mipk_image_align_operator());
}
