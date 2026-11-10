#include "mipk_detection.h"
#include "mipk_module.h"
#include "mipk_tensor.h"
#include "cJSON.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    MIPK_YOLO_LAYOUT_AUTO = 0,
    MIPK_YOLO_LAYOUT_ATTRIBUTES_FIRST,
    MIPK_YOLO_LAYOUT_CANDIDATES_FIRST,
} mipk_yolo_layout;

typedef struct {
    size_t output_index;
    mipk_yolo_layout layout;
    bool has_objectness;
} mipk_yolo_config;

typedef struct {
    mipk_postprocess_config component;
    mipk_yolo_config yolo;
    float score_threshold;
    size_t max_candidates;
    float *values;
    size_t value_capacity;
    mipk_result *candidates;
    size_t candidate_capacity;
    size_t model_candidate_count;
    bool attributes_first;
    /* Per candidate: order key of the best class score, and that class. */
    uint32_t *best_key;
    uint32_t *best_class;
} yolo_context;

typedef struct {
    const float *values;
    size_t value_count;
    size_t attributes;
    size_t candidates;
    bool attributes_first;
} tensor_view;

static cJSON *postprocess_config_parse(const char *options_json)
{
    if (options_json == NULL) return NULL;
    cJSON *json = cJSON_Parse(options_json);
    if (json == NULL || !cJSON_IsObject(json)) {
        cJSON_Delete(json);
        return NULL;
    }
    return json;
}

static bool config_size(const cJSON *json, const char *name,
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

static bool config_bool(const cJSON *json, const char *name,
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

static const char *config_string(const cJSON *json, const char *name,
    const char *default_value)
{
    const cJSON *item = cJSON_GetObjectItemCaseSensitive(json, name);
    if (item == NULL) return default_value;
    return cJSON_IsString(item) && item->valuestring != NULL ?
        item->valuestring : NULL;
}

static float tensor_at(const tensor_view *view, size_t candidate, size_t attribute)
{
    const size_t index = view->attributes_first ?
        attribute * view->candidates + candidate :
        candidate * view->attributes + attribute;
    return index < view->value_count ? view->values[index] : 0.0F;
}

/*
 * An unsigned integer that orders like the float it came from: compare the
 * keys and you have compared the scores, negative ones included.  The class
 * scan compares every class of every candidate, half a million comparisons
 * for an 80-class head at 640x480; on a soft-float build each float
 * comparison is a library call, so the scan works on these keys instead.
 */
static inline uint32_t score_key(float value)
{
    const union {
        float value;
        uint32_t bits;
    } pun = {.value = value};
    /* NaN never wins: it takes the lowest key. */
    if ((pun.bits & 0x7fffffffU) > 0x7f800000U) return 0;
    return (pun.bits & 0x80000000U) != 0 ? ~pun.bits : pun.bits | 0x80000000U;
}

/*
 * Fills best_key/best_class for every candidate.  With attributes first a
 * class is one contiguous row, so the scan walks row by row keeping a running
 * best per candidate; walking candidate by candidate would stride a whole row
 * between reads.  Ties keep the lowest class, as a strict > always did.
 */
static void find_best_classes(const yolo_context *context, const float *values,
    size_t attributes, size_t class_offset)
{
    const size_t candidates = context->model_candidate_count;
    const size_t classes = context->component.class_count;
    uint32_t *const best_key = context->best_key;
    uint32_t *const best_class = context->best_class;
    if (context->attributes_first) {
        const float *row = values + class_offset * candidates;
        for (size_t candidate = 0; candidate < candidates; ++candidate) {
            best_key[candidate] = score_key(row[candidate]);
            best_class[candidate] = 0;
        }
        for (size_t class_id = 1; class_id < classes; ++class_id) {
            row = values + (class_offset + class_id) * candidates;
            for (size_t candidate = 0; candidate < candidates; ++candidate) {
                const uint32_t key = score_key(row[candidate]);
                if (key > best_key[candidate]) {
                    best_key[candidate] = key;
                    best_class[candidate] = (uint32_t)class_id;
                }
            }
        }
        return;
    }
    for (size_t candidate = 0; candidate < candidates; ++candidate) {
        const float *scores = values + candidate * attributes + class_offset;
        uint32_t key = score_key(scores[0]);
        uint32_t class_id = 0;
        for (size_t index = 1; index < classes; ++index) {
            const uint32_t next = score_key(scores[index]);
            if (next > key) {
                key = next;
                class_id = (uint32_t)index;
            }
        }
        best_key[candidate] = key;
        best_class[candidate] = class_id;
    }
}

static float clamp_value(float value, float minimum, float maximum)
{
    return fminf(fmaxf(value, minimum), maximum);
}

static mipk_bbox bbox_to_source(const mipk_postprocess_config *config,
    mipk_bbox box)
{
    const float x[4] = {box.x1, box.x2, box.x2, box.x1};
    const float y[4] = {box.y1, box.y1, box.y2, box.y2};
    mipk_bbox mapped = {
        .x1 = INFINITY,
        .y1 = INFINITY,
        .x2 = -INFINITY,
        .y2 = -INFINITY,
    };
    for (size_t index = 0; index < 4; ++index) {
        const float denominator = config->to_source[6] * x[index] +
            config->to_source[7] * y[index] + config->to_source[8];
        if (!isfinite(denominator) || fabsf(denominator) < 1.0e-6F) {
            return (mipk_bbox) {0};
        }
        const float source_x = (config->to_source[0] * x[index] +
            config->to_source[1] * y[index] + config->to_source[2]) /
            denominator;
        const float source_y = (config->to_source[3] * x[index] +
            config->to_source[4] * y[index] + config->to_source[5]) /
            denominator;
        mapped.x1 = fminf(mapped.x1, source_x);
        mapped.y1 = fminf(mapped.y1, source_y);
        mapped.x2 = fmaxf(mapped.x2, source_x);
        mapped.y2 = fmaxf(mapped.y2, source_y);
    }
    mapped.x1 = clamp_value(mapped.x1, 0.0F, config->source_width);
    mapped.y1 = clamp_value(mapped.y1, 0.0F, config->source_height);
    mapped.x2 = clamp_value(mapped.x2, 0.0F, config->source_width);
    mapped.y2 = clamp_value(mapped.y2, 0.0F, config->source_height);
    return mapped;
}

static float class_threshold(const yolo_context *context, size_t class_id)
{
    const float threshold = context->component.classes[class_id].score_threshold;
    return threshold < 0.0F ? context->score_threshold : threshold;
}

static mipk_status resolve_shape(const yolo_context *context, const mipk_tensor *tensor,
    size_t value_count, size_t *candidates, bool *attributes_first)
{
    const size_t attributes = 4 + (context->yolo.has_objectness ? 1U : 0U) +
        context->component.class_count;
    int64_t dimensions[MIPK_MAX_TENSOR_DIMS];
    size_t dimension_count = 0;
    for (size_t i = 0; i < tensor->dim_count; ++i) {
        if (tensor->dims[i] > 1) dimensions[dimension_count++] = tensor->dims[i];
    }

    if (context->yolo.layout == MIPK_YOLO_LAYOUT_ATTRIBUTES_FIRST) {
        if (dimension_count < 2 || (size_t)dimensions[dimension_count - 2] != attributes) {
            return MIPK_ERR_MODEL;
        }
        *candidates = (size_t)dimensions[dimension_count - 1];
        *attributes_first = true;
    } else if (context->yolo.layout == MIPK_YOLO_LAYOUT_CANDIDATES_FIRST) {
        if (dimension_count < 2 || (size_t)dimensions[dimension_count - 1] != attributes) {
            return MIPK_ERR_MODEL;
        }
        *candidates = (size_t)dimensions[dimension_count - 2];
        *attributes_first = false;
    } else if (dimension_count >= 2 &&
        (size_t)dimensions[dimension_count - 2] == attributes) {
        *candidates = (size_t)dimensions[dimension_count - 1];
        *attributes_first = true;
    } else if (dimension_count >= 2 &&
        (size_t)dimensions[dimension_count - 1] == attributes) {
        *candidates = (size_t)dimensions[dimension_count - 2];
        *attributes_first = false;
    } else if (attributes != 0 && value_count % attributes == 0) {
        *candidates = value_count / attributes;
        *attributes_first = true;
    } else {
        return MIPK_ERR_MODEL;
    }
    return *candidates != 0 && *candidates <= value_count / attributes ? MIPK_OK : MIPK_ERR_MODEL;
}

static void *yolo_create(const mipk_postprocess_config *config,
    const char *options_json, mipk_status *status)
{
    if (status == NULL) return NULL;
    *status = MIPK_ERR_ARGUMENT;
    if (config == NULL || config->classes == NULL || config->class_count == 0 ||
        config->input_width == 0 || config->input_height == 0 ||
        options_json == NULL) return NULL;

    yolo_context *context = calloc(1, sizeof(*context));
    if (context == NULL) {
        *status = MIPK_ERR_MEMORY;
        return NULL;
    }
    context->component = *config;
    {
        cJSON *json = postprocess_config_parse(options_json);
        const char *layout = config_string(json, "layout", "auto");
        bool valid = json != NULL && layout != NULL &&
            config_size(json, "output_index", 0, &context->yolo.output_index) &&
            config_bool(json, "has_objectness", false,
                &context->yolo.has_objectness) &&
            config_size(json, "max_candidates", 10000,
                &context->max_candidates);
        const cJSON *score = json == NULL ? NULL :
            cJSON_GetObjectItemCaseSensitive(json, "score_threshold");
        valid = valid && (score == NULL || cJSON_IsNumber(score));
        context->score_threshold = score == NULL ? 0.4F :
            (float)score->valuedouble;
        valid = valid &&
            isfinite(context->score_threshold) &&
            context->score_threshold >= 0.0F && context->score_threshold <= 1.0F;
        if (valid && strcmp(layout, "auto") == 0) {
            context->yolo.layout = MIPK_YOLO_LAYOUT_AUTO;
        } else if (valid && strcmp(layout, "attributes_first") == 0) {
            context->yolo.layout = MIPK_YOLO_LAYOUT_ATTRIBUTES_FIRST;
        } else if (valid && strcmp(layout, "candidates_first") == 0) {
            context->yolo.layout = MIPK_YOLO_LAYOUT_CANDIDATES_FIRST;
        } else {
            valid = false;
        }
        cJSON_Delete(json);
        if (!valid) {
            free(context);
            return NULL;
        }
    }
    *status = MIPK_OK;
    return context;
}

static void yolo_destroy(void *context)
{
    yolo_context *yolo = context;
    if (yolo == NULL) return;
    free(yolo->values);
    free(yolo->candidates);
    free(yolo->best_key);
    free(yolo->best_class);
    free(yolo);
}

static mipk_status yolo_validate(void *private_data,
    const mipk_mutable_tensor *inputs, size_t input_count,
    const mipk_tensor *outputs, size_t output_count)
{
    yolo_context *context = private_data;
    if (context == NULL || inputs == NULL || input_count == 0 ||
        mipk_mutable_tensor_data(&inputs[0]) == NULL || outputs == NULL ||
        context->yolo.output_index >= output_count ||
        (context->component.input_size != 0 &&
            inputs[0].byte_size < context->component.input_size)) {
        if (context != NULL && outputs != NULL &&
            context->yolo.output_index >= output_count) {
            mipk_diag("yolo: output_index %zu but the model has %zu outputs",
                context->yolo.output_index, output_count);
        }
        return MIPK_ERR_MODEL;
    }
    const mipk_tensor *output = &outputs[context->yolo.output_index];
    size_t value_count = 0;
    mipk_status status = mipk_tensor_float_count(output, &value_count);
    if (status != MIPK_OK || value_count == 0 ||
        value_count > SIZE_MAX / sizeof(float)) return MIPK_ERR_MODEL;
    size_t candidate_count = 0;
    bool attributes_first = true;
    status = resolve_shape(context, output, value_count, &candidate_count,
        &attributes_first);
    if (status != MIPK_OK) {
        char shape[96];
        mipk_diag("yolo: output %zu is %s but the label set has %zu classes, "
            "so a %zu-attribute layout was expected (4 box%s)",
            context->yolo.output_index,
            mipk_tensor_describe(output, shape, sizeof(shape)),
            context->component.class_count,
            4U + (context->yolo.has_objectness ? 1U : 0U) +
                context->component.class_count,
            context->yolo.has_objectness ? " + objectness + classes" :
                " + classes");
        return status;
    }
    const size_t candidate_capacity = context->max_candidates == 0 ||
        context->max_candidates > candidate_count ? candidate_count :
        context->max_candidates;
    if (candidate_capacity > SIZE_MAX / sizeof(mipk_result)) return MIPK_ERR_MODEL;
    if (candidate_count > SIZE_MAX / sizeof(uint32_t)) return MIPK_ERR_MODEL;
    float *values = malloc(value_count * sizeof(*values));
    mipk_result *candidates = malloc(candidate_capacity * sizeof(*candidates));
    uint32_t *best_key = malloc(candidate_count * sizeof(*best_key));
    uint32_t *best_class = malloc(candidate_count * sizeof(*best_class));
    if (values == NULL || (candidate_capacity != 0 && candidates == NULL) ||
        best_key == NULL || best_class == NULL) {
        free(values);
        free(candidates);
        free(best_key);
        free(best_class);
        return MIPK_ERR_MEMORY;
    }
    free(context->values);
    free(context->candidates);
    free(context->best_key);
    free(context->best_class);
    context->best_key = best_key;
    context->best_class = best_class;
    context->values = values;
    context->value_capacity = value_count;
    context->candidates = candidates;
    context->candidate_capacity = candidate_capacity;
    context->model_candidate_count = candidate_count;
    context->attributes_first = attributes_first;
    return MIPK_OK;
}

static mipk_status yolo_process(void *private_data,
    const mipk_tensor *outputs, size_t output_count,
    mipk_postprocess_output *output)
{
    yolo_context *context = private_data;
    if (context == NULL || outputs == NULL || output == NULL ||
        context->yolo.output_index >= output_count) {
        return MIPK_ERR_ARGUMENT;
    }
    *output = (mipk_postprocess_output) {0};

    size_t value_count = 0;
    mipk_status status = mipk_tensor_to_float_buffer(
        &outputs[context->yolo.output_index], context->values,
        context->value_capacity, &value_count);
    if (status != MIPK_OK) return status;

    const size_t candidate_count = context->model_candidate_count;
    const bool attributes_first = context->attributes_first;

    const size_t attributes = 4 + (context->yolo.has_objectness ? 1U : 0U) +
        context->component.class_count;
    const size_t class_offset = context->yolo.has_objectness ? 5U : 4U;
    size_t retained_count = 0;
    tensor_view view = {context->values, value_count, attributes, candidate_count,
        attributes_first};
    if (candidate_count > value_count / attributes || context->best_key == NULL ||
        context->best_class == NULL) {
        return MIPK_ERR_MODEL;
    }
    find_best_classes(context, context->values, attributes, class_offset);
    for (size_t candidate_index = 0; candidate_index < candidate_count; ++candidate_index) {
        const size_t class_id = context->best_class[candidate_index];
        if (!context->component.classes[class_id].enabled) continue;
        const float class_score = tensor_at(&view, candidate_index,
            class_offset + class_id);
        const float objectness = context->yolo.has_objectness ?
            tensor_at(&view, candidate_index, 4) : 1.0F;
        const float score = objectness * class_score;
        /* Written so a NaN score is rejected too. */
        if (!(score >= class_threshold(context, class_id))) continue;

        const float center_x = tensor_at(&view, candidate_index, 0);
        const float center_y = tensor_at(&view, candidate_index, 1);
        const float width = tensor_at(&view, candidate_index, 2);
        const float height = tensor_at(&view, candidate_index, 3);
        mipk_result item = {
            .type = MIPK_RESULT_DETECTION,
            .class_id = (int32_t)class_id,
            .label_set_id = context->component.label_set_id,
            .track_id = MIPK_INVALID_TRACK_ID,
            .attachment_index = MIPK_INVALID_ATTACHMENT_INDEX,
            .score = score,
            .value.bbox = bbox_to_source(&context->component, (mipk_bbox) {
                clamp_value(center_x - width * 0.5F, 0.0F,
                    (float)context->component.input_width),
                clamp_value(center_y - height * 0.5F, 0.0F,
                    (float)context->component.input_height),
                clamp_value(center_x + width * 0.5F, 0.0F,
                    (float)context->component.input_width),
                clamp_value(center_y + height * 0.5F, 0.0F,
                    (float)context->component.input_height),
            }),
        };
        if (item.value.bbox.x2 > item.value.bbox.x1 &&
            item.value.bbox.y2 > item.value.bbox.y1) {
            mipk_detection_retain(context->candidates, context->candidate_capacity,
                &retained_count, &item);
        }
    }
    output->items = context->candidates;
    output->item_count = retained_count;
    output->item_capacity = context->candidate_capacity;
    return MIPK_OK;
}

static const mipk_capability_descriptor g_capability =
    MIPK_POSTPROCESS_CAPABILITY("yolo", "2.0.0");

static const char *const g_options[] = {
    "output_index",
    "layout",
    "has_objectness",
    "score_threshold",
    "max_candidates",
    "max_results",
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
        .result_type_mask = MIPK_RESULT_MASK(MIPK_RESULT_DETECTION),
        .label_set_rule = MIPK_LABEL_SET_FIXED,
    },
};

const mipk_postprocess_ops *mipk_yolo_postprocess(void)
{
    static const mipk_runtime_option runtime_options[] = {
        {.struct_size = sizeof(mipk_runtime_option),
            .name = "score_threshold", .type = MIPK_RUNTIME_OPTION_FLOAT,
            .description = "Minimum decoded detection score",
            .default_json = "0.4"},
    };
    static const mipk_postprocess_ops operations = {
        .capability = &g_capability,
        .schema = &g_schema,
        .name = "yolo",
        .create = yolo_create,
        .destroy = yolo_destroy,
        .validate = yolo_validate,
        .process = yolo_process,
        .runtime_options = runtime_options,
        .runtime_option_count = sizeof(runtime_options) /
            sizeof(runtime_options[0]),
    };
    return &operations;
}

mipk_status mipk_yolo_register(void)
{
    return mipk_postprocess_register(mipk_yolo_postprocess());
}
