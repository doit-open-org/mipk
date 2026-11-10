#include "mipk_module.h"
#include "mipk_tensor.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>


/*
 * Decodes the raw output of an Ultralytics YOLO pose head.
 *
 * Exported without its end-to-end NMS, the head emits one row per anchor:
 *
 *   cx, cy, w, h, class scores..., then x, y, visibility for each keypoint
 *
 * Every term is already decoded by the export -- box and keypoint coordinates
 * in model input pixels, class scores and keypoint visibility through their
 * sigmoid -- so the only work left is choosing the anchors worth keeping and
 * mapping them back to the source frame.
 *
 * Suppression happens here rather than in the shared `nms` operator because
 * the keypoints belong to the box that survives: the shared operator knows
 * only boxes, and would keep each box without the 24 points that go with it.
 * Each surviving object is reported as one detection followed by its
 * keypoints, which carry the detection's id as their parent so a caller can
 * put them back together.
 */

/* x, y and visibility per keypoint. */
#define POSE_VALUES_PER_KEYPOINT 3U
/* A keypoint count beyond anything a pose dataset uses is a manifest error. */
#define POSE_MAX_KEYPOINTS 256U

typedef struct {
    float score;
    int32_t class_id;
    /* Source-frame corners, x1 y1 x2 y2. */
    float box[4];
} pose_candidate;

typedef struct {
    mipk_postprocess_config config;
    size_t output_index;
    size_t keypoint_count;
    float score_threshold;
    float iou_threshold;
    size_t max_results;
    size_t max_candidates;

    /* Resolved from the model output in validate. */
    size_t attributes;
    size_t anchors;
    bool attributes_first;
    /*
     * The class scores of every anchor, and nothing else.  Almost no anchor
     * survives the threshold, so the rest of the output is read a value at a
     * time for the few that do; converting all of it would copy the whole
     * tensor out of memory the CPU does not cache.
     */
    float *scores;
    /* Only for the frame being decoded, so value_at can reach the tensor. */
    const mipk_tensor *output;

    pose_candidate *candidates;
    /* Keypoints of each candidate, x y visibility, in source pixels. */
    float *keypoints;
    size_t candidate_capacity;
    size_t *order;
    mipk_result *results;
    size_t result_capacity;
} pose_context;

static float clamp_value(float value, float minimum, float maximum)
{
    return fminf(fmaxf(value, minimum), maximum);
}

static bool to_source(const float matrix[9], float x, float y,
    float *source_x, float *source_y)
{
    const float denominator = matrix[6] * x + matrix[7] * y + matrix[8];
    if (!isfinite(denominator) || fabsf(denominator) < 1.0e-6F) return false;
    const float mapped_x = (matrix[0] * x + matrix[1] * y + matrix[2]) /
        denominator;
    const float mapped_y = (matrix[3] * x + matrix[4] * y + matrix[5]) /
        denominator;
    if (!isfinite(mapped_x) || !isfinite(mapped_y)) return false;
    *source_x = mapped_x;
    *source_y = mapped_y;
    return true;
}

/*
 * One attribute of one anchor, straight from the output tensor.
 *
 * Which axis is the row depends on how the model lays its output out, and
 * both shapes occur: the exports in this tree put the attributes first, but a
 * model that puts the candidates first is equally valid and is decoded the
 * same way from here.
 */
static float value_at(const pose_context *context, size_t anchor,
    size_t attribute)
{
    float value = 0.0F;
    const size_t row = context->attributes_first ? attribute : anchor;
    const size_t column = context->attributes_first ? anchor : attribute;
    if (mipk_tensor_float_at(context->output, row, column, &value) != MIPK_OK) {
        return NAN;
    }
    return value;
}

/* The class scores of every anchor, which is what decides the rest. */
static mipk_status read_scores(pose_context *context)
{
    const size_t classes = context->config.class_count;
    if (context->attributes_first) {
        /* A class is one row, so each comes across in a single burst. */
        for (size_t class_index = 0; class_index < classes; ++class_index) {
            const mipk_status status = mipk_tensor_row_to_float(
                context->output, 4U + class_index,
                context->scores + class_index * context->anchors,
                context->anchors);
            if (status != MIPK_OK) return status;
        }
        return MIPK_OK;
    }
    for (size_t anchor = 0; anchor < context->anchors; ++anchor) {
        for (size_t class_index = 0; class_index < classes; ++class_index) {
            float value = 0.0F;
            const mipk_status status = mipk_tensor_float_at(context->output,
                anchor, 4U + class_index, &value);
            if (status != MIPK_OK) return status;
            context->scores[class_index * context->anchors + anchor] = value;
        }
    }
    return MIPK_OK;
}

static float score_at(const pose_context *context, size_t anchor,
    size_t class_index)
{
    return context->scores[class_index * context->anchors + anchor];
}

static void pose_destroy(void *private_data)
{
    pose_context *context = private_data;
    if (context == NULL) return;
    free(context->scores);
    free(context->candidates);
    free(context->keypoints);
    free(context->order);
    free(context->results);
    free(context);
}

static void *pose_create(const mipk_postprocess_config *config,
    const char *options_json, mipk_status *status)
{
    if (status == NULL) return NULL;
    *status = MIPK_ERR_ARGUMENT;
    if (config == NULL || config->classes == NULL || config->class_count == 0 ||
        config->input_width == 0 || config->input_height == 0 ||
        config->result_capacity == 0 || options_json == NULL) return NULL;
    pose_context *context = calloc(1, sizeof(*context));
    if (context == NULL) {
        *status = MIPK_ERR_MEMORY;
        return NULL;
    }
    context->config = *config;
    mipk_status parsed = mipk_postprocess_option_get_size(options_json,
        "output_index", 0U, &context->output_index);
    if (parsed == MIPK_OK) {
        parsed = mipk_postprocess_option_get_size(options_json,
            "keypoint_count", 17U, &context->keypoint_count);
    }
    if (parsed == MIPK_OK) {
        parsed = mipk_postprocess_option_get_float(options_json,
            "score_threshold", 0.5F, &context->score_threshold);
    }
    if (parsed == MIPK_OK) {
        parsed = mipk_postprocess_option_get_float(options_json,
            "iou_threshold", 0.45F, &context->iou_threshold);
    }
    if (parsed == MIPK_OK) {
        parsed = mipk_postprocess_option_get_size(options_json, "max_results",
            4U, &context->max_results);
    }
    if (parsed == MIPK_OK) {
        parsed = mipk_postprocess_option_get_size(options_json,
            "max_candidates", 100U, &context->max_candidates);
    }
    const bool valid = parsed == MIPK_OK &&
        context->keypoint_count != 0 &&
        context->keypoint_count <= POSE_MAX_KEYPOINTS &&
        isfinite(context->score_threshold) &&
        context->score_threshold >= 0.0F && context->score_threshold <= 1.0F &&
        isfinite(context->iou_threshold) && context->iou_threshold >= 0.0F &&
        context->iou_threshold <= 1.0F && context->max_results != 0 &&
        context->max_candidates >= context->max_results;
    if (!valid) {
        *status = parsed == MIPK_OK ? MIPK_ERR_MODEL : parsed;
        pose_destroy(context);
        return NULL;
    }

    /* One detection plus its keypoints per reported object. */
    const size_t per_object = 1U + context->keypoint_count;
    if (context->max_results > SIZE_MAX / per_object) {
        *status = MIPK_ERR_MODEL;
        pose_destroy(context);
        return NULL;
    }
    context->result_capacity = context->max_results * per_object;
    if (context->result_capacity > config->result_capacity) {
        *status = MIPK_ERR_CAPACITY;
        pose_destroy(context);
        return NULL;
    }
    const size_t keypoint_values =
        context->keypoint_count * POSE_VALUES_PER_KEYPOINT;
    context->candidate_capacity = context->max_candidates;
    if (context->candidate_capacity > SIZE_MAX / keypoint_values /
        sizeof(float)) {
        *status = MIPK_ERR_MODEL;
        pose_destroy(context);
        return NULL;
    }
    context->candidates = malloc(context->candidate_capacity *
        sizeof(*context->candidates));
    context->keypoints = malloc(context->candidate_capacity *
        keypoint_values * sizeof(*context->keypoints));
    context->order = malloc(context->candidate_capacity *
        sizeof(*context->order));
    context->results = malloc(context->result_capacity *
        sizeof(*context->results));
    if (context->candidates == NULL || context->keypoints == NULL ||
        context->order == NULL || context->results == NULL) {
        *status = MIPK_ERR_MEMORY;
        pose_destroy(context);
        return NULL;
    }
    *status = MIPK_OK;
    return context;
}

/*
 * Works out which way round the output is stored.  The attribute count is
 * known from the manifest, so whichever of the last two dimensions matches it
 * is the attribute axis.  A mismatch means the model is not the one the
 * manifest describes -- the wrong keypoint count or class count -- and is
 * refused rather than decoded as garbage.
 */
static mipk_status pose_validate(void *private_data,
    const mipk_mutable_tensor *inputs, size_t input_count,
    const mipk_tensor *outputs, size_t output_count)
{
    pose_context *context = private_data;
    (void)inputs;
    (void)input_count;
    if (context == NULL || outputs == NULL ||
        context->output_index >= output_count) return MIPK_ERR_MODEL;
    const mipk_tensor *output = &outputs[context->output_index];
    size_t value_count = 0;
    if (mipk_tensor_float_count(output, &value_count) != MIPK_OK ||
        value_count == 0 || value_count > SIZE_MAX / sizeof(float)) {
        return MIPK_ERR_MODEL;
    }
    const size_t attributes = 4U + context->config.class_count +
        context->keypoint_count * POSE_VALUES_PER_KEYPOINT;

    int64_t dimensions[MIPK_MAX_TENSOR_DIMS];
    size_t dimension_count = 0;
    for (size_t index = 0; index < output->dim_count; ++index) {
        if (output->dims[index] > 1) {
            dimensions[dimension_count++] = output->dims[index];
        }
    }
    if (dimension_count < 2) return MIPK_ERR_MODEL;
    const size_t second_last = (size_t)dimensions[dimension_count - 2];
    const size_t last = (size_t)dimensions[dimension_count - 1];
    if (second_last == attributes) {
        context->attributes_first = true;
        context->anchors = last;
    } else if (last == attributes) {
        context->attributes_first = false;
        context->anchors = second_last;
    } else {
        char shape[96];
        mipk_diag("yolo_pose: output %zu is %s; expected %zu attributes per "
            "anchor (4 box + %zu classes + %zu keypoints x 3)",
            context->output_index,
            mipk_tensor_describe(output, shape, sizeof(shape)), attributes,
            context->config.class_count, context->keypoint_count);
        return MIPK_ERR_MODEL;
    }
    if (context->anchors == 0 ||
        context->anchors > value_count / attributes) return MIPK_ERR_MODEL;

    if (context->config.class_count == 0 ||
        context->anchors > SIZE_MAX / context->config.class_count) {
        return MIPK_ERR_MODEL;
    }
    const size_t score_count = context->anchors * context->config.class_count;
    float *scores = malloc(score_count * sizeof(*scores));
    if (scores == NULL) return MIPK_ERR_MEMORY;
    free(context->scores);
    context->scores = scores;
    context->attributes = attributes;
    return MIPK_OK;
}

/*
 * Keeps the highest scoring candidates when more pass the threshold than
 * there is room for.  Returns the slot the candidate went into, or SIZE_MAX
 * if it was not good enough to displace anything.
 */
static size_t insert_candidate(pose_context *context, size_t *count,
    const pose_candidate *candidate)
{
    if (*count < context->candidate_capacity) {
        context->candidates[*count] = *candidate;
        return (*count)++;
    }
    size_t lowest = 0;
    for (size_t index = 1; index < *count; ++index) {
        if (context->candidates[index].score <
            context->candidates[lowest].score) lowest = index;
    }
    if (candidate->score <= context->candidates[lowest].score) return SIZE_MAX;
    context->candidates[lowest] = *candidate;
    return lowest;
}

static float box_iou(const float first[4], const float second[4])
{
    const float left = fmaxf(first[0], second[0]);
    const float top = fmaxf(first[1], second[1]);
    const float right = fminf(first[2], second[2]);
    const float bottom = fminf(first[3], second[3]);
    const float width = right - left;
    const float height = bottom - top;
    if (width <= 0.0F || height <= 0.0F) return 0.0F;
    const float intersection = width * height;
    const float first_area = (first[2] - first[0]) * (first[3] - first[1]);
    const float second_area = (second[2] - second[0]) * (second[3] - second[1]);
    const float union_area = first_area + second_area - intersection;
    return union_area <= 0.0F ? 0.0F : intersection / union_area;
}

static void sort_by_score(pose_context *context, size_t count)
{
    for (size_t index = 0; index < count; ++index) {
        context->order[index] = index;
    }
    /* Insertion sort: the candidate list is bounded by max_candidates. */
    for (size_t index = 1; index < count; ++index) {
        const size_t current = context->order[index];
        const float score = context->candidates[current].score;
        size_t position = index;
        while (position > 0 &&
            context->candidates[context->order[position - 1]].score < score) {
            context->order[position] = context->order[position - 1];
            --position;
        }
        context->order[position] = current;
    }
}

/* Decodes one anchor's box, or returns false if it is not a usable box. */
static bool decode_box(const pose_context *context, size_t anchor,
    float box[4])
{
    const float center_x = value_at(context, anchor, 0);
    const float center_y = value_at(context, anchor, 1);
    const float width = value_at(context, anchor, 2);
    const float height = value_at(context, anchor, 3);
    if (!isfinite(center_x) || !isfinite(center_y) || !isfinite(width) ||
        !isfinite(height) || width <= 0.0F || height <= 0.0F) return false;
    const float input_width = (float)context->config.input_width;
    const float input_height = (float)context->config.input_height;
    /*
     * Map all four corners, not two: the transform is a letterbox today, but
     * nothing stops it being a projective one, and a projective map does not
     * keep a box's opposite corners opposite.
     */
    const float x[4] = {
        clamp_value(center_x - width * 0.5F, 0.0F, input_width),
        clamp_value(center_x + width * 0.5F, 0.0F, input_width),
        clamp_value(center_x + width * 0.5F, 0.0F, input_width),
        clamp_value(center_x - width * 0.5F, 0.0F, input_width),
    };
    const float y[4] = {
        clamp_value(center_y - height * 0.5F, 0.0F, input_height),
        clamp_value(center_y - height * 0.5F, 0.0F, input_height),
        clamp_value(center_y + height * 0.5F, 0.0F, input_height),
        clamp_value(center_y + height * 0.5F, 0.0F, input_height),
    };
    box[0] = INFINITY;
    box[1] = INFINITY;
    box[2] = -INFINITY;
    box[3] = -INFINITY;
    for (size_t corner = 0; corner < 4; ++corner) {
        float source_x = 0.0F;
        float source_y = 0.0F;
        if (!to_source(context->config.to_source, x[corner], y[corner],
            &source_x, &source_y)) return false;
        box[0] = fminf(box[0], source_x);
        box[1] = fminf(box[1], source_y);
        box[2] = fmaxf(box[2], source_x);
        box[3] = fmaxf(box[3], source_y);
    }
    const float source_width = (float)context->config.source_width;
    const float source_height = (float)context->config.source_height;
    box[0] = clamp_value(box[0], 0.0F, source_width);
    box[1] = clamp_value(box[1], 0.0F, source_height);
    box[2] = clamp_value(box[2], 0.0F, source_width);
    box[3] = clamp_value(box[3], 0.0F, source_height);
    return box[2] > box[0] && box[3] > box[1];
}

/* Decodes one anchor's keypoints into source pixels. */
static bool decode_keypoints(const pose_context *context, size_t anchor,
    float *keypoints)
{
    const size_t first = 4U + context->config.class_count;
    const float source_width = (float)context->config.source_width;
    const float source_height = (float)context->config.source_height;
    for (size_t point = 0; point < context->keypoint_count; ++point) {
        const size_t base = first + point * POSE_VALUES_PER_KEYPOINT;
        float x = 0.0F;
        float y = 0.0F;
        if (!to_source(context->config.to_source, value_at(context, anchor, base),
            value_at(context, anchor, base + 1U), &x, &y)) return false;
        const float visibility = value_at(context, anchor, base + 2U);
        float *out = keypoints + point * POSE_VALUES_PER_KEYPOINT;
        out[0] = clamp_value(x, 0.0F, source_width);
        out[1] = clamp_value(y, 0.0F, source_height);
        out[2] = isfinite(visibility) ?
            clamp_value(visibility, 0.0F, 1.0F) : 0.0F;
    }
    return true;
}

static mipk_status pose_process(void *private_data,
    const mipk_tensor *outputs, size_t output_count,
    mipk_postprocess_output *output)
{
    pose_context *context = private_data;
    if (context == NULL || outputs == NULL || output == NULL ||
        context->output_index >= output_count || context->scores == NULL) {
        return MIPK_ERR_ARGUMENT;
    }
    *output = (mipk_postprocess_output) {0};
    context->output = &outputs[context->output_index];
    const mipk_status status = read_scores(context);
    if (status != MIPK_OK) return status;

    const size_t keypoint_values =
        context->keypoint_count * POSE_VALUES_PER_KEYPOINT;
    size_t candidate_count = 0;
    for (size_t anchor = 0; anchor < context->anchors; ++anchor) {
        int32_t class_id = 0;
        float score = score_at(context, anchor, 0);
        for (size_t class_index = 1; class_index < context->config.class_count;
            ++class_index) {
            const float class_score = score_at(context, anchor, class_index);
            if (class_score > score) {
                score = class_score;
                class_id = (int32_t)class_index;
            }
        }
        if (!isfinite(score) || score < context->score_threshold ||
            !context->config.classes[class_id].enabled) continue;
        pose_candidate candidate = {.score = score, .class_id = class_id};
        if (!decode_box(context, anchor, candidate.box)) continue;
        const size_t slot = insert_candidate(context, &candidate_count,
            &candidate);
        if (slot == SIZE_MAX) continue;
        /* Decoded only for anchors that made it in, which are few. */
        if (!decode_keypoints(context, anchor,
            context->keypoints + slot * keypoint_values)) {
            context->candidates[slot].score = -1.0F;
        }
    }

    sort_by_score(context, candidate_count);
    size_t kept = 0;
    size_t result_count = 0;
    for (size_t index = 0; index < candidate_count &&
        kept < context->max_results; ++index) {
        const size_t slot = context->order[index];
        const pose_candidate *candidate = &context->candidates[slot];
        if (candidate->score < 0.0F) continue;
        bool suppressed = false;
        for (size_t previous = 0; previous < index && !suppressed; ++previous) {
            const pose_candidate *other =
                &context->candidates[context->order[previous]];
            if (other->score < 0.0F) continue;
            suppressed = box_iou(candidate->box, other->box) >
                context->iou_threshold;
        }
        if (suppressed) {
            /* Mark the loser so it cannot suppress a later candidate. */
            context->candidates[slot].score = -1.0F;
            continue;
        }

        /*
         * Every result gets its id here, so the executor's own numbering never
         * lands one on top of an id already given out.  The keypoints point at
         * their detection through parent_id.
         */
        const uint64_t object_id =
            (uint64_t)kept * (1U + context->keypoint_count) + 1U;
        context->results[result_count++] = (mipk_result) {
            .type = MIPK_RESULT_DETECTION,
            .result_id = object_id,
            .class_id = candidate->class_id,
            .label_set_id = context->config.label_set_id,
            .track_id = MIPK_INVALID_TRACK_ID,
            .attachment_index = MIPK_INVALID_ATTACHMENT_INDEX,
            .score = candidate->score,
            .value.bbox = {
                candidate->box[0], candidate->box[1],
                candidate->box[2], candidate->box[3],
            },
        };
        const float *points = context->keypoints + slot * keypoint_values;
        for (size_t point = 0; point < context->keypoint_count; ++point) {
            const float *value = points + point * POSE_VALUES_PER_KEYPOINT;
            context->results[result_count++] = (mipk_result) {
                .type = MIPK_RESULT_KEYPOINT,
                .result_id = object_id + 1U + point,
                .parent_id = object_id,
                .class_id = candidate->class_id,
                .label_set_id = context->config.label_set_id,
                .track_id = MIPK_INVALID_TRACK_ID,
                .attachment_index = MIPK_INVALID_ATTACHMENT_INDEX,
                .score = value[2],
                .value.keypoint = {
                    .x = value[0],
                    .y = value[1],
                    .visibility = value[2],
                    .keypoint_id = (int32_t)point,
                },
            };
        }
        ++kept;
    }
    output->items = context->results;
    output->item_count = result_count;
    output->item_capacity = context->result_capacity;
    return MIPK_OK;
}

static const mipk_capability_descriptor g_capability =
    MIPK_POSTPROCESS_CAPABILITY("yolo_pose", "2.0.0");

static const char *const g_options[] = {
    "output_index",
    "keypoint_count",
    "score_threshold",
    "iou_threshold",
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
        .result_type_mask = MIPK_RESULT_MASK(MIPK_RESULT_DETECTION) |
            MIPK_RESULT_MASK(MIPK_RESULT_KEYPOINT),
        .label_set_rule = MIPK_LABEL_SET_FIXED,
    },
};

const mipk_postprocess_ops *mipk_yolo_pose_postprocess(void)
{
    static const mipk_runtime_option runtime_options[] = {
        {.struct_size = sizeof(mipk_runtime_option),
            .name = "score_threshold", .type = MIPK_RUNTIME_OPTION_FLOAT,
            .description = "Minimum object score",
            .default_json = "0.5"},
        {.struct_size = sizeof(mipk_runtime_option),
            .name = "iou_threshold", .type = MIPK_RUNTIME_OPTION_FLOAT,
            .description = "Object suppression IoU threshold",
            .default_json = "0.45"},
    };
    static const mipk_postprocess_ops operations = {
        .capability = &g_capability,
        .schema = &g_schema,
        .name = "yolo_pose",
        .create = pose_create,
        .destroy = pose_destroy,
        .validate = pose_validate,
        .process = pose_process,
        .runtime_options = runtime_options,
        .runtime_option_count = sizeof(runtime_options) /
            sizeof(runtime_options[0]),
    };
    return &operations;
}

mipk_status mipk_yolo_pose_register(void)
{
    return mipk_postprocess_register(mipk_yolo_pose_postprocess());
}
