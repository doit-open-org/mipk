#include "mipk_module.h"
#include "mipk_bytetrack.h"

#include "cJSON.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define MIPK_BYTETRACK_MAX_TRACKS 256U

typedef struct {
    bool active;
    bool matched;
    bool created;
    int64_t id;
    int32_t class_id;
    uint32_t label_set_id;
    mipk_bbox bbox;
    float vx;
    float vy;
    float vw;
    float vh;
    float score;
    uint64_t last_pts_us;
} track_slot;

typedef struct {
    track_slot tracks[MIPK_BYTETRACK_MAX_TRACKS];
    int64_t next_track_id;
    uint64_t last_pts_us;
    bool has_pts;
} bytetrack_state;

typedef struct {
    bytetrack_state state;
    mipk_result *output;
    size_t max_tracks;
    float high_threshold;
    float low_threshold;
    float new_threshold;
    float min_match_iou;
    uint64_t lost_timeout_us;
    uint64_t max_prediction_dt_us;
    uint64_t reset_gap_us;
    bool class_aware;
    bool emit_lost;
    bool emit_removed;
} bytetrack_context;

static bool json_number(const cJSON *json, const char *name, double *value)
{
    const cJSON *item = cJSON_GetObjectItemCaseSensitive(json, name);
    if (!cJSON_IsNumber(item) || !isfinite(item->valuedouble)) return false;
    *value = item->valuedouble;
    return true;
}

static bool json_bool(const cJSON *json, const char *name, bool *value)
{
    const cJSON *item = cJSON_GetObjectItemCaseSensitive(json, name);
    if (!cJSON_IsBool(item)) return false;
    *value = cJSON_IsTrue(item);
    return true;
}

static void reset_state(bytetrack_context *context)
{
    memset(&context->state, 0, sizeof(context->state));
    context->state.next_track_id = 1;
}

static void *bytetrack_create(const mipk_operator_config *config,
    const char *options_json, mipk_status *status)
{
    if (status == NULL) return NULL;
    *status = MIPK_ERR_ARGUMENT;
    cJSON *json = options_json == NULL ? NULL : cJSON_Parse(options_json);
    double high = 0, low = 0, fresh = 0, iou = 0, lost = 0, max_dt = 0,
        reset_gap = 0, max_tracks = 0;
    bool class_aware = false, emit_lost = false, emit_removed = false;
    bool valid = config != NULL && cJSON_IsObject(json) &&
        json_number(json, "track_high_threshold", &high) &&
        json_number(json, "track_low_threshold", &low) &&
        json_number(json, "new_track_threshold", &fresh) &&
        json_number(json, "min_match_iou", &iou) &&
        json_number(json, "lost_timeout_ms", &lost) &&
        json_number(json, "max_prediction_dt_ms", &max_dt) &&
        json_number(json, "reset_gap_ms", &reset_gap) &&
        json_number(json, "max_tracks", &max_tracks) &&
        json_bool(json, "class_aware", &class_aware) &&
        json_bool(json, "emit_lost", &emit_lost) &&
        json_bool(json, "emit_removed", &emit_removed) &&
        0.0 <= low && low <= high && high <= fresh && fresh <= 1.0 &&
        0.0 <= iou && iou <= 1.0 && lost > 0.0 && max_dt > 0.0 &&
        max_dt <= reset_gap && floor(max_tracks) == max_tracks &&
        max_tracks >= 1.0 && max_tracks <= MIPK_BYTETRACK_MAX_TRACKS &&
        max_tracks <= (double)config->result_capacity;
    if (!valid) {
        cJSON_Delete(json);
        return NULL;
    }
    bytetrack_context *context = calloc(1, sizeof(*context));
    if (context == NULL) {
        cJSON_Delete(json);
        *status = MIPK_ERR_MEMORY;
        return NULL;
    }
    context->max_tracks = (size_t)max_tracks;
    context->output = calloc(context->max_tracks, sizeof(*context->output));
    if (context->output == NULL) {
        free(context);
        cJSON_Delete(json);
        *status = MIPK_ERR_MEMORY;
        return NULL;
    }
    context->high_threshold = (float)high;
    context->low_threshold = (float)low;
    context->new_threshold = (float)fresh;
    context->min_match_iou = (float)iou;
    context->lost_timeout_us = (uint64_t)lost * UINT64_C(1000);
    context->max_prediction_dt_us = (uint64_t)max_dt * UINT64_C(1000);
    context->reset_gap_us = (uint64_t)reset_gap * UINT64_C(1000);
    context->class_aware = class_aware;
    context->emit_lost = emit_lost;
    context->emit_removed = emit_removed;
    reset_state(context);
    cJSON_Delete(json);
    *status = MIPK_OK;
    return context;
}

static void bytetrack_destroy(void *private_data)
{
    bytetrack_context *context = private_data;
    if (context == NULL) return;
    free(context->output);
    free(context);
}

static float bbox_iou(mipk_bbox a, mipk_bbox b)
{
    const float x1 = fmaxf(a.x1, b.x1);
    const float y1 = fmaxf(a.y1, b.y1);
    const float x2 = fminf(a.x2, b.x2);
    const float y2 = fminf(a.y2, b.y2);
    const float intersection = fmaxf(0.0F, x2 - x1) * fmaxf(0.0F, y2 - y1);
    const float area_a = fmaxf(0.0F, a.x2 - a.x1) * fmaxf(0.0F, a.y2 - a.y1);
    const float area_b = fmaxf(0.0F, b.x2 - b.x1) * fmaxf(0.0F, b.y2 - b.y1);
    const float total = area_a + area_b - intersection;
    return total <= 0.0F ? 0.0F : intersection / total;
}

static mipk_bbox predicted_bbox(const track_slot *track, uint64_t pts_us,
    uint64_t max_dt_us)
{
    const uint64_t raw_dt = pts_us - track->last_pts_us;
    const float dt = (float)(raw_dt > max_dt_us ? max_dt_us : raw_dt) /
        1000000.0F;
    return (mipk_bbox) {
        .x1 = track->bbox.x1 + track->vx * dt,
        .y1 = track->bbox.y1 + track->vy * dt,
        .x2 = track->bbox.x2 + (track->vx + track->vw) * dt,
        .y2 = track->bbox.y2 + (track->vy + track->vh) * dt,
    };
}

static void update_track(track_slot *track, const mipk_result *detection,
    uint64_t pts_us)
{
    const float dt = track->last_pts_us == pts_us ? 0.0F :
        (float)(pts_us - track->last_pts_us) / 1000000.0F;
    if (dt > 0.0F) {
        track->vx = (detection->value.bbox.x1 - track->bbox.x1) / dt;
        track->vy = (detection->value.bbox.y1 - track->bbox.y1) / dt;
        track->vw = ((detection->value.bbox.x2 - detection->value.bbox.x1) -
            (track->bbox.x2 - track->bbox.x1)) / dt;
        track->vh = ((detection->value.bbox.y2 - detection->value.bbox.y1) -
            (track->bbox.y2 - track->bbox.y1)) / dt;
    }
    track->bbox = detection->value.bbox;
    track->score = detection->score;
    track->last_pts_us = pts_us;
    track->matched = true;
}

static track_slot *best_match(bytetrack_context *context,
    const mipk_result *detection, uint64_t pts_us)
{
    track_slot *best = NULL;
    float best_iou = context->min_match_iou;
    for (size_t index = 0; index < context->max_tracks; ++index) {
        track_slot *track = &context->state.tracks[index];
        if (!track->active || track->matched ||
            (context->class_aware && track->class_id != detection->class_id)) {
            continue;
        }
        const float iou = bbox_iou(predicted_bbox(track, pts_us,
            context->max_prediction_dt_us), detection->value.bbox);
        if (iou > best_iou || (iou == best_iou &&
                (best == NULL || track->id < best->id))) {
            best = track;
            best_iou = iou;
        }
    }
    return best;
}

static bool emit_track(bytetrack_context *context, const track_slot *track,
    mipk_track_state state, size_t *count)
{
    if (*count >= context->max_tracks) return false;
    context->output[(*count)++] = (mipk_result) {
        .type = MIPK_RESULT_TRACK,
        .track_id = track->id,
        .label_set_id = track->label_set_id,
        .class_id = track->class_id,
        .score = track->score,
        .attachment_index = MIPK_INVALID_ATTACHMENT_INDEX,
        .value.track = {.bbox = track->bbox, .state = state},
    };
    return true;
}

static mipk_status bytetrack_process(void *private_data,
    mipk_operator_io *execution)
{
    bytetrack_context *context = private_data;
    if (context == NULL || execution == NULL || execution->input == NULL ||
        execution->input->pts_us == MIPK_PTS_UNKNOWN ||
        (execution->item_count != 0 && execution->items == NULL)) {
        return MIPK_ERR_ARGUMENT;
    }
    const uint64_t pts_us = execution->input->pts_us;
    if (context->state.has_pts && (pts_us <= context->state.last_pts_us ||
        pts_us - context->state.last_pts_us > context->reset_gap_us)) {
        reset_state(context);
    }
    for (size_t index = 0; index < context->max_tracks; ++index) {
        context->state.tracks[index].matched = false;
        context->state.tracks[index].created = false;
    }
    for (int pass = 0; pass < 2; ++pass) {
        for (size_t index = 0; index < execution->item_count; ++index) {
            const mipk_result *detection = &execution->items[index];
            if (detection->type != MIPK_RESULT_DETECTION ||
                !isfinite(detection->score)) return MIPK_ERR_OPERATOR;
            const bool in_pass = pass == 0 ?
                detection->score >= context->high_threshold :
                detection->score >= context->low_threshold &&
                    detection->score < context->high_threshold;
            if (!in_pass) continue;
            track_slot *match = best_match(context, detection, pts_us);
            if (match != NULL) update_track(match, detection, pts_us);
        }
    }
    for (size_t index = 0; index < execution->item_count; ++index) {
        const mipk_result *detection = &execution->items[index];
        if (detection->score < context->new_threshold) continue;
        bool already_matched = false;
        for (size_t track_index = 0; track_index < context->max_tracks;
            ++track_index) {
            const track_slot *track = &context->state.tracks[track_index];
            if (track->active && track->matched &&
                track->bbox.x1 == detection->value.bbox.x1 &&
                track->bbox.y1 == detection->value.bbox.y1) {
                already_matched = true;
                break;
            }
        }
        if (already_matched) continue;
        for (size_t slot = 0; slot < context->max_tracks; ++slot) {
            track_slot *track = &context->state.tracks[slot];
            if (track->active) continue;
            *track = (track_slot) {
                .active = true,
                .matched = true,
                .created = true,
                .id = context->state.next_track_id++,
                .class_id = detection->class_id,
                .label_set_id = detection->label_set_id,
                .bbox = detection->value.bbox,
                .score = detection->score,
                .last_pts_us = pts_us,
            };
            break;
        }
    }
    size_t output_count = 0;
    for (size_t index = 0; index < context->max_tracks; ++index) {
        track_slot *track = &context->state.tracks[index];
        if (!track->active) continue;
        if (track->matched) {
            emit_track(context, track, track->created ? MIPK_TRACK_NEW :
                MIPK_TRACK_UPDATE,
                &output_count);
            continue;
        }
        const bool removed = pts_us - track->last_pts_us >
            context->lost_timeout_us;
        if (removed) {
            if (context->emit_removed) {
                emit_track(context, track, MIPK_TRACK_REMOVED, &output_count);
            }
            track->active = false;
        } else if (context->emit_lost) {
            emit_track(context, track, MIPK_TRACK_LOST, &output_count);
        }
    }
    context->state.last_pts_us = pts_us;
    context->state.has_pts = true;
    execution->items = context->output;
    execution->item_count = output_count;
    execution->item_capacity = context->max_tracks;
    return MIPK_OK;
}

static mipk_status bytetrack_validate(void *context,
    const mipk_mutable_tensor *inputs, size_t input_count,
    const mipk_tensor *outputs, size_t output_count)
{
    (void)inputs;
    (void)input_count;
    (void)outputs;
    (void)output_count;
    return context == NULL ? MIPK_ERR_ARGUMENT : MIPK_OK;
}

static mipk_status bytetrack_reset(void *private_data)
{
    bytetrack_context *context = private_data;
    if (context == NULL) return MIPK_ERR_ARGUMENT;
    reset_state(context);
    return MIPK_OK;
}

static mipk_status bytetrack_checkpoint(void *private_data, void *buffer,
    size_t size)
{
    bytetrack_context *context = private_data;
    if (context == NULL || buffer == NULL || size != sizeof(context->state)) {
        return MIPK_ERR_ARGUMENT;
    }
    memcpy(buffer, &context->state, size);
    return MIPK_OK;
}

static mipk_status bytetrack_restore(void *private_data, const void *buffer,
    size_t size)
{
    bytetrack_context *context = private_data;
    if (context == NULL || buffer == NULL || size != sizeof(context->state)) {
        return MIPK_ERR_ARGUMENT;
    }
    memcpy(&context->state, buffer, size);
    return MIPK_OK;
}

static const mipk_capability_descriptor g_capability =
    MIPK_OPERATOR_CAPABILITY("bytetrack", "2.0.0");

static const char *const g_options[] = {
    "track_high_threshold",
    "track_low_threshold",
    "new_track_threshold",
    "min_match_iou",
    "lost_timeout_ms",
    "max_prediction_dt_ms",
    "reset_gap_ms",
    "class_aware",
    "max_tracks",
    "emit_lost",
    "emit_removed",
};

static const mipk_port_schema g_inputs[] = {
    {
        .port = "frame",
        .accepted_value_mask = MIPK_VALUE_MASK(MIPK_VALUE_INPUT),
        .accepted_domain_mask = MIPK_DOMAIN_MASK(MIPK_DOMAIN_IMAGE),
        .required = true,
    },
    {
        .port = "detections",
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
        .result_type_mask = MIPK_RESULT_MASK(MIPK_RESULT_TRACK),
        .label_set_rule = MIPK_LABEL_SET_INHERIT,
        .label_set_from_port = "detections",
    },
};

const mipk_operator_ops *mipk_bytetrack_operator(void)
{
    static const mipk_runtime_option runtime_options[] = {
        {.struct_size = sizeof(mipk_runtime_option),
            .name = "track_high_threshold", .type = MIPK_RUNTIME_OPTION_FLOAT,
            .description = "Minimum score for first-pass association"},
        {.struct_size = sizeof(mipk_runtime_option),
            .name = "track_low_threshold", .type = MIPK_RUNTIME_OPTION_FLOAT,
            .description = "Minimum score for second-pass association"},
        {.struct_size = sizeof(mipk_runtime_option),
            .name = "new_track_threshold", .type = MIPK_RUNTIME_OPTION_FLOAT,
            .description = "Minimum score for creating a track"},
        {.struct_size = sizeof(mipk_runtime_option), .name = "min_match_iou",
            .type = MIPK_RUNTIME_OPTION_FLOAT,
            .description = "Minimum IoU accepted for association"},
        {.struct_size = sizeof(mipk_runtime_option), .name = "lost_timeout_ms",
            .type = MIPK_RUNTIME_OPTION_UINT32,
            .unit = "ms", .description = "Time before a lost track is removed"},
        {.struct_size = sizeof(mipk_runtime_option),
            .name = "max_prediction_dt_ms", .type = MIPK_RUNTIME_OPTION_UINT32,
            .unit = "ms", .description = "Maximum prediction time step"},
        {.struct_size = sizeof(mipk_runtime_option), .name = "reset_gap_ms",
            .type = MIPK_RUNTIME_OPTION_UINT32,
            .unit = "ms", .description = "Input gap that resets tracking"},
        {.struct_size = sizeof(mipk_runtime_option), .name = "class_aware",
            .type = MIPK_RUNTIME_OPTION_BOOL,
            .description = "Match detections only within the same class"},
        {.struct_size = sizeof(mipk_runtime_option), .name = "emit_lost",
            .type = MIPK_RUNTIME_OPTION_BOOL,
            .description = "Include lost tracks in output"},
        {.struct_size = sizeof(mipk_runtime_option), .name = "emit_removed",
            .type = MIPK_RUNTIME_OPTION_BOOL,
            .description = "Include removed tracks in output"},
    };
    static const mipk_operator_ops operations = {
        .capability = &g_capability,
        .schema = &g_schema,
        .name = "bytetrack",
        .create = bytetrack_create,
        .destroy = bytetrack_destroy,
        .process = bytetrack_process,
        .validate = bytetrack_validate,
        .stateful = true,
        .reset = bytetrack_reset,
        .checkpoint_size = sizeof(bytetrack_state),
        .checkpoint = bytetrack_checkpoint,
        .restore = bytetrack_restore,
        .runtime_options = runtime_options,
        .runtime_option_count = sizeof(runtime_options) /
            sizeof(runtime_options[0]),
    };
    return &operations;
}

mipk_status mipk_bytetrack_register(void)
{
    return mipk_operator_register(mipk_bytetrack_operator());
}

mipk_status mipk_pipeline_update_bytetrack(mipk_pipeline *pipeline,
    const char *node_id, const mipk_bytetrack_runtime_options *options)
{
    if (pipeline == NULL || node_id == NULL || options == NULL ||
        options->struct_size != sizeof(*options) || options->field_mask == 0 ||
        (options->field_mask & ~MIPK_BYTETRACK_UPDATE_ALL) != 0) {
        return MIPK_ERR_ARGUMENT;
    }
    cJSON *json = cJSON_CreateObject();
    mipk_status status = json == NULL ? MIPK_ERR_MEMORY : MIPK_OK;
#define ADD_NUMBER(bit, field, json_name) \
    if (status == MIPK_OK && (options->field_mask & (bit)) != 0 && \
        cJSON_AddNumberToObject(json, (json_name), options->field) == NULL) \
        status = MIPK_ERR_MEMORY
#define ADD_BOOL(bit, field, json_name) \
    if (status == MIPK_OK && (options->field_mask & (bit)) != 0 && \
        cJSON_AddBoolToObject(json, (json_name), options->field) == NULL) \
        status = MIPK_ERR_MEMORY
    ADD_NUMBER(MIPK_BYTETRACK_UPDATE_HIGH_THRESHOLD, track_high_threshold,
        "track_high_threshold");
    ADD_NUMBER(MIPK_BYTETRACK_UPDATE_LOW_THRESHOLD, track_low_threshold,
        "track_low_threshold");
    ADD_NUMBER(MIPK_BYTETRACK_UPDATE_NEW_THRESHOLD, new_track_threshold,
        "new_track_threshold");
    ADD_NUMBER(MIPK_BYTETRACK_UPDATE_MIN_MATCH_IOU, min_match_iou,
        "min_match_iou");
    ADD_NUMBER(MIPK_BYTETRACK_UPDATE_LOST_TIMEOUT, lost_timeout_ms,
        "lost_timeout_ms");
    ADD_NUMBER(MIPK_BYTETRACK_UPDATE_MAX_PREDICTION_DT, max_prediction_dt_ms,
        "max_prediction_dt_ms");
    ADD_NUMBER(MIPK_BYTETRACK_UPDATE_RESET_GAP, reset_gap_ms, "reset_gap_ms");
    ADD_BOOL(MIPK_BYTETRACK_UPDATE_CLASS_AWARE, class_aware, "class_aware");
    ADD_BOOL(MIPK_BYTETRACK_UPDATE_EMIT_LOST, emit_lost, "emit_lost");
    ADD_BOOL(MIPK_BYTETRACK_UPDATE_EMIT_REMOVED, emit_removed, "emit_removed");
#undef ADD_NUMBER
#undef ADD_BOOL
    char *options_json = status == MIPK_OK ? cJSON_PrintUnformatted(json) : NULL;
    if (status == MIPK_OK && options_json == NULL) status = MIPK_ERR_MEMORY;
    if (status == MIPK_OK) {
        status = mipk_pipeline_update_stage_options(pipeline, node_id,
            MIPK_STAGE_OPERATOR, "bytetrack", options_json);
    }
    free(options_json);
    cJSON_Delete(json);
    return status;
}
