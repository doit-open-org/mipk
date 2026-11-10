#include "mipk_module.h"
#include "mipk_wake_word_trigger.h"

#include "cJSON.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    SMOOTH_NONE = 0,
    SMOOTH_MEAN,
    SMOOTH_MAX,
} smoothing_method;

typedef struct {
    float *scores;
    size_t next_score;
    size_t score_count;
    float score_sum;
    size_t confirmation_hits;
    size_t confirmation_gaps;
} class_state;

typedef struct {
    mipk_operator_config config;
    smoothing_method smoothing;
    size_t smoothing_window;
    size_t confirmation_hits;
    size_t confirmation_gaps;
    size_t cooldown_inferences;
    size_t cooldown_remaining;
    float score_threshold;
    class_state *classes;
    float *score_storage;
    mipk_result output;
} wake_word_context;

#define WAKE_WORD_CHECKPOINT_BYTES (64U * 1024U)

static bool checked_product(size_t first, size_t second, size_t *value)
{
    if (first != 0 && second > SIZE_MAX / first) return false;
    *value = first * second;
    return true;
}

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

static bool json_float(const cJSON *json, const char *name,
    float default_value, float *value)
{
    const cJSON *item = cJSON_GetObjectItemCaseSensitive(json, name);
    if (item == NULL) {
        *value = default_value;
        return true;
    }
    if (!cJSON_IsNumber(item) || !isfinite(item->valuedouble)) return false;
    *value = (float)item->valuedouble;
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

static void reset_state(wake_word_context *context)
{
    for (size_t index = 0; index < context->config.class_count; ++index) {
        class_state *state = &context->classes[index];
        memset(state->scores, 0,
            context->smoothing_window * sizeof(*state->scores));
        state->next_score = 0;
        state->score_count = 0;
        state->score_sum = 0.0F;
        state->confirmation_hits = 0;
        state->confirmation_gaps = 0;
    }
}

static void wake_word_destroy(void *private_data)
{
    wake_word_context *context = private_data;
    if (context == NULL) return;
    free(context->score_storage);
    free(context->classes);
    free(context);
}

static void *wake_word_create(const mipk_operator_config *config,
    const char *options_json, mipk_status *status)
{
    if (status == NULL) return NULL;
    *status = MIPK_ERR_ARGUMENT;
    if (config == NULL || config->classes == NULL || config->class_count == 0 ||
        config->inference_interval_ms == 0 || options_json == NULL) return NULL;
    wake_word_context *context = calloc(1, sizeof(*context));
    if (context == NULL) {
        *status = MIPK_ERR_MEMORY;
        return NULL;
    }
    context->config = *config;
    if (context->config.label_set_id == MIPK_INVALID_LABEL_SET_ID) {
        context->config.label_set_id = 0;
    }
    cJSON *json = cJSON_Parse(options_json);
    const char *smoothing = json == NULL ? NULL :
        json_string(json, "smoothing", "mean");
    size_t cooldown_ms = 0;
    bool valid = json != NULL && smoothing != NULL &&
        json_size(json, "smoothing_window", 5, &context->smoothing_window) &&
        json_size(json, "confirmation_hits", 1, &context->confirmation_hits) &&
        json_size(json, "confirmation_gaps", 0, &context->confirmation_gaps) &&
        json_size(json, "cooldown_ms", 2000, &cooldown_ms) &&
        json_float(json, "score_threshold", 0.8F, &context->score_threshold) &&
        context->confirmation_hits > 0 && context->score_threshold >= 0.0F &&
        context->score_threshold <= 1.0F;
    if (valid && strcmp(smoothing, "none") == 0) {
        context->smoothing = SMOOTH_NONE;
        context->smoothing_window = 1;
    } else if (valid && strcmp(smoothing, "mean") == 0 &&
        context->smoothing_window > 0) {
        context->smoothing = SMOOTH_MEAN;
    } else if (valid && strcmp(smoothing, "max") == 0 &&
        context->smoothing_window > 0) {
        context->smoothing = SMOOTH_MAX;
    } else {
        valid = false;
    }
    const size_t interval_ms = context->config.inference_interval_ms;
    if (valid && cooldown_ms <= SIZE_MAX - interval_ms + 1) {
        context->cooldown_inferences =
            (cooldown_ms + interval_ms - 1) / interval_ms;
    } else {
        valid = false;
    }
    cJSON_Delete(json);
    size_t score_count = 0;
    valid = valid && checked_product(config->class_count,
        context->smoothing_window, &score_count);
    if (valid) {
        context->classes = calloc(config->class_count,
            sizeof(*context->classes));
        context->score_storage = calloc(score_count,
            sizeof(*context->score_storage));
        valid = context->classes != NULL && context->score_storage != NULL;
    }
    if (!valid) {
        wake_word_destroy(context);
        return NULL;
    }
    for (size_t index = 0; index < config->class_count; ++index) {
        context->classes[index].scores = context->score_storage +
            index * context->smoothing_window;
    }
    *status = MIPK_OK;
    return context;
}

static mipk_status wake_word_validate(void *private_data,
    const mipk_mutable_tensor *inputs, size_t input_count,
    const mipk_tensor *outputs, size_t output_count)
{
    (void)inputs;
    (void)input_count;
    (void)outputs;
    (void)output_count;
    return private_data == NULL ? MIPK_ERR_OPERATOR : MIPK_OK;
}

static bool smooth_score(const wake_word_context *context,
    class_state *state, float score, float *smoothed)
{
    if (state->score_count == context->smoothing_window) {
        state->score_sum -= state->scores[state->next_score];
    } else {
        ++state->score_count;
    }
    state->scores[state->next_score] = score;
    state->score_sum += score;
    state->next_score = (state->next_score + 1) % context->smoothing_window;
    if (context->smoothing == SMOOTH_MAX) {
        *smoothed = state->scores[0];
        for (size_t index = 1; index < state->score_count; ++index) {
            if (state->scores[index] > *smoothed) *smoothed = state->scores[index];
        }
    } else {
        *smoothed = state->score_sum / (float)state->score_count;
    }
    return state->score_count == context->smoothing_window;
}

static float class_threshold(const wake_word_context *context, size_t class_id)
{
    const float threshold = context->config.classes[class_id].score_threshold;
    return threshold < 0.0F ? context->score_threshold : threshold;
}

static mipk_status wake_word_process(void *private_data,
    mipk_operator_io *execution)
{
    wake_word_context *context = private_data;
    if (context == NULL || execution == NULL || execution->scores == NULL ||
        execution->score_count != context->config.class_count) {
        return MIPK_ERR_ARGUMENT;
    }
    const float *scores = execution->scores;
    execution->items = &context->output;
    execution->item_count = 0;
    execution->item_capacity = 1;
    execution->scores = NULL;
    execution->score_count = 0;
    if (context->cooldown_remaining > 0) {
        --context->cooldown_remaining;
        reset_state(context);
        return MIPK_OK;
    }
    size_t triggered_class = SIZE_MAX;
    float triggered_score = -FLT_MAX;
    for (size_t index = 0; index < context->config.class_count; ++index) {
        class_state *state = &context->classes[index];
        float smoothed = 0.0F;
        const bool ready = smooth_score(context, state,
            scores[index], &smoothed);
        if (!ready || !context->config.classes[index].enabled) continue;
        if (smoothed >= class_threshold(context, index)) {
            ++state->confirmation_hits;
            state->confirmation_gaps = 0;
        } else if (state->confirmation_hits > 0 &&
            state->confirmation_gaps < context->confirmation_gaps) {
            ++state->confirmation_gaps;
        } else {
            state->confirmation_hits = 0;
            state->confirmation_gaps = 0;
        }
        if (state->confirmation_hits >= context->confirmation_hits &&
            (triggered_class == SIZE_MAX || smoothed > triggered_score)) {
            triggered_class = index;
            triggered_score = smoothed;
        }
    }
    if (triggered_class == SIZE_MAX) return MIPK_OK;
    context->output = (mipk_result) {
        .type = MIPK_RESULT_CLASSIFICATION,
        .class_id = (int32_t)triggered_class,
        .label_set_id = context->config.label_set_id,
        .track_id = MIPK_INVALID_TRACK_ID,
        .attachment_index = MIPK_INVALID_ATTACHMENT_INDEX,
        .score = triggered_score,
    };
    execution->item_count = 1;
    context->cooldown_remaining = context->cooldown_inferences;
    reset_state(context);
    return MIPK_OK;
}

static mipk_status wake_word_reset(void *private_data)
{
    wake_word_context *context = private_data;
    if (context == NULL) return MIPK_ERR_ARGUMENT;
    context->cooldown_remaining = 0;
    reset_state(context);
    return MIPK_OK;
}

static mipk_status wake_word_checkpoint(void *private_data, void *buffer,
    size_t size)
{
    wake_word_context *context = private_data;
    if (context == NULL || buffer == NULL || size != WAKE_WORD_CHECKPOINT_BYTES) {
        return MIPK_ERR_ARGUMENT;
    }
    const size_t states_size = context->config.class_count *
        sizeof(*context->classes);
    const size_t scores_size = context->config.class_count *
        context->smoothing_window * sizeof(*context->score_storage);
    if (states_size > size - sizeof(context->cooldown_remaining) ||
        scores_size > size - sizeof(context->cooldown_remaining) - states_size) {
        return MIPK_ERR_CAPACITY;
    }
    uint8_t *bytes = buffer;
    memcpy(bytes, &context->cooldown_remaining,
        sizeof(context->cooldown_remaining));
    memcpy(bytes + sizeof(context->cooldown_remaining), context->classes,
        states_size);
    memcpy(bytes + sizeof(context->cooldown_remaining) + states_size,
        context->score_storage, scores_size);
    return MIPK_OK;
}

static mipk_status wake_word_restore(void *private_data, const void *buffer,
    size_t size)
{
    wake_word_context *context = private_data;
    if (context == NULL || buffer == NULL || size != WAKE_WORD_CHECKPOINT_BYTES) {
        return MIPK_ERR_ARGUMENT;
    }
    const size_t states_size = context->config.class_count *
        sizeof(*context->classes);
    const size_t scores_size = context->config.class_count *
        context->smoothing_window * sizeof(*context->score_storage);
    if (states_size > size - sizeof(context->cooldown_remaining) ||
        scores_size > size - sizeof(context->cooldown_remaining) - states_size) {
        return MIPK_ERR_RUNTIME;
    }
    const uint8_t *bytes = buffer;
    memcpy(&context->cooldown_remaining, bytes,
        sizeof(context->cooldown_remaining));
    memcpy(context->classes, bytes + sizeof(context->cooldown_remaining),
        states_size);
    memcpy(context->score_storage,
        bytes + sizeof(context->cooldown_remaining) + states_size, scores_size);
    for (size_t index = 0; index < context->config.class_count; ++index) {
        context->classes[index].scores = context->score_storage +
            index * context->smoothing_window;
    }
    return MIPK_OK;
}

static const mipk_capability_descriptor g_capability =
    MIPK_OPERATOR_CAPABILITY("wake_word_trigger", "2.0.0");

static const char *const g_options[] = {
    "score_threshold",
    "smoothing",
    "smoothing_window",
    "confirmation_hits",
    "confirmation_gaps",
    "cooldown_ms",
};

static const mipk_port_schema g_inputs[] = {
    {
        .port = "scores",
        .accepted_value_mask = MIPK_VALUE_MASK(MIPK_VALUE_RESULTS),
        .accepted_domain_mask = MIPK_DOMAIN_MASK(MIPK_DOMAIN_AUDIO),
        .results = {
            .result_type_mask = MIPK_RESULT_MASK(MIPK_RESULT_CLASSIFICATION),
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
    .output_domain = MIPK_DOMAIN_AUDIO,
    .output_results = {
        .result_type_mask = MIPK_RESULT_MASK(MIPK_RESULT_CLASSIFICATION),
        .label_set_rule = MIPK_LABEL_SET_INHERIT,
        .label_set_from_port = "scores",
    },
};

const mipk_operator_ops *mipk_wake_word_trigger_operator(void)
{
    static const mipk_runtime_option runtime_options[] = {
        {.struct_size = sizeof(mipk_runtime_option),
            .name = "score_threshold", .type = MIPK_RUNTIME_OPTION_FLOAT,
            .description = "Minimum smoothed score for a hit",
            .default_json = "0.8"},
        {.struct_size = sizeof(mipk_runtime_option),
            .name = "confirmation_hits", .type = MIPK_RUNTIME_OPTION_UINT32,
            .description = "Hits required to trigger a wake event",
            .default_json = "1"},
        {.struct_size = sizeof(mipk_runtime_option),
            .name = "confirmation_gaps", .type = MIPK_RUNTIME_OPTION_UINT32,
            .description = "Misses allowed between confirmation hits",
            .default_json = "0"},
        {.struct_size = sizeof(mipk_runtime_option), .name = "cooldown_ms",
            .type = MIPK_RUNTIME_OPTION_UINT32, .unit = "ms",
            .description = "Delay before another wake event",
            .default_json = "2000"},
    };
    static const mipk_operator_ops operations = {
        .capability = &g_capability,
        .schema = &g_schema,
        .name = "wake_word_trigger",
        .create = wake_word_create,
        .destroy = wake_word_destroy,
        .process = wake_word_process,
        .validate = wake_word_validate,
        .stateful = true,
        .reset = wake_word_reset,
        .checkpoint_size = WAKE_WORD_CHECKPOINT_BYTES,
        .checkpoint = wake_word_checkpoint,
        .restore = wake_word_restore,
        .runtime_options = runtime_options,
        .runtime_option_count = sizeof(runtime_options) /
            sizeof(runtime_options[0]),
    };
    return &operations;
}

mipk_status mipk_wake_word_trigger_register(void)
{
    return mipk_operator_register(mipk_wake_word_trigger_operator());
}

mipk_status mipk_pipeline_update_wake_word(mipk_pipeline *pipeline,
    const char *node_id, const mipk_wake_word_runtime_options *options)
{
    if (pipeline == NULL || node_id == NULL || options == NULL ||
        options->struct_size != sizeof(*options) || options->field_mask == 0 ||
        (options->field_mask & ~MIPK_WAKE_WORD_UPDATE_ALL) != 0) {
        return MIPK_ERR_ARGUMENT;
    }
    cJSON *json = cJSON_CreateObject();
    mipk_status status = json == NULL ? MIPK_ERR_MEMORY : MIPK_OK;
#define ADD_NUMBER(bit, field, json_name) \
    if (status == MIPK_OK && (options->field_mask & (bit)) != 0 && \
        cJSON_AddNumberToObject(json, (json_name), options->field) == NULL) \
        status = MIPK_ERR_MEMORY
    ADD_NUMBER(MIPK_WAKE_WORD_UPDATE_SCORE_THRESHOLD, score_threshold,
        "score_threshold");
    ADD_NUMBER(MIPK_WAKE_WORD_UPDATE_CONFIRMATION_HITS, confirmation_hits,
        "confirmation_hits");
    ADD_NUMBER(MIPK_WAKE_WORD_UPDATE_CONFIRMATION_GAPS, confirmation_gaps,
        "confirmation_gaps");
    ADD_NUMBER(MIPK_WAKE_WORD_UPDATE_COOLDOWN, cooldown_ms, "cooldown_ms");
#undef ADD_NUMBER
    char *options_json = status == MIPK_OK ? cJSON_PrintUnformatted(json) : NULL;
    if (status == MIPK_OK && options_json == NULL) status = MIPK_ERR_MEMORY;
    if (status == MIPK_OK) {
        status = mipk_pipeline_update_stage_options(pipeline, node_id,
            MIPK_STAGE_OPERATOR, "wake_word_trigger", options_json);
    }
    free(options_json);
    cJSON_Delete(json);
    return status;
}
