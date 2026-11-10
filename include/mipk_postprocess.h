#ifndef MIPK_POSTPROCESS_H
#define MIPK_POSTPROCESS_H

#include "mipk_capability.h"
#include "mipk_runtime.h"
#include "mipk_value.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char *name;
    /* Backend/model class ID; -1 identifies a derived result class. */
    int32_t source_id;
    bool enabled;
    /* A negative value uses the postprocess stage's default threshold. */
    float score_threshold;
} mipk_postprocess_class;

typedef struct {
    const mipk_postprocess_class *classes;
    size_t class_count;
    size_t result_capacity;
    int32_t input_format;
    uint32_t source_width;
    uint32_t source_height;
    uint32_t input_width;
    uint32_t input_height;
    size_t input_size;
    uint32_t resized_width;
    uint32_t resized_height;
    uint32_t pad_x;
    uint32_t pad_y;
    float to_source[9];
    uint32_t input_sample_rate;
    uint32_t input_channels;
    uint32_t input_frame_samples;
    uint32_t inference_interval_ms;
    uint32_t label_set_id;
    /* The platform the pipeline was opened with, or NULL. */
    const struct mipk_platform_context *platform;
} mipk_postprocess_config;

typedef struct {
    /* Storage remains owned by the postprocess until its next process call. */
    mipk_result *items;
    size_t item_count;
    size_t item_capacity;
    const float *scores;
    size_t score_count;
} mipk_postprocess_output;

typedef struct mipk_postprocess_ops {
    const mipk_capability_descriptor *capability;
    const mipk_node_schema *schema;
    const char *name;
    void *(*create)(const mipk_postprocess_config *config,
        const char *options_json, mipk_status *status);
    void (*destroy)(void *context);
    mipk_status (*validate)(void *context,
        const mipk_mutable_tensor *model_inputs, size_t input_count,
        const mipk_tensor *model_outputs, size_t output_count);
    mipk_status (*process)(void *context, const mipk_tensor *model_outputs,
        size_t output_count, mipk_postprocess_output *output);
    /* Options that applications may replace without rebuilding the graph. */
    const mipk_runtime_option *runtime_options;
    size_t runtime_option_count;
} mipk_postprocess_ops;

/* Register static-lifetime postprocess operations before opening a pipeline. */
mipk_status mipk_postprocess_register(const mipk_postprocess_ops *ops);

mipk_status mipk_postprocess_option_get_size(const char *options_json,
    const char *name, size_t default_value, size_t *value);
mipk_status mipk_postprocess_option_get_bool(const char *options_json,
    const char *name, bool default_value, bool *value);
mipk_status mipk_postprocess_option_get_float(const char *options_json,
    const char *name, float default_value, float *value);

#ifdef __cplusplus
}
#endif
#endif
