#ifndef MIPK_PIPELINE_INTERNAL_H
#define MIPK_PIPELINE_INTERNAL_H

#include "mipk.h"
#include "mipk_operator.h"
#include "mipk_postprocess.h"
#include "mipk_task.h"
#include "mipk_module.h"

typedef mipk_postprocess_class mipk_class_config;

typedef struct {
    const char *name;
    mipk_class_config *classes;
    size_t class_count;
} mipk_label_set_config;

typedef struct {
    const char *id;
    const char *backend_name;
    const char *kind;
    const char *options_json;
    const char *model_path;
    size_t model_size;
    mipk_status (*model_loader)(void *context, void *destination,
        size_t destination_size);
    void *model_loader_context;
    const char *runtime_config_path;
} mipk_model_config;

typedef struct {
    const mipk_model_config *models;
    size_t model_count;
    mipk_data_format input_format;
    const mipk_class_config *classes;
    size_t class_count;
    const mipk_label_set_config *label_sets;
    size_t label_set_count;
    size_t result_capacity;
    uint64_t result_type_mask;
    uint32_t input_width;
    uint32_t input_height;
    size_t input_size;
    uint32_t model_input_width;
    uint32_t model_input_height;
    size_t model_input_size;
    uint32_t resized_width;
    uint32_t resized_height;
    uint32_t pad_x;
    uint32_t pad_y;
    float to_source[9];
    uint32_t input_sample_rate;
    uint32_t input_channels;
    uint32_t input_frame_samples;
    uint32_t inference_interval_ms;
    const mipk_stage_config *stages;
    size_t stage_count;
    const char *output;
    const mipk_platform_context *platform;
} mipk_pipeline_config;

typedef mipk_operator_io mipk_execution_context;

typedef struct mipk_pipeline_runtime mipk_pipeline_runtime;

mipk_status mipk_pipeline_runtime_create(const mipk_pipeline_config *config,
    mipk_pipeline_runtime **runtime);
void mipk_pipeline_runtime_destroy(mipk_pipeline_runtime *runtime);
mipk_status mipk_pipeline_runtime_get_memory_plan(
    const mipk_pipeline_runtime *runtime, mipk_memory_plan *plan);
mipk_status mipk_pipeline_runtime_run(mipk_pipeline_runtime *runtime,
    const mipk_input *input, mipk_result_buffer *results);
mipk_status mipk_pipeline_runtime_reset(mipk_pipeline_runtime *runtime);
bool mipk_pipeline_runtime_busy(mipk_pipeline_runtime *runtime);
mipk_status mipk_pipeline_runtime_update_node_options(
    mipk_pipeline_runtime *runtime, const char *node_id,
    const char *options_json);
mipk_status mipk_pipeline_runtime_get_option_at(
    const mipk_pipeline_runtime *runtime, size_t index,
    mipk_runtime_option_info *info);
size_t mipk_pipeline_runtime_get_option_count(
    const mipk_pipeline_runtime *runtime);
mipk_status mipk_pipeline_runtime_get_option_value_at(
    mipk_pipeline_runtime *runtime, size_t index, char *buffer,
    size_t capacity, size_t *required_size);
mipk_status mipk_pipeline_runtime_update_node_options_checked(
    mipk_pipeline_runtime *runtime, const char *node_id,
    mipk_stage_type expected_type, const char *expected_name,
    const char *options_json);
mipk_status mipk_pipeline_runtime_apply_overrides_json(
    mipk_pipeline_runtime *runtime, const char *overrides_json);
const char *mipk_pipeline_runtime_class_name(
    const mipk_pipeline_runtime *runtime, int32_t class_id);

const mipk_operator_ops *mipk_operator_find(const char *name);
const mipk_postprocess_ops *mipk_postprocess_find(const char *name);
mipk_status mipk_capability_registry_freeze(void);
/* Arms/disarms the diagnostic sink for the calling thread. */
void mipk_diag_begin(char *buffer, size_t size);
void mipk_diag_end(void);
const mipk_capability_descriptor *mipk_capability_find(
    mipk_capability_kind kind, const char *name);
bool mipk_capability_version_at_least(const char *actual,
    const char *minimum);

/*
 * Whether an image_preprocess stage should keep the resize geometry the
 * manifest worked out, rather than resizing straight to its target.
 *
 * The manifest computes the letterbox for the pipeline's own model input --
 * how much the frame is scaled by and how much padding goes around it -- and
 * the same numbers are what a postprocess later uses to map results back to
 * the frame.  A stage that IS that resize has to use them, or the two ends
 * disagree and every result comes back stretched about the frame centre.
 *
 * A stage that is not is on its own.  A per-item stage resizes a crop whose
 * size is only known per frame, and a whole-frame stage feeding a differently
 * sized model is not the one the manifest measured; both simply resize to
 * their target.
 */
/*
 * Works out the letterbox: how large the frame is drawn and where, inside a
 * target of the given size.  Returns false when the result would be empty.
 */
bool mipk_preprocess_letterbox(uint32_t source_width, uint32_t source_height,
    uint32_t target_width, uint32_t target_height, bool yuv,
    uint32_t *resized_width, uint32_t *resized_height, uint32_t *pad_x,
    uint32_t *pad_y);

bool mipk_preprocess_keeps_manifest_geometry(mipk_execution_mode execution,
    size_t target_width, size_t target_height, uint32_t manifest_width,
    uint32_t manifest_height);

#endif
