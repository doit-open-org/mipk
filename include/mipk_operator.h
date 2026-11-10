#ifndef MIPK_OPERATOR_H
#define MIPK_OPERATOR_H

#include "mipk_capability.h"
#include "mipk_postprocess.h"
#include "mipk_value.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef mipk_postprocess_config mipk_operator_config;

typedef struct {
    /* Named graph inputs and output for graph-aware extensions. */
    const mipk_named_value *stage_inputs;
    size_t stage_input_count;
    mipk_value *stage_output;
    mipk_value *collection_items;
    size_t collection_capacity;
    mipk_result *collection_results;
    size_t collection_result_capacity;

    /* Compatibility view used by the built-in V2 operators. */
    const mipk_input *input;
    const mipk_image_view *image_view;
    mipk_mutable_tensor *model_inputs;
    size_t model_input_count;
    const mipk_tensor *model_outputs;
    size_t model_output_count;
    mipk_result *items;
    size_t item_count;
    size_t item_capacity;
    const float *scores;
    size_t score_count;
    mipk_result_buffer *results;
} mipk_operator_io;

typedef struct mipk_operator_ops {
    const mipk_capability_descriptor *capability;
    const mipk_node_schema *schema;
    const char *name;
    void *(*create)(const mipk_operator_config *config,
        const char *options_json, mipk_status *status);
    void (*destroy)(void *context);
    mipk_status (*process)(void *context, mipk_operator_io *io);
    mipk_status (*validate)(void *context,
        const mipk_mutable_tensor *model_inputs, size_t input_count,
        const mipk_tensor *model_outputs, size_t output_count);
    bool stateful;
    bool may_return_again;
    mipk_status (*reset)(void *context);
    size_t checkpoint_size;
    mipk_status (*checkpoint)(void *context, void *buffer, size_t size);
    mipk_status (*restore)(void *context, const void *buffer, size_t size);
    /* Options that applications may replace without rebuilding the graph. */
    const mipk_runtime_option *runtime_options;
    size_t runtime_option_count;
} mipk_operator_ops;

/* Register static-lifetime operator operations before opening a pipeline. */
mipk_status mipk_operator_register(const mipk_operator_ops *ops);

#ifdef __cplusplus
}
#endif
#endif
