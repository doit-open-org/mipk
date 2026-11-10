#ifndef MIPK_PIPELINE_RUNTIME_INTERNAL_H
#define MIPK_PIPELINE_RUNTIME_INTERNAL_H

#include "mipk_model.h"
#include "mipk_pipeline_internal.h"

#include <pthread.h>

mipk_status mipk_collect_results(const mipk_value *output,
    mipk_result_buffer *results);

typedef struct {
    const mipk_stage_config *config;
    mipk_stage_type type;
    mipk_model *model;
    const mipk_operator_ops *operator_ops;
    const mipk_postprocess_ops *postprocess_ops;
    mipk_operator_config node_config;
    char *active_options_json;
    void *context;
    mipk_tensor *prepared_tensors;
    size_t prepared_tensor_count;
    void *checkpoint;
    mipk_value *collection_items;
    mipk_result *collection_results;
    size_t collection_capacity;
    size_t collection_result_capacity;
    size_t collection_results_per_item;
} mipk_runtime_stage;

struct mipk_pipeline_runtime {
    mipk_pipeline_config config;
    mipk_model **models;
    size_t model_count;
    /*
     * Turns the graph's output value into the caller's result buffer.  The
     * builder installs the generic collector; tests may substitute one.
     */
    mipk_status (*collect)(const mipk_value *output,
        mipk_result_buffer *results);
    mipk_runtime_stage *stages;
    size_t stage_count;
    size_t output_stage_index;
    bool poisoned;
    mipk_value *execution_values;
    mipk_named_value *execution_inputs;
    mipk_memory_plan memory_plan;
    pthread_mutex_t run_lock;
    bool run_lock_initialized;
};

#endif
