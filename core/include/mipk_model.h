#ifndef MIPK_MODEL_INTERNAL_H
#define MIPK_MODEL_INTERNAL_H

#include "mipk_pipeline_internal.h"

typedef struct mipk_model mipk_model;

mipk_status mipk_core_model_create(const mipk_pipeline_config *pipeline_config,
    const mipk_model_config *config,
    mipk_model **model);
void mipk_core_model_destroy(mipk_model *model);
mipk_mutable_tensor *mipk_core_model_inputs(mipk_model *model);
size_t mipk_core_model_input_count(const mipk_model *model);
const mipk_tensor *mipk_core_model_outputs(const mipk_model *model);
size_t mipk_core_model_output_count(const mipk_model *model);
mipk_status mipk_core_model_execute(mipk_model *model,
    mipk_execution_context *execution);

#endif
