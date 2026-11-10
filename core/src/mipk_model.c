#include "mipk_model.h"

#include "mipk_backend.h"

#include <stdlib.h>

struct mipk_model {
    const mipk_backend_ops *backend_ops;
    void *backend;
};

mipk_status mipk_core_model_create(const mipk_pipeline_config *pipeline_config,
    const mipk_model_config *config,
    mipk_model **output)
{
    if (pipeline_config == NULL || config == NULL || output == NULL) {
        return MIPK_ERR_ARGUMENT;
    }
    *output = NULL;
    mipk_model *model = calloc(1, sizeof(*model));
    if (model == NULL) return MIPK_ERR_MEMORY;
    model->backend_ops = mipk_backend_find(config->backend_name);
    if (model->backend_ops == NULL) {
        free(model);
        return MIPK_ERR_UNSUPPORTED;
    }
    const mipk_backend_config backend_config = {
        .model_path = config->model_path,
        .model_size = config->model_size,
        .model_loader = config->model_loader,
        .model_loader_context = config->model_loader_context,
        .runtime_config_path = config->runtime_config_path,
        .kind = config->kind,
        .options_json = config->options_json,
        .input_format = pipeline_config->input_format,
        .input_width = pipeline_config->input_width,
        .input_height = pipeline_config->input_height,
        .input_size = pipeline_config->input_size,
        .classes = pipeline_config->classes,
        .class_count = pipeline_config->class_count,
        .result_capacity = pipeline_config->result_capacity,
        .platform = pipeline_config->platform,
    };
    mipk_status status = MIPK_OK;
    model->backend = model->backend_ops->create(&backend_config, &status);
    if (model->backend == NULL) {
        free(model);
        return status;
    }
    *output = model;
    return MIPK_OK;
}

void mipk_core_model_destroy(mipk_model *model)
{
    if (model == NULL) return;
    model->backend_ops->destroy(model->backend);
    free(model);
}

mipk_mutable_tensor *mipk_core_model_inputs(mipk_model *model)
{
    return model == NULL ? NULL : model->backend_ops->inputs(model->backend);
}

size_t mipk_core_model_input_count(const mipk_model *model)
{
    return model == NULL ? 0 : model->backend_ops->input_count(model->backend);
}

const mipk_tensor *mipk_core_model_outputs(const mipk_model *model)
{
    return model == NULL ? NULL : model->backend_ops->outputs(model->backend);
}

size_t mipk_core_model_output_count(const mipk_model *model)
{
    return model == NULL ? 0 : model->backend_ops->output_count(model->backend);
}

mipk_status mipk_core_model_execute(mipk_model *model,
    mipk_execution_context *execution)
{
    return model == NULL || execution == NULL ? MIPK_ERR_ARGUMENT :
        model->backend_ops->execute(model->backend, execution);
}
