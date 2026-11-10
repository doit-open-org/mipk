#include "mipk_module.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * A backend that runs nothing.  The model "file" is a JSON description of
 * the tensors it should expose, so a pipeline's graph, memory plan and node
 * contracts can be exercised on any host without an inference engine:
 *
 *   {"inputs": [{"dtype": 0, "layout": 0, "dims": [1, 3, 640, 640]}],
 *    "outputs": [{"dtype": 0, "layout": 2, "dims": [1, 84, 8400]}]}
 *
 * Executing leaves the output buffers as they are, so a test can fill them
 * through the tensor pointers before running the pipeline.
 */

#include "cJSON.h"

typedef struct {
    mipk_mutable_tensor *inputs;
    mipk_mutable_tensor *outputs;
    mipk_buffer *buffers;
    size_t input_count;
    size_t output_count;
} null_model;

static size_t element_size(int32_t data_type)
{
    switch (data_type) {
        case MIPK_DTYPE_FLOAT32: case MIPK_DTYPE_INT32:
        case MIPK_DTYPE_UINT32: return 4;
        case MIPK_DTYPE_FLOAT16: case MIPK_DTYPE_INT16:
        case MIPK_DTYPE_UINT16: return 2;
        case MIPK_DTYPE_INT8: case MIPK_DTYPE_UINT8:
        case MIPK_DTYPE_BOOL: return 1;
        case MIPK_DTYPE_INT64: case MIPK_DTYPE_UINT64:
        case MIPK_DTYPE_DOUBLE: return 8;
        default: return 0;
    }
}

static bool describe_tensor(const cJSON *json, mipk_mutable_tensor *tensor,
    mipk_buffer *buffer)
{
    const cJSON *dtype = cJSON_GetObjectItemCaseSensitive(json, "dtype");
    const cJSON *layout = cJSON_GetObjectItemCaseSensitive(json, "layout");
    const cJSON *dims = cJSON_GetObjectItemCaseSensitive(json, "dims");
    if (!cJSON_IsArray(dims) || cJSON_GetArraySize(dims) == 0 ||
        cJSON_GetArraySize(dims) > (int)MIPK_MAX_TENSOR_DIMS) return false;
    tensor->data_type = cJSON_IsNumber(dtype) ? (int32_t)dtype->valuedouble :
        MIPK_DTYPE_FLOAT32;
    tensor->format = cJSON_IsNumber(layout) ? (int32_t)layout->valuedouble :
        MIPK_LAYOUT_ND;
    const size_t element = element_size(tensor->data_type);
    if (element == 0) return false;
    size_t count = 1;
    tensor->dim_count = (size_t)cJSON_GetArraySize(dims);
    for (size_t index = 0; index < tensor->dim_count; ++index) {
        const cJSON *dim = cJSON_GetArrayItem(dims, (int)index);
        if (!cJSON_IsNumber(dim) || dim->valuedouble < 1.0) return false;
        tensor->dims[index] = (int64_t)dim->valuedouble;
        if (count > SIZE_MAX / (size_t)tensor->dims[index]) return false;
        count *= (size_t)tensor->dims[index];
    }
    tensor->byte_size = count * element;
    tensor->strides[tensor->dim_count - 1U] =
        (size_t)tensor->dims[tensor->dim_count - 1U] * element;
    void *memory = calloc(1, tensor->byte_size);
    if (memory == NULL) return false;
    *buffer = (mipk_buffer) {
        .desc = {
            .virtual_address = memory,
            .physical_address = MIPK_PHYSICAL_ADDRESS_INVALID,
            .size = tensor->byte_size,
            .alignment = 16,
            .memory_type = MIPK_MEMORY_CPU,
            .owner = MIPK_BUFFER_OWNER_BACKEND,
        },
        .sync_state = MIPK_BUFFER_SYNCED,
    };
    tensor->buffer = buffer;
    return true;
}

static void null_destroy(void *private_data)
{
    null_model *model = private_data;
    if (model == NULL) return;
    for (size_t index = 0; index < model->input_count + model->output_count;
        ++index) {
        if (model->buffers != NULL) free(model->buffers[index].desc.virtual_address);
    }
    free(model->buffers);
    free(model->inputs);
    free(model->outputs);
    free(model);
}

static char *read_description(const mipk_backend_config *config)
{
    if (config->model_loader != NULL) {
        char *text = malloc(config->model_size + 1U);
        if (text == NULL) return NULL;
        if (config->model_loader(config->model_loader_context, text,
                config->model_size) != MIPK_OK) {
            free(text);
            return NULL;
        }
        text[config->model_size] = '\0';
        return text;
    }
    FILE *file = fopen(config->model_path, "rb");
    if (file == NULL) return NULL;
    char *text = NULL;
    if (fseek(file, 0, SEEK_END) == 0) {
        const long length = ftell(file);
        if (length > 0 && fseek(file, 0, SEEK_SET) == 0) {
            text = malloc((size_t)length + 1U);
            if (text != NULL && fread(text, 1, (size_t)length, file) !=
                    (size_t)length) {
                free(text);
                text = NULL;
            } else if (text != NULL) {
                text[length] = '\0';
            }
        }
    }
    fclose(file);
    return text;
}

static void *null_create(const mipk_backend_config *config,
    mipk_status *status)
{
    if (status == NULL) return NULL;
    *status = MIPK_ERR_ARGUMENT;
    if (config == NULL) return NULL;
    char *text = read_description(config);
    if (text == NULL) {
        *status = MIPK_ERR_IO;
        return NULL;
    }
    cJSON *json = cJSON_Parse(text);
    free(text);
    const cJSON *inputs = cJSON_GetObjectItemCaseSensitive(json, "inputs");
    const cJSON *outputs = cJSON_GetObjectItemCaseSensitive(json, "outputs");
    null_model *model = calloc(1, sizeof(*model));
    if (!cJSON_IsArray(inputs) || !cJSON_IsArray(outputs) ||
        cJSON_GetArraySize(inputs) == 0 || cJSON_GetArraySize(outputs) == 0 ||
        model == NULL) {
        cJSON_Delete(json);
        free(model);
        *status = model == NULL ? MIPK_ERR_MEMORY : MIPK_ERR_MODEL;
        return NULL;
    }
    model->input_count = (size_t)cJSON_GetArraySize(inputs);
    model->output_count = (size_t)cJSON_GetArraySize(outputs);
    model->inputs = calloc(model->input_count, sizeof(*model->inputs));
    model->outputs = calloc(model->output_count, sizeof(*model->outputs));
    model->buffers = calloc(model->input_count + model->output_count,
        sizeof(*model->buffers));
    bool valid = model->inputs != NULL && model->outputs != NULL &&
        model->buffers != NULL;
    for (size_t index = 0; valid && index < model->input_count; ++index) {
        valid = describe_tensor(cJSON_GetArrayItem(inputs, (int)index),
            &model->inputs[index], &model->buffers[index]);
    }
    for (size_t index = 0; valid && index < model->output_count; ++index) {
        valid = describe_tensor(cJSON_GetArrayItem(outputs, (int)index),
            &model->outputs[index], &model->buffers[model->input_count + index]);
    }
    cJSON_Delete(json);
    if (!valid) {
        null_destroy(model);
        *status = MIPK_ERR_MODEL;
        return NULL;
    }
    *status = MIPK_OK;
    return model;
}

static size_t null_input_count(const void *model)
{
    return model == NULL ? 0 : ((const null_model *)model)->input_count;
}

static size_t null_output_count(const void *model)
{
    return model == NULL ? 0 : ((const null_model *)model)->output_count;
}

static mipk_mutable_tensor *null_inputs(void *model)
{
    return model == NULL ? NULL : ((null_model *)model)->inputs;
}

static const mipk_tensor *null_outputs(const void *model)
{
    return model == NULL ? NULL : ((const null_model *)model)->outputs;
}

static mipk_status null_execute(void *model, mipk_operator_io *io)
{
    (void)io;
    return model == NULL ? MIPK_ERR_ARGUMENT : MIPK_OK;
}

static const mipk_capability_descriptor g_capability =
    MIPK_BACKEND_CAPABILITY("null", "2.0.0");

const mipk_backend_ops *mipk_null_backend(void)
{
    static const mipk_backend_ops operations = {
        .capability = &g_capability,
        .name = "null",
        .create = null_create,
        .destroy = null_destroy,
        .input_count = null_input_count,
        .output_count = null_output_count,
        .inputs = null_inputs,
        .outputs = null_outputs,
        .execute = null_execute,
    };
    return &operations;
}

mipk_status mipk_null_backend_register(void)
{
    return mipk_backend_register(mipk_null_backend());
}
