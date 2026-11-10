#define _POSIX_C_SOURCE 200809L

#include "mipk_module.h"
#include "mipk_svp_acl.h"

#include "svp_acl.h"
#include "svp_acl_mdl.h"
#include "svp_acl_rt.h"

#include "cJSON.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    bool loaded;
    uint32_t id;
    void *data;
    size_t size;
    svp_acl_mdl_desc *description;
    svp_acl_mdl_dataset *input_dataset;
    svp_acl_mdl_dataset *output_dataset;
    mipk_mutable_tensor *inputs;
    mipk_mutable_tensor *outputs;
    mipk_buffer *input_buffers;
    mipk_buffer *output_buffers;
    size_t input_count;
    size_t output_count;
    bool runtime_acquired;
    const mipk_platform_arbiter *arbiter;
} svp_acl_model;

typedef mipk_status (*model_loader_fn)(void *context, void *destination,
    size_t destination_size);

static pthread_mutex_t g_runtime_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t g_execute_lock = PTHREAD_MUTEX_INITIALIZER;
static size_t g_runtime_references;
static char *g_runtime_config;

static const char *normalized_runtime_config(const char *path)
{
    return path == NULL ? "" : path;
}

/* models[].runtime_config, if the manifest sets one. */
static char *runtime_config_from_options(const char *options_json)
{
    cJSON *json = options_json == NULL ? NULL : cJSON_Parse(options_json);
    const cJSON *item = cJSON_GetObjectItemCaseSensitive(json,
        "runtime_config");
    char *path = cJSON_IsString(item) && item->valuestring != NULL ?
        strdup(item->valuestring) : NULL;
    cJSON_Delete(json);
    return path;
}

static mipk_status runtime_acquire(const char *config_path)
{
    const char *requested_config = normalized_runtime_config(config_path);
    mipk_status status = MIPK_OK;
    if (pthread_mutex_lock(&g_runtime_lock) != 0) return MIPK_ERR_RUNTIME;
    if (g_runtime_references == 0) {
        if (svp_acl_init(requested_config) != SVP_ACL_SUCCESS ||
            svp_acl_rt_set_device(0) != SVP_ACL_SUCCESS) {
            (void)svp_acl_finalize();
            mipk_diag("svp_acl: svp_acl_init/set_device failed; is the NPU "
                "driver loaded and the config '%s' readable?",
                requested_config);
            status = MIPK_ERR_RUNTIME;
        } else {
            g_runtime_config = strdup(requested_config);
            if (g_runtime_config == NULL) {
                (void)svp_acl_rt_reset_device(0);
                (void)svp_acl_finalize();
                status = MIPK_ERR_MEMORY;
            }
        }
    } else if (g_runtime_config == NULL || strcmp(g_runtime_config, requested_config) != 0) {
        mipk_diag("svp_acl: runtime already initialised with config '%s', "
            "cannot switch to '%s'", g_runtime_config, requested_config);
        status = MIPK_ERR_RUNTIME;
    }
    if (status == MIPK_OK) ++g_runtime_references;
    (void)pthread_mutex_unlock(&g_runtime_lock);
    return status;
}

static void runtime_release(void)
{
    if (pthread_mutex_lock(&g_runtime_lock) != 0) return;
    if (g_runtime_references != 0 && --g_runtime_references == 0) {
        (void)svp_acl_rt_reset_device(0);
        (void)svp_acl_finalize();
        free(g_runtime_config);
        g_runtime_config = NULL;
    }
    (void)pthread_mutex_unlock(&g_runtime_lock);
}

static void destroy_dataset(svp_acl_mdl_dataset **dataset)
{
    if (dataset == NULL || *dataset == NULL) return;
    const size_t count = svp_acl_mdl_get_dataset_num_buffers(*dataset);
    for (size_t i = 0; i < count; ++i) {
        svp_acl_data_buffer *buffer = svp_acl_mdl_get_dataset_buffer(*dataset, i);
        if (buffer != NULL) (void)svp_acl_destroy_data_buffer(buffer);
    }
    (void)svp_acl_mdl_destroy_dataset(*dataset);
    *dataset = NULL;
}

static void model_close(svp_acl_model *model)
{
    if (model == NULL) return;
    destroy_dataset(&model->input_dataset);
    destroy_dataset(&model->output_dataset);
    for (size_t i = 0; i < model->input_count; ++i) {
        if (model->input_buffers != NULL &&
            model->input_buffers[i].desc.virtual_address != NULL) {
            (void)svp_acl_rt_free(
                model->input_buffers[i].desc.virtual_address);
        }
    }
    for (size_t i = 0; i < model->output_count; ++i) {
        if (model->output_buffers != NULL &&
            model->output_buffers[i].desc.virtual_address != NULL) {
            (void)svp_acl_rt_free(
                model->output_buffers[i].desc.virtual_address);
        }
    }
    free(model->inputs);
    free(model->outputs);
    free(model->input_buffers);
    free(model->output_buffers);
    model->inputs = NULL;
    model->outputs = NULL;
    model->input_buffers = NULL;
    model->output_buffers = NULL;
    model->input_count = 0;
    model->output_count = 0;
    if (model->description != NULL) {
        (void)svp_acl_mdl_destroy_desc(model->description);
        model->description = NULL;
    }
    if (model->loaded) {
        (void)svp_acl_mdl_unload(model->id);
        model->loaded = false;
    }
    if (model->data != NULL) {
        (void)svp_acl_rt_free(model->data);
        model->data = NULL;
    }
    model->size = 0;
}

static mipk_status read_model_file(const char *path, void **data, size_t *size)
{
    FILE *file = fopen(path, "rb");
    if (file == NULL) return MIPK_ERR_IO;
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return MIPK_ERR_IO;
    }
    const long length = ftell(file);
    if (length <= 0 || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return MIPK_ERR_IO;
    }
    if (svp_acl_rt_malloc_cached(data, (size_t)length,
            SVP_ACL_MEM_MALLOC_NORMAL_ONLY) != SVP_ACL_SUCCESS) {
        fclose(file);
        return MIPK_ERR_MEMORY;
    }
    if (fread(*data, 1, (size_t)length, file) != (size_t)length) {
        fclose(file);
        (void)svp_acl_rt_free(*data);
        *data = NULL;
        return MIPK_ERR_IO;
    }
    fclose(file);
    *size = (size_t)length;
    if (svp_acl_rt_mem_flush(*data, *size) != SVP_ACL_SUCCESS) {
        (void)svp_acl_rt_free(*data);
        *data = NULL;
        *size = 0;
        return MIPK_ERR_RUNTIME;
    }
    return MIPK_OK;
}

static mipk_status allocate_tensor_set(svp_acl_model *model, bool input,
    size_t count, mipk_mutable_tensor **tensors, mipk_buffer **buffers,
    svp_acl_mdl_dataset *dataset)
{
    *tensors = calloc(count, sizeof(**tensors));
    *buffers = calloc(count, sizeof(**buffers));
    if (count != 0 && (*tensors == NULL || *buffers == NULL)) {
        return MIPK_ERR_MEMORY;
    }
    for (size_t i = 0; i < count; ++i) {
        mipk_mutable_tensor *tensor = &(*tensors)[i];
        svp_acl_mdl_io_dims dimensions = {0};
        tensor->byte_size = input ?
            svp_acl_mdl_get_input_size_by_index(model->description, i) :
            svp_acl_mdl_get_output_size_by_index(model->description, i);
        const size_t row_stride = input ?
            svp_acl_mdl_get_input_default_stride(model->description, i) :
            svp_acl_mdl_get_output_default_stride(model->description, i);
        tensor->data_type = input ?
            svp_acl_mdl_get_input_data_type(model->description, i) :
            svp_acl_mdl_get_output_data_type(model->description, i);
        tensor->format = input ?
            svp_acl_mdl_get_input_format(model->description, i) :
            svp_acl_mdl_get_output_format(model->description, i);
        const svp_acl_error result = input ?
            svp_acl_mdl_get_input_dims(model->description, i, &dimensions) :
            svp_acl_mdl_get_output_dims(model->description, i, &dimensions);
        if (result != SVP_ACL_SUCCESS || dimensions.dim_count > MIPK_MAX_TENSOR_DIMS ||
            tensor->byte_size == 0 || svp_acl_rt_malloc_cached(
                &(*buffers)[i].desc.virtual_address, tensor->byte_size,
                SVP_ACL_MEM_MALLOC_NORMAL_ONLY) != SVP_ACL_SUCCESS) {
            mipk_diag("svp_acl: cannot describe or allocate %s %zu (%zu bytes)",
                input ? "input" : "output", i, tensor->byte_size);
            return MIPK_ERR_MODEL;
        }
        (*buffers)[i].desc = (mipk_buffer_desc) {
            .virtual_address = (*buffers)[i].desc.virtual_address,
            .physical_address = MIPK_PHYSICAL_ADDRESS_INVALID,
            .size = tensor->byte_size,
            .alignment = 1,
            .memory_type = MIPK_MEMORY_PHYS_CACHED,
            .owner = MIPK_BUFFER_OWNER_BACKEND,
        };
        (*buffers)[i].sync_state = MIPK_BUFFER_SYNCED;
        tensor->buffer = &(*buffers)[i];
        if (mipk_memory_zero(mipk_mutable_tensor_data(tensor),
                tensor->byte_size) != 0) return MIPK_ERR_RUNTIME;
        tensor->dim_count = dimensions.dim_count;
        if (mipk_memory_copy(tensor->dims, sizeof(tensor->dims), dimensions.dims,
                dimensions.dim_count * sizeof(dimensions.dims[0])) != 0) {
            return MIPK_ERR_RUNTIME;
        }
        if (tensor->dim_count != 0) {
            tensor->strides[tensor->dim_count - 1U] = row_stride;
        }
        svp_acl_data_buffer *buffer = svp_acl_create_data_buffer(
            mipk_mutable_tensor_data(tensor), tensor->byte_size, row_stride);
        if (buffer == NULL ||
            svp_acl_mdl_add_dataset_buffer(dataset, buffer) != SVP_ACL_SUCCESS) {
            if (buffer != NULL) (void)svp_acl_destroy_data_buffer(buffer);
            return MIPK_ERR_MODEL;
        }
    }
    return MIPK_OK;
}

static mipk_status model_open(svp_acl_model *model, const char *model_path,
    size_t model_size, model_loader_fn model_loader, void *loader_context)
{
    mipk_status status = MIPK_OK;
    if (model_loader != NULL) {
        if (model_size == 0 || svp_acl_rt_malloc_cached(&model->data, model_size,
                SVP_ACL_MEM_MALLOC_NORMAL_ONLY) != SVP_ACL_SUCCESS) {
            return model_size == 0 ? MIPK_ERR_ARGUMENT : MIPK_ERR_MEMORY;
        }
        model->size = model_size;
        status = model_loader(loader_context, model->data, model->size);
        if (status == MIPK_OK &&
            svp_acl_rt_mem_flush(model->data, model->size) != SVP_ACL_SUCCESS) {
            status = MIPK_ERR_RUNTIME;
        }
    } else {
        status = read_model_file(model_path, &model->data, &model->size);
    }
    if (status != MIPK_OK) return status;
    if (svp_acl_mdl_load_from_mem(model->data, model->size, &model->id) != SVP_ACL_SUCCESS) {
        mipk_diag("svp_acl: svp_acl_mdl_load_from_mem rejected the %zu byte "
            "model; is it an .om built for this SoC?", model->size);
        return MIPK_ERR_MODEL;
    }
    model->loaded = true;
    model->description = svp_acl_mdl_create_desc();
    if (model->description == NULL ||
        svp_acl_mdl_get_desc(model->description, model->id) != SVP_ACL_SUCCESS) {
        return MIPK_ERR_MODEL;
    }
    model->input_dataset = svp_acl_mdl_create_dataset();
    model->output_dataset = svp_acl_mdl_create_dataset();
    if (model->input_dataset == NULL || model->output_dataset == NULL) return MIPK_ERR_MODEL;
    model->input_count = svp_acl_mdl_get_num_inputs(model->description);
    model->output_count = svp_acl_mdl_get_num_outputs(model->description);
    if (model->input_count == 0 || model->output_count == 0) return MIPK_ERR_MODEL;
    status = allocate_tensor_set(model, true, model->input_count,
        &model->inputs, &model->input_buffers, model->input_dataset);
    if (status == MIPK_OK) {
        status = allocate_tensor_set(model, false, model->output_count,
            &model->outputs, &model->output_buffers, model->output_dataset);
    }
    return status;
}

static mipk_status model_execute(svp_acl_model *model)
{
    mipk_status status = MIPK_OK;
    bool locked = false;
    if (model->arbiter != NULL) {
        status = model->arbiter->lock(model->arbiter->context);
        locked = status == MIPK_OK;
    } else {
        locked = pthread_mutex_lock(&g_execute_lock) == 0;
        status = locked ? MIPK_OK : MIPK_ERR_RUNTIME;
    }
    if (status != MIPK_OK) return status;
    for (size_t i = 0; i < model->input_count; ++i) {
        if (svp_acl_rt_mem_flush(mipk_mutable_tensor_data(&model->inputs[i]),
                model->inputs[i].byte_size) != SVP_ACL_SUCCESS) {
            status = MIPK_ERR_RUNTIME;
            goto done;
        }
    }
    if (svp_acl_mdl_execute(model->id, model->input_dataset,
            model->output_dataset) != SVP_ACL_SUCCESS) {
        status = MIPK_ERR_RUNTIME;
        goto done;
    }
    for (size_t i = 0; i < model->output_count; ++i) {
        if (svp_acl_rt_mem_invalidate(
                mipk_mutable_tensor_data(&model->outputs[i]),
                model->outputs[i].byte_size) != SVP_ACL_SUCCESS) {
            status = MIPK_ERR_RUNTIME;
            goto done;
        }
    }
done:
    if (model->arbiter != NULL) model->arbiter->unlock(model->arbiter->context);
    else if (locked) (void)pthread_mutex_unlock(&g_execute_lock);
    return status;
}

static void *svp_acl_create(const mipk_backend_config *config,
    mipk_status *status)
{
    if (status == NULL || config == NULL ||
        ((config->model_path == NULL || *config->model_path == '\0') &&
            (config->model_loader == NULL || config->model_size == 0)) ||
        (config->model_path != NULL && config->model_loader != NULL)) {
        if (status != NULL) *status = MIPK_ERR_ARGUMENT;
        return NULL;
    }
    svp_acl_model *model = calloc(1, sizeof(*model));
    if (model == NULL) {
        *status = MIPK_ERR_MEMORY;
        return NULL;
    }
    if (config->platform != NULL && config->platform->shared_arbiter != NULL) {
        const mipk_platform_arbiter *arbiter =
            config->platform->shared_arbiter;
        if (arbiter->struct_size != sizeof(*arbiter) ||
            arbiter->context == NULL || arbiter->lock == NULL ||
            arbiter->unlock == NULL) {
            *status = MIPK_ERR_ARGUMENT;
            free(model);
            return NULL;
        }
        model->arbiter = arbiter;
    }
    char *runtime_config = runtime_config_from_options(config->options_json);
    *status = runtime_acquire(runtime_config != NULL ? runtime_config :
        config->runtime_config_path);
    free(runtime_config);
    if (*status != MIPK_OK) goto failed;
    model->runtime_acquired = true;
    *status = model_open(model, config->model_path, config->model_size,
        config->model_loader, config->model_loader_context);
    if (*status != MIPK_OK) goto failed;
    return model;

failed:
    model_close(model);
    if (model->runtime_acquired) runtime_release();
    free(model);
    return NULL;
}

static void svp_acl_destroy(void *private_data)
{
    svp_acl_model *model = private_data;
    if (model == NULL) return;
    model_close(model);
    if (model->runtime_acquired) runtime_release();
    free(model);
}

static size_t svp_acl_input_count(const void *private_data)
{
    const svp_acl_model *model = private_data;
    return model == NULL ? 0 : model->input_count;
}

static size_t svp_acl_output_count(const void *private_data)
{
    const svp_acl_model *model = private_data;
    return model == NULL ? 0 : model->output_count;
}

static mipk_mutable_tensor *svp_acl_inputs(void *private_data)
{
    svp_acl_model *model = private_data;
    return model == NULL ? NULL : model->inputs;
}

static const mipk_tensor *svp_acl_outputs(const void *private_data)
{
    const svp_acl_model *model = private_data;
    return model == NULL ? NULL : model->outputs;
}

static mipk_status svp_acl_execute(void *private_data, mipk_operator_io *io)
{
    (void)io;
    return private_data == NULL ? MIPK_ERR_ARGUMENT :
        model_execute(private_data);
}

static const mipk_capability_descriptor g_capability =
    MIPK_BACKEND_CAPABILITY("svp_acl", "2.0.0");

const mipk_backend_ops *mipk_svp_acl_backend(void)
{
    static const mipk_backend_ops operations = {
        .capability = &g_capability,
        .name = "svp_acl",
        .create = svp_acl_create,
        .destroy = svp_acl_destroy,
        .input_count = svp_acl_input_count,
        .output_count = svp_acl_output_count,
        .inputs = svp_acl_inputs,
        .outputs = svp_acl_outputs,
        .execute = svp_acl_execute,
    };
    return &operations;
}

mipk_status mipk_svp_acl_register(void)
{
    return mipk_backend_register(mipk_svp_acl_backend());
}
