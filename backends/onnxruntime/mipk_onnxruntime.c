#define _POSIX_C_SOURCE 200809L

#include "mipk_module.h"

#include "cJSON.h"
#include "onnxruntime_c_api.h"

#include <pthread.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef mipk_aligned_free
#define mipk_aligned_free free
#endif

/*
 * ONNX Runtime backend.
 *
 * One OrtEnv is shared by every model in the process and released with the
 * last one.  Each model owns its session, its input and output buffers, and
 * OrtValues that wrap those buffers, so a run copies nothing: the runtime
 * writes model inputs straight into the buffers and reads outputs straight
 * out of them.  Shapes therefore have to be static; a model with a symbolic
 * dimension other than a leading batch of one is refused at create time.
 *
 * models[] options understood, all optional:
 *   "threads": intra-op thread count (default: ORT's choice)
 *   "optimization": "none" | "basic" | "extended" | "all" (default "all")
 */

typedef struct {
    OrtSession *session;
    OrtMemoryInfo *memory_info;
    /* Buffers come from here when the pipeline was opened with a platform. */
    const mipk_platform_context *platform;
    mipk_mutable_tensor *inputs;
    mipk_mutable_tensor *outputs;
    mipk_buffer *buffers;
    OrtValue **input_values;
    OrtValue **output_values;
    char **input_names;
    char **output_names;
    size_t input_count;
    size_t output_count;
    bool env_acquired;
} ort_model;

static const OrtApi *g_ort;
static OrtEnv *g_env;
static size_t g_env_references;
static pthread_mutex_t g_env_lock = PTHREAD_MUTEX_INITIALIZER;

static bool check(OrtStatus *status, const char *what)
{
    if (status == NULL) return true;
    mipk_diag("onnxruntime: %s: %s", what, g_ort->GetErrorMessage(status));
    g_ort->ReleaseStatus(status);
    return false;
}

static mipk_status env_acquire(void)
{
    mipk_status status = MIPK_OK;
    if (pthread_mutex_lock(&g_env_lock) != 0) return MIPK_ERR_RUNTIME;
    if (g_ort == NULL) g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (g_ort == NULL) {
        status = MIPK_ERR_RUNTIME;
    } else if (g_env_references == 0 &&
        !check(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "mipk", &g_env),
            "CreateEnv")) {
        status = MIPK_ERR_RUNTIME;
    }
    if (status == MIPK_OK) ++g_env_references;
    (void)pthread_mutex_unlock(&g_env_lock);
    return status;
}

static void env_release(void)
{
    if (pthread_mutex_lock(&g_env_lock) != 0) return;
    if (g_env_references != 0 && --g_env_references == 0) {
        g_ort->ReleaseEnv(g_env);
        g_env = NULL;
    }
    (void)pthread_mutex_unlock(&g_env_lock);
}

static int32_t map_data_type(ONNXTensorElementDataType type, size_t *size)
{
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: *size = 4; return MIPK_DTYPE_FLOAT32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: *size = 2; return MIPK_DTYPE_FLOAT16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: *size = 1; return MIPK_DTYPE_INT8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: *size = 1; return MIPK_DTYPE_UINT8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: *size = 2; return MIPK_DTYPE_INT16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: *size = 2; return MIPK_DTYPE_UINT16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: *size = 4; return MIPK_DTYPE_INT32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: *size = 4; return MIPK_DTYPE_UINT32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: *size = 8; return MIPK_DTYPE_INT64;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: *size = 8; return MIPK_DTYPE_UINT64;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: *size = 8; return MIPK_DTYPE_DOUBLE;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: *size = 1; return MIPK_DTYPE_BOOL;
        default: *size = 0; return -1;
    }
}

static ONNXTensorElementDataType ort_element_type(int32_t data_type)
{
    switch (data_type) {
        case MIPK_DTYPE_FLOAT32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        case MIPK_DTYPE_FLOAT16: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
        case MIPK_DTYPE_INT8: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
        case MIPK_DTYPE_UINT8: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
        case MIPK_DTYPE_INT16: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
        case MIPK_DTYPE_UINT16: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
        case MIPK_DTYPE_INT32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
        case MIPK_DTYPE_UINT32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
        case MIPK_DTYPE_INT64: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
        case MIPK_DTYPE_UINT64: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
        case MIPK_DTYPE_DOUBLE: return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
        case MIPK_DTYPE_BOOL: return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
        default: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
}

static void free_names(char **names, size_t count)
{
    if (names == NULL) return;
    OrtAllocator *allocator = NULL;
    if (check(g_ort->GetAllocatorWithDefaultOptions(&allocator), "allocator")) {
        for (size_t index = 0; index < count; ++index) {
            if (names[index] != NULL) {
                OrtStatus *freed = g_ort->AllocatorFree(allocator,
                    names[index]);
                if (freed != NULL) g_ort->ReleaseStatus(freed);
            }
        }
    }
    free(names);
}

static void model_close(ort_model *model)
{
    const size_t total = model->input_count + model->output_count;
    for (size_t index = 0; index < model->input_count; ++index) {
        if (model->input_values != NULL && model->input_values[index] != NULL) {
            g_ort->ReleaseValue(model->input_values[index]);
        }
    }
    for (size_t index = 0; index < model->output_count; ++index) {
        if (model->output_values != NULL && model->output_values[index] != NULL) {
            g_ort->ReleaseValue(model->output_values[index]);
        }
    }
    free(model->input_values);
    free(model->output_values);
    for (size_t index = 0; index < total; ++index) {
        if (model->buffers == NULL ||
            model->buffers[index].desc.virtual_address == NULL) continue;
        if (model->platform != NULL) {
            model->platform->buffer_free(model->platform->context,
                &model->buffers[index]);
        } else {
            mipk_aligned_free(model->buffers[index].desc.virtual_address);
        }
    }
    free(model->buffers);
    free(model->inputs);
    free(model->outputs);
    free_names(model->input_names, model->input_count);
    free_names(model->output_names, model->output_count);
    if (model->memory_info != NULL) g_ort->ReleaseMemoryInfo(model->memory_info);
    if (model->session != NULL) g_ort->ReleaseSession(model->session);
    if (model->env_acquired) env_release();
}

static mipk_status describe_tensor(OrtSession *session, size_t index,
    bool input, mipk_mutable_tensor *tensor, mipk_buffer *buffer,
    OrtValue **value, char **name, OrtMemoryInfo *memory_info,
    const mipk_platform_context *platform)
{
    OrtTypeInfo *type_info = NULL;
    const OrtTensorTypeAndShapeInfo *shape = NULL;
    ONNXTensorElementDataType element_type;
    size_t dim_count = 0;
    int64_t dims[MIPK_MAX_TENSOR_DIMS];
    OrtAllocator *allocator = NULL;
    if (!check(g_ort->GetAllocatorWithDefaultOptions(&allocator), "allocator") ||
        !check(input ? g_ort->SessionGetInputName(session, index, allocator, name) :
            g_ort->SessionGetOutputName(session, index, allocator, name), "name") ||
        !check(input ? g_ort->SessionGetInputTypeInfo(session, index, &type_info) :
            g_ort->SessionGetOutputTypeInfo(session, index, &type_info), "type")) {
        return MIPK_ERR_MODEL;
    }
    mipk_status status = MIPK_ERR_MODEL;
    if (!check(g_ort->CastTypeInfoToTensorInfo(type_info, &shape), "tensor info") ||
        shape == NULL ||
        !check(g_ort->GetTensorElementType(shape, &element_type), "element type") ||
        !check(g_ort->GetDimensionsCount(shape, &dim_count), "rank") ||
        dim_count == 0 || dim_count > MIPK_MAX_TENSOR_DIMS ||
        !check(g_ort->GetDimensions(shape, dims, dim_count), "dims")) {
        goto done;
    }
    size_t element_size = 0;
    tensor->data_type = map_data_type(element_type, &element_size);
    if (element_size == 0) {
        mipk_diag("onnxruntime: %s '%s' has an unsupported element type",
            input ? "input" : "output", *name);
        goto done;
    }
    size_t count = 1;
    for (size_t dim = 0; dim < dim_count; ++dim) {
        if (dims[dim] <= 0) {
            /* A symbolic batch is taken as one; anything else is dynamic. */
            if (dim == 0) dims[dim] = 1;
            else {
                mipk_diag("onnxruntime: %s '%s' has a dynamic dimension %zu; "
                    "export the model with static shapes",
                    input ? "input" : "output", *name, dim);
                goto done;
            }
        }
        if (count > SIZE_MAX / (size_t)dims[dim]) goto done;
        count *= (size_t)dims[dim];
        tensor->dims[dim] = dims[dim];
    }
    tensor->dim_count = dim_count;
    tensor->format = dim_count == 4 ? MIPK_LAYOUT_NCHW : MIPK_LAYOUT_ND;
    tensor->byte_size = count * element_size;
    tensor->strides[dim_count - 1U] = (size_t)dims[dim_count - 1U] * element_size;
    void *memory = NULL;
    if (platform != NULL) {
        if (platform->buffer_alloc(platform->context, tensor->byte_size, 64,
                MIPK_MEMORY_CPU, buffer) != MIPK_OK) {
            status = MIPK_ERR_MEMORY;
            goto done;
        }
        memory = buffer->desc.virtual_address;
    } else {
        if (posix_memalign(&memory, 64, tensor->byte_size) != 0) {
            status = MIPK_ERR_MEMORY;
            goto done;
        }
        *buffer = (mipk_buffer) {
            .desc = {
                .virtual_address = memory,
                .physical_address = MIPK_PHYSICAL_ADDRESS_INVALID,
                .size = tensor->byte_size,
                .alignment = 64,
                .memory_type = MIPK_MEMORY_CPU,
                .owner = MIPK_BUFFER_OWNER_BACKEND,
            },
            .sync_state = MIPK_BUFFER_SYNCED,
        };
    }
    memset(memory, 0, tensor->byte_size);
    tensor->buffer = buffer;
    if (!check(g_ort->CreateTensorWithDataAsOrtValue(memory_info, memory,
            tensor->byte_size, tensor->dims, dim_count,
            ort_element_type(tensor->data_type), value), "bind tensor")) {
        goto done;
    }
    status = MIPK_OK;
done:
    g_ort->ReleaseTypeInfo(type_info);
    return status;
}

static bool apply_session_options(OrtSessionOptions *options,
    const char *options_json)
{
    cJSON *json = options_json == NULL ? NULL : cJSON_Parse(options_json);
    const cJSON *threads = cJSON_GetObjectItemCaseSensitive(json, "threads");
    const cJSON *optimization = cJSON_GetObjectItemCaseSensitive(json,
        "optimization");
    bool valid = true;
    if (threads != NULL) {
        valid = cJSON_IsNumber(threads) && threads->valuedouble >= 1.0 &&
            threads->valuedouble <= 256.0 &&
            check(g_ort->SetIntraOpNumThreads(options,
                (int)threads->valuedouble), "threads");
    }
    GraphOptimizationLevel level = ORT_ENABLE_ALL;
    if (valid && optimization != NULL) {
        const char *text = cJSON_IsString(optimization) ?
            optimization->valuestring : NULL;
        if (text == NULL) valid = false;
        else if (strcmp(text, "none") == 0) level = ORT_DISABLE_ALL;
        else if (strcmp(text, "basic") == 0) level = ORT_ENABLE_BASIC;
        else if (strcmp(text, "extended") == 0) level = ORT_ENABLE_EXTENDED;
        else if (strcmp(text, "all") == 0) level = ORT_ENABLE_ALL;
        else valid = false;
    }
    valid = valid && check(g_ort->SetSessionGraphOptimizationLevel(options,
        level), "optimization") &&
        check(g_ort->SetSessionLogSeverityLevel(options, 3), "log level");
    cJSON_Delete(json);
    return valid;
}

/* onnxruntime takes ORTCHAR_T paths: char elsewhere, wchar_t on Windows,
 * where the UTF-8 manifest path is widened first. */
static bool create_session_from_path(const char *path,
    const OrtSessionOptions *options, OrtSession **session)
{
#ifdef _WIN32
    const int length = MultiByteToWideChar(CP_UTF8, 0, path, -1, NULL, 0);
    if (length <= 0) return false;
    wchar_t *wide = malloc((size_t)length * sizeof(*wide));
    if (wide == NULL) return false;
    (void)MultiByteToWideChar(CP_UTF8, 0, path, -1, wide, length);
    const bool created = check(g_ort->CreateSession(g_env, wide, options,
        session), "CreateSession");
    free(wide);
    return created;
#else
    return check(g_ort->CreateSession(g_env, path, options, session),
        "CreateSession");
#endif
}

static mipk_status create_session(ort_model *model,
    const mipk_backend_config *config)
{
    OrtSessionOptions *options = NULL;
    if (!check(g_ort->CreateSessionOptions(&options), "session options")) {
        return MIPK_ERR_RUNTIME;
    }
    mipk_status status = MIPK_OK;
    void *bytes = NULL;
    if (!apply_session_options(options, config->options_json)) {
        status = MIPK_ERR_MODEL;
    } else if (config->model_loader != NULL) {
        bytes = malloc(config->model_size);
        if (bytes == NULL) status = MIPK_ERR_MEMORY;
        else status = config->model_loader(config->model_loader_context,
            bytes, config->model_size);
        if (status == MIPK_OK && !check(g_ort->CreateSessionFromArray(g_env,
                bytes, config->model_size, options, &model->session),
                "CreateSessionFromArray")) {
            status = MIPK_ERR_MODEL;
        }
    } else if (!create_session_from_path(config->model_path, options,
            &model->session)) {
        status = MIPK_ERR_MODEL;
    }
    free(bytes);
    g_ort->ReleaseSessionOptions(options);
    return status;
}

static void *ort_create(const mipk_backend_config *config, mipk_status *status)
{
    if (status == NULL) return NULL;
    *status = MIPK_ERR_ARGUMENT;
    if (config == NULL ||
        ((config->model_path == NULL || *config->model_path == '\0') &&
            (config->model_loader == NULL || config->model_size == 0))) {
        return NULL;
    }
    ort_model *model = calloc(1, sizeof(*model));
    if (model == NULL) {
        *status = MIPK_ERR_MEMORY;
        return NULL;
    }
    model->platform = config->platform;
    *status = env_acquire();
    if (*status != MIPK_OK) goto failed;
    model->env_acquired = true;
    *status = create_session(model, config);
    if (*status != MIPK_OK) goto failed;
    *status = MIPK_ERR_MODEL;
    if (!check(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault,
            &model->memory_info), "memory info") ||
        !check(g_ort->SessionGetInputCount(model->session, &model->input_count),
            "input count") ||
        !check(g_ort->SessionGetOutputCount(model->session,
            &model->output_count), "output count") ||
        model->input_count == 0 || model->output_count == 0) {
        goto failed;
    }
    model->inputs = calloc(model->input_count, sizeof(*model->inputs));
    model->outputs = calloc(model->output_count, sizeof(*model->outputs));
    model->buffers = calloc(model->input_count + model->output_count,
        sizeof(*model->buffers));
    model->input_values = calloc(model->input_count, sizeof(*model->input_values));
    model->output_values = calloc(model->output_count, sizeof(*model->output_values));
    model->input_names = calloc(model->input_count, sizeof(*model->input_names));
    model->output_names = calloc(model->output_count, sizeof(*model->output_names));
    if (model->inputs == NULL || model->outputs == NULL ||
        model->buffers == NULL || model->input_values == NULL ||
        model->output_values == NULL || model->input_names == NULL ||
        model->output_names == NULL) {
        *status = MIPK_ERR_MEMORY;
        goto failed;
    }
    for (size_t index = 0; index < model->input_count; ++index) {
        *status = describe_tensor(model->session, index, true,
            &model->inputs[index], &model->buffers[index],
            &model->input_values[index], &model->input_names[index],
            model->memory_info, model->platform);
        if (*status != MIPK_OK) goto failed;
    }
    for (size_t index = 0; index < model->output_count; ++index) {
        *status = describe_tensor(model->session, index, false,
            &model->outputs[index], &model->buffers[model->input_count + index],
            &model->output_values[index], &model->output_names[index],
            model->memory_info, model->platform);
        if (*status != MIPK_OK) goto failed;
    }
    *status = MIPK_OK;
    return model;

failed:
    model_close(model);
    free(model);
    return NULL;
}

static void ort_destroy(void *private_data)
{
    ort_model *model = private_data;
    if (model == NULL) return;
    model_close(model);
    free(model);
}

static size_t ort_input_count(const void *model)
{
    return model == NULL ? 0 : ((const ort_model *)model)->input_count;
}

static size_t ort_output_count(const void *model)
{
    return model == NULL ? 0 : ((const ort_model *)model)->output_count;
}

static mipk_mutable_tensor *ort_inputs(void *model)
{
    return model == NULL ? NULL : ((ort_model *)model)->inputs;
}

static const mipk_tensor *ort_outputs(const void *model)
{
    return model == NULL ? NULL : ((const ort_model *)model)->outputs;
}

static mipk_status ort_execute(void *private_data, mipk_operator_io *io)
{
    (void)io;
    ort_model *model = private_data;
    if (model == NULL) return MIPK_ERR_ARGUMENT;
    return check(g_ort->Run(model->session, NULL,
        (const char *const *)model->input_names,
        (const OrtValue *const *)model->input_values, model->input_count,
        (const char *const *)model->output_names, model->output_count,
        model->output_values), "Run") ? MIPK_OK : MIPK_ERR_RUNTIME;
}

static const mipk_capability_descriptor g_capability =
    MIPK_BACKEND_CAPABILITY("onnxruntime", "2.0.0");

const mipk_backend_ops *mipk_onnxruntime_backend(void)
{
    static const mipk_backend_ops operations = {
        .capability = &g_capability,
        .name = "onnxruntime",
        .create = ort_create,
        .destroy = ort_destroy,
        .input_count = ort_input_count,
        .output_count = ort_output_count,
        .inputs = ort_inputs,
        .outputs = ort_outputs,
        .execute = ort_execute,
    };
    return &operations;
}

mipk_status mipk_onnxruntime_register(void)
{
    return mipk_backend_register(mipk_onnxruntime_backend());
}
