#ifndef MIPK_BACKEND_H
#define MIPK_BACKEND_H

#include "mipk.h"
#include "mipk_capability.h"
#include "mipk_operator.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * A backend loads one model file and runs it.  The runtime binds pipeline
 * values to the backend's input tensors, calls execute, and reads the output
 * tensors back.  Tensor storage is owned by the backend for the lifetime of
 * the model.
 */
typedef struct {
    /* Either a path on disk, or a loader that copies the bytes on demand. */
    const char *model_path;
    size_t model_size;
    mipk_status (*model_loader)(void *context, void *destination,
        size_t destination_size);
    void *model_loader_context;
    const char *runtime_config_path;
    const char *kind;
    /* The manifest's models[] entry, for backend-specific options. */
    const char *options_json;
    mipk_data_format input_format;
    uint32_t input_width;
    uint32_t input_height;
    size_t input_size;
    const mipk_postprocess_class *classes;
    size_t class_count;
    size_t result_capacity;
    const mipk_platform_context *platform;
} mipk_backend_config;

typedef struct mipk_backend_ops {
    const mipk_capability_descriptor *capability;
    const char *name;
    void *(*create)(const mipk_backend_config *config, mipk_status *status);
    void (*destroy)(void *model);
    size_t (*input_count)(const void *model);
    size_t (*output_count)(const void *model);
    mipk_mutable_tensor *(*inputs)(void *model);
    const mipk_tensor *(*outputs)(const void *model);
    mipk_status (*execute)(void *model, mipk_operator_io *io);
} mipk_backend_ops;

/* Register static-lifetime backend operations before opening a pipeline. */
mipk_status mipk_backend_register(const mipk_backend_ops *ops);
const mipk_backend_ops *mipk_backend_find(const char *name);

#ifdef __cplusplus
}
#endif
#endif
