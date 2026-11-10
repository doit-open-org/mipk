#ifndef MIPK_TASK_H
#define MIPK_TASK_H

/* Stage types shared by the manifest, the builder and module authors. */

#include "mipk_operator.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    MIPK_STAGE_OPERATOR = 0,
    MIPK_STAGE_MODEL,
    MIPK_STAGE_POSTPROCESS,
} mipk_stage_type;

typedef enum {
    MIPK_EXECUTION_ONCE = 0,
    MIPK_EXECUTION_PER_ITEM,
} mipk_execution_mode;

typedef struct {
    /* Port name (for example "image", "faces", or model tensor index "0"). */
    const char *port;
    /* "$input" or a stage output name. */
    const char *source;
    /* Select one tensor from a TENSORS value; false binds the complete value. */
    bool select_tensor;
    size_t tensor_index;
} mipk_stage_binding;

typedef struct {
    /* Stable manifest identity; independent from the graph output name. */
    const char *id;
    mipk_stage_type type;
    const char *name;
    const char *backend_name;
    const char *model_kind;
    const char *options_json;
    mipk_stage_binding *inputs;
    size_t input_count;
    const char *output;
    size_t model_index;
    mipk_execution_mode execution;
    size_t max_items;
    uint32_t label_set_id;
} mipk_stage_config;


#ifdef __cplusplus
}
#endif
#endif
