#ifndef MIPK_VALUE_H
#define MIPK_VALUE_H

#include "mipk_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    MIPK_VALUE_NONE = 0,
    MIPK_VALUE_INPUT,
    MIPK_VALUE_IMAGE_VIEW,
    MIPK_VALUE_TENSORS,
    MIPK_VALUE_RESULTS,
    MIPK_VALUE_COLLECTION,
    MIPK_VALUE_OPAQUE,
} mipk_value_type;

typedef enum {
    MIPK_DOMAIN_NONE = 0,
    MIPK_DOMAIN_IMAGE,
    MIPK_DOMAIN_AUDIO,
} mipk_value_domain;

typedef struct {
    const mipk_input *root_input;
    float region[8];
    size_t point_count;
} mipk_image_view;

typedef struct {
    uint64_t data_type_mask;
    uint64_t format_mask;
    uint32_t memory_type_mask;
    uint32_t min_alignment;
    uint32_t min_rank;
    uint32_t max_rank;
    const int64_t *dims;
    size_t dim_count;
    bool require_virtual_address;
    bool require_physical_address;
} mipk_tensor_constraint;

typedef enum {
    MIPK_LABEL_SET_ANY = 0,
    MIPK_LABEL_SET_NONE,
    MIPK_LABEL_SET_HOMOGENEOUS,
    MIPK_LABEL_SET_FIXED,
    MIPK_LABEL_SET_INHERIT,
    MIPK_LABEL_SET_MIXED,
} mipk_label_set_rule;

typedef struct {
    uint64_t result_type_mask;
    mipk_label_set_rule label_set_rule;
    const char *fixed_label_set;
    const char *label_set_from_port;
} mipk_result_constraint;

typedef enum {
    MIPK_LABEL_SET_STATE_NONE = 0,
    MIPK_LABEL_SET_STATE_FIXED,
    MIPK_LABEL_SET_STATE_MIXED,
} mipk_label_set_state;

typedef struct {
    uint64_t result_type_mask;
    mipk_label_set_state label_set_state;
    uint32_t label_set_id;
    size_t max_results;
} mipk_result_descriptor;

typedef struct {
    int32_t data_type;
    int32_t format;
    mipk_memory_type memory_type;
    uint32_t alignment;
    int64_t dims[MIPK_MAX_TENSOR_DIMS];
    size_t strides[MIPK_MAX_TENSOR_DIMS];
    size_t dim_count;
    size_t byte_size;
    bool require_virtual_address;
    bool require_physical_address;
} mipk_tensor_descriptor;

typedef struct {
    mipk_value_type value_type;
    mipk_value_domain domain;
    mipk_value_type collection_item_type;
    size_t max_items;
    const mipk_tensor_descriptor *tensors;
    size_t tensor_count;
    mipk_result_descriptor results;
} mipk_value_descriptor;

typedef struct {
    mipk_value_descriptor value;
    mipk_tensor_descriptor *tensor_storage;
    size_t tensor_capacity;
} mipk_mutable_value_descriptor;

typedef mipk_status (*mipk_schema_infer_fn)(const void *parsed_config,
    const mipk_value_descriptor *inputs, size_t input_count,
    mipk_mutable_value_descriptor *output);

typedef struct {
    const char *port;
    uint32_t accepted_value_mask;
    uint32_t accepted_collection_item_mask;
    uint32_t accepted_domain_mask;
    mipk_tensor_constraint tensor;
    mipk_result_constraint results;
    bool required;
} mipk_port_schema;

typedef struct {
    const mipk_port_schema *inputs;
    size_t input_count;
    /* Node-specific manifest fields; graph fields are supplied by Runtime. */
    const char *const *option_names;
    size_t option_count;
    mipk_value_type output_type;
    mipk_value_type collection_item_type;
    size_t max_items;
    mipk_value_domain output_domain;
    mipk_tensor_constraint output_tensor;
    mipk_result_constraint output_results;
    mipk_schema_infer_fn infer;
    bool may_return_again;
    /*
     * Name of the manifest option that bounds how many results the node
     * emits per run; NULL means "max_results".  The runtime sizes result
     * storage from it, so a node with a different name (say "top_k") must
     * say so here.
     */
    const char *result_bound_option;
} mipk_node_schema;

typedef struct mipk_value mipk_value;
typedef void (*mipk_value_release_fn)(void *context, mipk_value *value);

typedef struct {
    uint64_t item_id;
    uint64_t parent_item_id;
} mipk_item_metadata;

typedef struct {
    mipk_result *items;
    size_t item_count;
    size_t item_capacity;
    const float *scores;
    size_t score_count;
} mipk_result_value;

typedef struct {
    mipk_value *items;
    size_t count;
    size_t capacity;
    mipk_value_type item_type;
} mipk_collection_value;

struct mipk_value {
    mipk_value_type type;
    mipk_item_metadata metadata;
    union {
        const mipk_input *input;
        mipk_image_view image_view;
        struct {
            const mipk_tensor *items;
            size_t count;
        } tensors;
        mipk_result_value results;
        mipk_collection_value collection;
        struct {
            const char *type_name;
            void *data;
            size_t size;
        } opaque;
    } value;
    /* Optional owner hook. The executor calls it after the pipeline run. */
    mipk_value_release_fn release;
    void *release_context;
};

typedef struct {
    const char *port;
    const mipk_value *value;
} mipk_named_value;

const mipk_value *mipk_named_value_find(const mipk_named_value *values,
    size_t count, const char *port);
mipk_value_type mipk_value_descriptor_payload_type(
    const mipk_value_descriptor *descriptor);
bool mipk_value_descriptor_valid(const mipk_value_descriptor *descriptor);
mipk_status mipk_value_validate(const mipk_value *value);
const mipk_value *mipk_value_collection_at(const mipk_value *value,
    size_t index);

#ifdef __cplusplus
}
#endif
#endif
