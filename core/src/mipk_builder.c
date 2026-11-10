#include "mipk_pipeline_runtime_internal.h"

#include "mipk_config.h"
#include "mipk_tensor.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void mipk_pipeline_runtime_destroy(mipk_pipeline_runtime *runtime)
{
    if (runtime == NULL) return;
    for (size_t index = runtime->stage_count; index > 0; --index) {
        mipk_runtime_stage *stage = &runtime->stages[index - 1];
        if (stage->operator_ops != NULL) {
            stage->operator_ops->destroy(stage->context);
        } else if (stage->postprocess_ops != NULL) {
            stage->postprocess_ops->destroy(stage->context);
        }
        free(stage->prepared_tensors);
        free(stage->checkpoint);
        free(stage->collection_results);
        free(stage->collection_items);
        free(stage->active_options_json);
    }
    free(runtime->stages);
    for (size_t index = runtime->model_count; index > 0; --index) {
        mipk_core_model_destroy(runtime->models[index - 1]);
    }
    free(runtime->models);
    free(runtime->execution_inputs);
    free(runtime->execution_values);
    if (runtime->run_lock_initialized) {
        (void)pthread_mutex_destroy(&runtime->run_lock);
    }
    free(runtime);
}

static bool size_add(size_t left, size_t right, size_t *result)
{
    if (right > SIZE_MAX - left) return false;
    *result = left + right;
    return true;
}

static bool size_multiply(size_t left, size_t right, size_t *result)
{
    if (left != 0 && right > SIZE_MAX / left) return false;
    *result = left * right;
    return true;
}

static size_t stage_size_option(const mipk_stage_config *stage,
    const char *name, size_t fallback)
{
    size_t value = fallback;
    return stage != NULL && stage->options_json != NULL &&
        mipk_postprocess_option_get_size(stage->options_json, name, fallback,
            &value) == MIPK_OK ? value : fallback;
}

static const mipk_node_schema *stage_schema(const mipk_stage_config *stage);

static size_t stage_result_bound(const mipk_pipeline_runtime *runtime,
    const mipk_stage_config *stage)
{
    size_t bound = runtime->config.result_capacity;
    if (stage->type == MIPK_STAGE_POSTPROCESS ||
        stage->type == MIPK_STAGE_OPERATOR) {
        const mipk_node_schema *schema = stage_schema(stage);
        const char *option = schema != NULL &&
            schema->result_bound_option != NULL ?
            schema->result_bound_option : "max_results";
        bound = stage_size_option(stage, option, bound);
    }
    return bound == 0 || bound > runtime->config.result_capacity ?
        runtime->config.result_capacity : bound;
}

static bool stage_produces_results(const mipk_stage_config *stage)
{
    if (stage->type == MIPK_STAGE_POSTPROCESS) return true;
    if (stage->type != MIPK_STAGE_OPERATOR) return false;
    const mipk_operator_ops *ops = mipk_operator_find(stage->name);
    return ops != NULL && ops->schema != NULL &&
        (ops->schema->output_type == MIPK_VALUE_RESULTS ||
         ops->schema->collection_item_type == MIPK_VALUE_RESULTS);
}

static size_t stage_collection_capacity(const mipk_pipeline_runtime *runtime,
    size_t index)
{
    const mipk_stage_config *stage = &runtime->config.stages[index];
    if (stage->execution == MIPK_EXECUTION_PER_ITEM) {
        const bool continues = index + 1 < runtime->stage_count &&
            runtime->config.stages[index + 1].execution ==
                MIPK_EXECUTION_PER_ITEM &&
            runtime->config.stages[index + 1].max_items == stage->max_items;
        return continues ? 0 : stage->max_items;
    }
    if (stage->type == MIPK_STAGE_OPERATOR) {
        const mipk_operator_ops *ops = mipk_operator_find(stage->name);
        if (ops != NULL && ops->schema != NULL &&
            ops->schema->output_type == MIPK_VALUE_COLLECTION) {
            return stage_size_option(stage, "max_items", 0);
        }
    }
    return 0;
}

static mipk_status prepare_memory_plan(mipk_pipeline_runtime *runtime)
{
    mipk_memory_plan plan = {.struct_size = sizeof(plan)};
    size_t value = 0;
    plan.persistent_cpu_bytes = sizeof(*runtime);
    if (!size_multiply(runtime->stage_count, sizeof(*runtime->stages), &value) ||
        !size_add(plan.persistent_cpu_bytes, value,
            &plan.persistent_cpu_bytes) ||
        !size_multiply(runtime->model_count, sizeof(*runtime->models), &value) ||
        !size_add(plan.persistent_cpu_bytes, value,
            &plan.persistent_cpu_bytes) ||
        !size_multiply(runtime->stage_count + 1,
            sizeof(*runtime->execution_values), &plan.scratch_cpu_bytes) ||
        !size_multiply(8, sizeof(*runtime->execution_inputs), &value) ||
        !size_add(plan.scratch_cpu_bytes, value, &plan.scratch_cpu_bytes)) {
        return MIPK_ERR_CAPACITY;
    }
    for (size_t index = 0; index < runtime->model_count; ++index) {
        const mipk_model_config *config = &runtime->config.models[index];
        if (!size_add(plan.model_source_bytes, config->model_size,
                &plan.model_source_bytes)) return MIPK_ERR_CAPACITY;
        mipk_model *model = runtime->models[index];
        const mipk_tensor *inputs = mipk_core_model_inputs(model);
        const mipk_tensor *outputs = mipk_core_model_outputs(model);
        for (size_t tensor = 0;
            tensor < mipk_core_model_input_count(model); ++tensor) {
            if (!size_add(plan.model_io_bytes, inputs[tensor].byte_size,
                    &plan.model_io_bytes)) return MIPK_ERR_CAPACITY;
        }
        for (size_t tensor = 0;
            tensor < mipk_core_model_output_count(model); ++tensor) {
            if (!size_add(plan.model_io_bytes, outputs[tensor].byte_size,
                    &plan.model_io_bytes)) return MIPK_ERR_CAPACITY;
        }
    }
    size_t active_fusion_max_items = 0;
    for (size_t index = 0; index < runtime->stage_count; ++index) {
        mipk_runtime_stage *stage = &runtime->stages[index];
        const size_t capacity = stage_collection_capacity(runtime, index);
        if (stage->config->execution == MIPK_EXECUTION_PER_ITEM) {
            if (active_fusion_max_items != stage->config->max_items) {
                ++plan.fusion_group_count;
            }
            active_fusion_max_items = stage->config->max_items;
        } else {
            active_fusion_max_items = 0;
        }
        if (capacity == 0) continue;
        if (capacity > runtime->config.result_capacity &&
            stage->config->execution != MIPK_EXECUTION_PER_ITEM) {
            return MIPK_ERR_CAPACITY;
        }
        stage->collection_capacity = capacity;
        if (!size_multiply(capacity, sizeof(*stage->collection_items), &value) ||
            !size_add(plan.collection_bytes, value,
                &plan.collection_bytes)) return MIPK_ERR_CAPACITY;
        if (stage_produces_results(stage->config)) {
            stage->collection_results_per_item =
                stage_result_bound(runtime, stage->config);
            if (!size_multiply(capacity,
                    stage->collection_results_per_item,
                    &stage->collection_result_capacity) ||
                !size_multiply(stage->collection_result_capacity,
                    sizeof(*stage->collection_results), &value) ||
                !size_add(plan.collection_bytes, value,
                    &plan.collection_bytes)) return MIPK_ERR_CAPACITY;
        }
    }
    if (!size_add(plan.model_source_bytes, plan.model_io_bytes,
            &plan.peak_mmz_bytes) ||
        !size_add(plan.scratch_cpu_bytes, plan.collection_bytes,
            &plan.scratch_cpu_bytes) ||
        !size_add(plan.persistent_cpu_bytes, plan.scratch_cpu_bytes,
            &plan.peak_cpu_bytes) ||
        plan.scratch_cpu_bytes > MIPK_MAX_SCRATCH_BYTES ||
        plan.peak_mmz_bytes > MIPK_MAX_MMZ_BYTES) {
        return MIPK_ERR_CAPACITY;
    }
    for (size_t index = 0; index < runtime->stage_count; ++index) {
        mipk_runtime_stage *stage = &runtime->stages[index];
        if (stage->collection_capacity == 0) continue;
        stage->collection_items = calloc(stage->collection_capacity,
            sizeof(*stage->collection_items));
        if (stage->collection_items == NULL) return MIPK_ERR_MEMORY;
        if (stage->collection_result_capacity != 0) {
            stage->collection_results = calloc(
                stage->collection_result_capacity,
                sizeof(*stage->collection_results));
            if (stage->collection_results == NULL) return MIPK_ERR_MEMORY;
        }
    }
    runtime->memory_plan = plan;
    return MIPK_OK;
}

mipk_status mipk_pipeline_runtime_get_memory_plan(
    const mipk_pipeline_runtime *runtime, mipk_memory_plan *plan)
{
    if (runtime == NULL || plan == NULL ||
        plan->struct_size != sizeof(*plan)) return MIPK_ERR_ARGUMENT;
    *plan = runtime->memory_plan;
    return MIPK_OK;
}

static mipk_status create_models(mipk_pipeline_runtime *runtime)
{
    runtime->models = calloc(runtime->config.model_count,
        sizeof(*runtime->models));
    if (runtime->models == NULL) return MIPK_ERR_MEMORY;
    runtime->model_count = runtime->config.model_count;
    for (size_t index = 0; index < runtime->model_count; ++index) {
        mipk_status status = mipk_core_model_create(&runtime->config,
            &runtime->config.models[index], &runtime->models[index]);
        if (status != MIPK_OK) return status;
    }
    return MIPK_OK;
}

static mipk_postprocess_config node_config(const mipk_pipeline_config *config,
    uint32_t label_set_id)
{
    const mipk_class_config *classes = config->classes;
    size_t class_count = config->class_count;
    if (label_set_id != MIPK_INVALID_LABEL_SET_ID &&
        label_set_id < config->label_set_count) {
        classes = config->label_sets[label_set_id].classes;
        class_count = config->label_sets[label_set_id].class_count;
    }
    return (mipk_postprocess_config) {
        .classes = classes,
        .class_count = class_count,
        .result_capacity = config->result_capacity,
        .input_format = config->input_format,
        .source_width = config->input_width,
        .source_height = config->input_height,
        .input_width = config->model_input_width,
        .input_height = config->model_input_height,
        .input_size = config->model_input_size,
        .resized_width = config->resized_width,
        .resized_height = config->resized_height,
        .pad_x = config->pad_x,
        .pad_y = config->pad_y,
        .to_source = {
            config->to_source[0], config->to_source[1], config->to_source[2],
            config->to_source[3], config->to_source[4], config->to_source[5],
            config->to_source[6], config->to_source[7], config->to_source[8],
        },
        .input_sample_rate = config->input_sample_rate,
        .input_channels = config->input_channels,
        .input_frame_samples = config->input_frame_samples,
        .inference_interval_ms = config->inference_interval_ms,
        .label_set_id = label_set_id,
        .platform = config->platform,
    };
}

static bool source_descriptor(const mipk_pipeline_config *config,
    const mipk_value_descriptor descriptors[33], size_t stage_index,
    const char *source, mipk_value_descriptor *descriptor)
{
    if (strcmp(source, "$input") == 0) {
        *descriptor = descriptors[0];
        return true;
    }
    for (size_t index = 0; index < stage_index; ++index) {
        if (strcmp(config->stages[index].output, source) == 0) {
            *descriptor = descriptors[index + 1];
            return true;
        }
    }
    return false;
}

static bool binding_descriptor(const mipk_pipeline_config *config,
    const mipk_value_descriptor descriptors[33], size_t stage_index,
    const char *port, mipk_value_descriptor *descriptor)
{
    const mipk_stage_config *stage = &config->stages[stage_index];
    for (size_t input = 0; input < stage->input_count; ++input) {
        if (strcmp(stage->inputs[input].port, port) != 0) continue;
        return source_descriptor(config, descriptors, stage_index,
            stage->inputs[input].source, descriptor);
    }
    return false;
}

static const mipk_node_schema *stage_schema(const mipk_stage_config *stage)
{
    if (stage->type == MIPK_STAGE_OPERATOR) {
        const mipk_operator_ops *ops = mipk_operator_find(stage->name);
        return ops == NULL ? NULL : ops->schema;
    }
    if (stage->type == MIPK_STAGE_POSTPROCESS) {
        const mipk_postprocess_ops *ops = mipk_postprocess_find(stage->name);
        return ops == NULL ? NULL : ops->schema;
    }
    return NULL;
}

static bool result_constraint_accepts(const mipk_result_constraint *constraint,
    const mipk_result_descriptor *descriptor)
{
    if (constraint->result_type_mask != 0 &&
        (descriptor->result_type_mask == 0 ||
         (descriptor->result_type_mask & ~constraint->result_type_mask) != 0)) {
        return false;
    }
    if (constraint->label_set_rule == MIPK_LABEL_SET_NONE) {
        return descriptor->label_set_state == MIPK_LABEL_SET_STATE_NONE;
    }
    if (constraint->label_set_rule == MIPK_LABEL_SET_HOMOGENEOUS ||
        constraint->label_set_rule == MIPK_LABEL_SET_FIXED) {
        return descriptor->label_set_state == MIPK_LABEL_SET_STATE_FIXED;
    }
    return true;
}

static bool collection_constraint_accepts(const mipk_port_schema *port,
    const mipk_value_descriptor *descriptor)
{
    if (descriptor->value_type != MIPK_VALUE_COLLECTION) return true;
    const uint32_t mask = port->accepted_collection_item_mask;
    return mask == 0 || (mask & (UINT32_C(1) <<
        descriptor->collection_item_type)) != 0;
}

static void describe_tensor(const mipk_tensor *tensor,
    mipk_tensor_descriptor *descriptor)
{
    *descriptor = (mipk_tensor_descriptor) {
        .data_type = tensor->data_type,
        .format = tensor->format,
        .memory_type = tensor->buffer->desc.memory_type,
        .alignment = tensor->buffer->desc.alignment,
        .dim_count = tensor->dim_count,
        .byte_size = tensor->byte_size,
        .require_virtual_address =
            tensor->buffer->desc.virtual_address != NULL,
        .require_physical_address = tensor->buffer->desc.physical_address !=
            MIPK_PHYSICAL_ADDRESS_INVALID,
    };
    memcpy(descriptor->dims, tensor->dims,
        tensor->dim_count * sizeof(tensor->dims[0]));
    memcpy(descriptor->strides, tensor->strides,
        tensor->dim_count * sizeof(tensor->strides[0]));
}

static bool tensor_descriptors_equal(const mipk_tensor_descriptor *left,
    const mipk_tensor_descriptor *right)
{
    return left->data_type == right->data_type &&
        left->format == right->format && left->dim_count == right->dim_count &&
        left->byte_size == right->byte_size &&
        memcmp(left->dims, right->dims,
            left->dim_count * sizeof(left->dims[0])) == 0 &&
        memcmp(left->strides, right->strides,
            left->dim_count * sizeof(left->strides[0])) == 0;
}

static mipk_status set_tensor_contract(const mipk_tensor *tensors, size_t count,
    mipk_value_domain domain, mipk_value_descriptor *descriptor,
    mipk_tensor_descriptor **storage)
{
    if (tensors == NULL || count == 0 || storage == NULL || *storage != NULL) {
        return MIPK_ERR_MODEL;
    }
    mipk_tensor_descriptor *items = calloc(count, sizeof(*items));
    if (items == NULL) return MIPK_ERR_MEMORY;
    for (size_t index = 0; index < count; ++index) {
        if (!mipk_tensor_range_valid(&tensors[index]) ||
            tensors[index].buffer == NULL ||
            tensors[index].dim_count > MIPK_MAX_TENSOR_DIMS) {
            free(items);
            return MIPK_ERR_MODEL;
        }
        describe_tensor(&tensors[index], &items[index]);
    }
    descriptor->value_type = MIPK_VALUE_TENSORS;
    descriptor->domain = domain;
    descriptor->tensors = items;
    descriptor->tensor_count = count;
    *storage = items;
    return MIPK_OK;
}

static bool tensor_constraint_accepts(const mipk_tensor_constraint *constraint,
    const mipk_value_descriptor *descriptor)
{
    if (descriptor->value_type != MIPK_VALUE_TENSORS ||
        descriptor->tensor_count == 0 || descriptor->tensors == NULL) {
        return false;
    }
    for (size_t index = 0; index < descriptor->tensor_count; ++index) {
        const mipk_tensor_descriptor *tensor = &descriptor->tensors[index];
        if ((constraint->data_type_mask != 0 &&
                (tensor->data_type < 0 || tensor->data_type >= 64 ||
                 (constraint->data_type_mask &
                    (UINT64_C(1) << (uint32_t)tensor->data_type)) == 0)) ||
            (constraint->format_mask != 0 &&
                (tensor->format < 0 || tensor->format >= 64 ||
                 (constraint->format_mask &
                    (UINT64_C(1) << (uint32_t)tensor->format)) == 0)) ||
            (constraint->memory_type_mask != 0 &&
                (constraint->memory_type_mask &
                    (UINT32_C(1) << tensor->memory_type)) == 0) ||
            tensor->alignment < constraint->min_alignment ||
            tensor->dim_count < constraint->min_rank ||
            (constraint->max_rank != 0 &&
                tensor->dim_count > constraint->max_rank) ||
            (constraint->dim_count != 0 &&
                tensor->dim_count != constraint->dim_count) ||
            (constraint->require_virtual_address &&
                !tensor->require_virtual_address) ||
            (constraint->require_physical_address &&
                !tensor->require_physical_address)) return false;
        for (size_t dim = 0; dim < constraint->dim_count; ++dim) {
            if (constraint->dims == NULL ||
                (constraint->dims[dim] != -1 &&
                 constraint->dims[dim] != tensor->dims[dim])) return false;
        }
    }
    return true;
}

static bool scalar_descriptor(const mipk_value_descriptor *source,
    mipk_value_descriptor *scalar)
{
    if (source == NULL || scalar == NULL ||
        !mipk_value_descriptor_valid(source)) return false;
    *scalar = *source;
    if (source->value_type != MIPK_VALUE_COLLECTION) return true;
    scalar->value_type = source->collection_item_type;
    scalar->collection_item_type = MIPK_VALUE_NONE;
    scalar->max_items = 0;
    return mipk_value_descriptor_valid(scalar);
}

static bool per_item_sources_valid(const mipk_pipeline_config *config,
    const mipk_value_descriptor descriptors[33], size_t stage_index)
{
    const mipk_stage_config *stage = &config->stages[stage_index];
    if (stage->execution != MIPK_EXECUTION_PER_ITEM) return true;
    bool found_collection = false;
    for (size_t binding = 0; binding < stage->input_count; ++binding) {
        mipk_value_descriptor descriptor;
        if (!source_descriptor(config, descriptors, stage_index,
                stage->inputs[binding].source, &descriptor)) return false;
        if (descriptor.value_type != MIPK_VALUE_COLLECTION) continue;
        if (descriptor.max_items > stage->max_items) return false;
        found_collection = true;
    }
    return found_collection;
}

static mipk_status validate_schema_inputs(const mipk_pipeline_config *config,
    const mipk_value_descriptor descriptors[33], size_t stage_index,
    const mipk_node_schema *schema)
{
    const mipk_stage_config *stage = &config->stages[stage_index];
    for (size_t binding = 0; binding < stage->input_count; ++binding) {
        const mipk_port_schema *port = NULL;
        for (size_t index = 0; index < schema->input_count; ++index) {
            if (strcmp(stage->inputs[binding].port,
                    schema->inputs[index].port) == 0) {
                port = &schema->inputs[index];
                break;
            }
        }
        mipk_value_descriptor descriptor;
        if (port == NULL || !source_descriptor(config, descriptors, stage_index,
                stage->inputs[binding].source, &descriptor) ||
            (stage->execution == MIPK_EXECUTION_PER_ITEM &&
                !scalar_descriptor(&descriptor, &descriptor)) ||
            !mipk_value_descriptor_valid(&descriptor) ||
            (port->accepted_value_mask &
                (UINT32_C(1) << descriptor.value_type)) == 0 ||
            !collection_constraint_accepts(port, &descriptor) ||
            (port->accepted_domain_mask != 0 &&
                (port->accepted_domain_mask &
                    (UINT32_C(1) << descriptor.domain)) == 0) ||
            (descriptor.value_type == MIPK_VALUE_TENSORS &&
                !tensor_constraint_accepts(&port->tensor, &descriptor)) ||
            (mipk_value_descriptor_payload_type(&descriptor) ==
                MIPK_VALUE_RESULTS &&
                !result_constraint_accepts(&port->results,
                    &descriptor.results))) {
            return MIPK_ERR_MODEL;
        }
    }
    for (size_t index = 0; index < schema->input_count; ++index) {
        mipk_value_descriptor descriptor;
        if (schema->inputs[index].required &&
            !binding_descriptor(config, descriptors, stage_index,
                schema->inputs[index].port, &descriptor)) {
            return MIPK_ERR_MODEL;
        }
    }
    return MIPK_OK;
}

static mipk_status infer_schema_output(const mipk_pipeline_config *config,
    const mipk_value_descriptor descriptors[33], size_t stage_index,
    const mipk_node_schema *schema, mipk_value_descriptor *output)
{
    const mipk_stage_config *stage = &config->stages[stage_index];
    output->value_type = schema->output_type;
    output->collection_item_type = schema->collection_item_type;
    output->max_items = schema->max_items;
    if (output->value_type == MIPK_VALUE_COLLECTION &&
        output->max_items == 0) {
        output->max_items = stage_size_option(stage, "max_items", 0);
    }
    output->domain = schema->output_domain;
    if (output->domain == MIPK_DOMAIN_NONE && stage->input_count != 0) {
        mipk_value_descriptor input;
        if (!source_descriptor(config, descriptors, stage_index,
                stage->inputs[0].source, &input)) return MIPK_ERR_MODEL;
        output->domain = input.domain;
    }
    if (mipk_value_descriptor_payload_type(output) != MIPK_VALUE_RESULTS) {
        return mipk_value_descriptor_valid(output) ? MIPK_OK : MIPK_ERR_MODEL;
    }

    output->results.result_type_mask =
        schema->output_results.result_type_mask;
    output->results.max_results = config->result_capacity;
    switch (schema->output_results.label_set_rule) {
    case MIPK_LABEL_SET_FIXED:
        if (stage->label_set_id == MIPK_INVALID_LABEL_SET_ID ||
            stage->label_set_id >= config->label_set_count) {
            return MIPK_ERR_MODEL;
        }
        output->results.label_set_state = MIPK_LABEL_SET_STATE_FIXED;
        output->results.label_set_id = stage->label_set_id;
        break;
    case MIPK_LABEL_SET_INHERIT: {
        mipk_value_descriptor input;
        const char *port = schema->output_results.label_set_from_port;
        if (port == NULL || !binding_descriptor(config, descriptors,
                stage_index, port, &input) ||
            mipk_value_descriptor_payload_type(&input) !=
                MIPK_VALUE_RESULTS) return MIPK_ERR_MODEL;
        output->results.label_set_state = input.results.label_set_state;
        output->results.label_set_id = input.results.label_set_id;
        output->results.max_results = input.results.max_results;
        if (output->results.result_type_mask == 0) {
            output->results.result_type_mask = input.results.result_type_mask;
        }
        break;
    }
    case MIPK_LABEL_SET_MIXED:
        output->results.label_set_state = MIPK_LABEL_SET_STATE_MIXED;
        break;
    case MIPK_LABEL_SET_NONE:
        output->results.label_set_state = MIPK_LABEL_SET_STATE_NONE;
        break;
    case MIPK_LABEL_SET_ANY:
    case MIPK_LABEL_SET_HOMOGENEOUS:
    default:
        return MIPK_ERR_MODEL;
    }
    return mipk_value_descriptor_valid(output) ? MIPK_OK : MIPK_ERR_MODEL;
}

static mipk_status infer_operator_tensor_contract(mipk_pipeline_runtime *runtime,
    size_t stage_index, mipk_value_descriptor *descriptor,
    mipk_tensor_descriptor **storage)
{
    const mipk_pipeline_config *config = &runtime->config;
    const char *output = config->stages[stage_index].output;
    const mipk_tensor *selected = NULL;
    size_t selected_count = 0;
    for (size_t next = stage_index + 1; next < config->stage_count; ++next) {
        const mipk_stage_config *consumer = &config->stages[next];
        if (consumer->type != MIPK_STAGE_MODEL) continue;
        bool consumes = false;
        for (size_t binding = 0; binding < consumer->input_count; ++binding) {
            if (strcmp(consumer->inputs[binding].source, output) == 0) {
                consumes = true;
                break;
            }
        }
        if (!consumes || consumer->model_index >= runtime->model_count) continue;
        mipk_model *model = runtime->models[consumer->model_index];
        const mipk_tensor *inputs = mipk_core_model_inputs(model);
        const size_t input_count = mipk_core_model_input_count(model);
        if (selected == NULL) {
            selected = inputs;
            selected_count = input_count;
            continue;
        }
        if (input_count != selected_count) return MIPK_ERR_MODEL;
        for (size_t index = 0; index < input_count; ++index) {
            mipk_tensor_descriptor left;
            mipk_tensor_descriptor right;
            describe_tensor(&selected[index], &left);
            describe_tensor(&inputs[index], &right);
            if (!tensor_descriptors_equal(&left, &right)) return MIPK_ERR_MODEL;
        }
    }
    return set_tensor_contract(selected, selected_count, descriptor->domain,
        descriptor, storage);
}

static mipk_status validate_model_inputs(mipk_pipeline_runtime *runtime,
    const mipk_value_descriptor descriptors[33], size_t stage_index)
{
    const mipk_stage_config *stage = &runtime->config.stages[stage_index];
    if (stage->model_index >= runtime->model_count) return MIPK_ERR_MODEL;
    mipk_model *model = runtime->models[stage->model_index];
    const mipk_tensor *destinations = mipk_core_model_inputs(model);
    const size_t destination_count = mipk_core_model_input_count(model);
    if (destinations == NULL || destination_count == 0) return MIPK_ERR_MODEL;
    bool *covered = calloc(destination_count, sizeof(*covered));
    if (covered == NULL) return MIPK_ERR_MEMORY;
    mipk_status status = MIPK_OK;
    for (size_t binding = 0; binding < stage->input_count; ++binding) {
        const mipk_stage_binding *input = &stage->inputs[binding];
        errno = 0;
        char *end = NULL;
        const unsigned long parsed = strtoul(input->port, &end, 10);
        if (errno != 0 || end == input->port || *end != '\0' ||
            parsed >= destination_count) {
            status = MIPK_ERR_MODEL;
            break;
        }
        const size_t destination = (size_t)parsed;
        mipk_value_descriptor source;
        if (!source_descriptor(&runtime->config, descriptors, stage_index,
                input->source, &source) ||
            (stage->execution == MIPK_EXECUTION_PER_ITEM &&
                !scalar_descriptor(&source, &source)) ||
            source.value_type != MIPK_VALUE_TENSORS ||
            source.tensor_count == 0 || source.tensors == NULL) {
            status = MIPK_ERR_MODEL;
            break;
        }
        if (!input->select_tensor && stage->input_count == 1 &&
            destination == 0 && source.tensor_count == destination_count) {
            for (size_t index = 0; index < destination_count; ++index) {
                mipk_tensor_descriptor target;
                describe_tensor(&destinations[index], &target);
                if (!tensor_descriptors_equal(&source.tensors[index], &target)) {
                    status = MIPK_ERR_MODEL;
                    break;
                }
                covered[index] = true;
            }
            break;
        }
        const size_t source_index = input->select_tensor ? input->tensor_index : 0;
        if ((!input->select_tensor && source.tensor_count != 1) ||
            source_index >= source.tensor_count || covered[destination]) {
            status = MIPK_ERR_MODEL;
            break;
        }
        mipk_tensor_descriptor target;
        describe_tensor(&destinations[destination], &target);
        if (!tensor_descriptors_equal(&source.tensors[source_index], &target)) {
            status = MIPK_ERR_MODEL;
            break;
        }
        covered[destination] = true;
    }
    for (size_t index = 0; status == MIPK_OK && index < destination_count;
        ++index) {
        if (!covered[index]) status = MIPK_ERR_MODEL;
    }
    free(covered);
    return status;
}

static void free_tensor_contracts(mipk_tensor_descriptor *storage[33])
{
    for (size_t index = 0; index < 33; ++index) free(storage[index]);
}

static mipk_status validate_typed_graph(mipk_pipeline_runtime *runtime)
{
    const mipk_pipeline_config *config = &runtime->config;
    mipk_value_descriptor descriptors[33] = {0};
    mipk_tensor_descriptor *tensor_storage[33] = {0};
    descriptors[0].value_type = MIPK_VALUE_INPUT;
    descriptors[0].domain = config->input_format == MIPK_FORMAT_PCM_S16LE ?
        MIPK_DOMAIN_AUDIO : MIPK_DOMAIN_IMAGE;
    mipk_status status = MIPK_OK;
    for (size_t index = 0; status == MIPK_OK && index < config->stage_count;
        ++index) {
        const mipk_stage_config *stage = &config->stages[index];
        mipk_value_descriptor *output = &descriptors[index + 1];
        if (!per_item_sources_valid(config, descriptors, index)) {
            status = MIPK_ERR_MODEL;
            break;
        }
        if (stage->type == MIPK_STAGE_MODEL) {
            status = stage->input_count == 0 ? MIPK_ERR_MODEL :
                validate_model_inputs(runtime, descriptors, index);
            mipk_value_descriptor input;
            if (status == MIPK_OK && !source_descriptor(config, descriptors,
                    index, stage->inputs[0].source, &input)) {
                status = MIPK_ERR_MODEL;
            }
            if (status == MIPK_OK &&
                stage->execution == MIPK_EXECUTION_PER_ITEM &&
                !scalar_descriptor(&input, &input)) status = MIPK_ERR_MODEL;
            if (status == MIPK_OK) {
                status = set_tensor_contract(mipk_core_model_outputs(
                        runtime->models[stage->model_index]),
                    mipk_core_model_output_count(
                        runtime->models[stage->model_index]),
                    input.domain, output, &tensor_storage[index + 1]);
            }
        } else {
            const mipk_node_schema *schema = stage_schema(stage);
            if (schema == NULL || schema->inputs == NULL ||
                schema->input_count == 0) {
                status = MIPK_ERR_UNSUPPORTED;
            } else {
                status = validate_schema_inputs(config, descriptors, index,
                    schema);
            }
            if (status == MIPK_OK) {
                status = infer_schema_output(config, descriptors, index,
                    schema, output);
            }
            if (status == MIPK_OK &&
                mipk_value_descriptor_payload_type(output) ==
                    MIPK_VALUE_TENSORS &&
                output->tensor_count == 0) {
                status = infer_operator_tensor_contract(runtime, index, output,
                    &tensor_storage[index + 1]);
            }
        }
        if (status == MIPK_OK &&
            stage->execution == MIPK_EXECUTION_PER_ITEM) {
            const mipk_value_type item_type = output->value_type;
            if (item_type == MIPK_VALUE_COLLECTION ||
                item_type == MIPK_VALUE_NONE) {
                status = MIPK_ERR_MODEL;
            } else {
                output->value_type = MIPK_VALUE_COLLECTION;
                output->collection_item_type = item_type;
                output->max_items = stage->max_items;
            }
        }
        if (status == MIPK_OK && (!mipk_value_descriptor_valid(output) ||
            (mipk_value_descriptor_payload_type(output) ==
                MIPK_VALUE_TENSORS &&
                output->tensor_count == 0))) status = MIPK_ERR_MODEL;
    }
    bool found_output = false;
    for (size_t index = 0; status == MIPK_OK && index < config->stage_count;
        ++index) {
        if (strcmp(config->stages[index].output, config->output) != 0) continue;
        found_output = true;
        const mipk_value_descriptor *output = &descriptors[index + 1];
        if (mipk_value_descriptor_payload_type(output) !=
                MIPK_VALUE_RESULTS ||
            output->results.result_type_mask == 0 ||
            (output->results.result_type_mask & ~config->result_type_mask) != 0) {
            status = MIPK_ERR_MODEL;
        }
    }
    if (status == MIPK_OK && !found_output) status = MIPK_ERR_MODEL;
    free_tensor_contracts(tensor_storage);
    return status;
}

static mipk_value_type stage_scalar_output_type(
    const mipk_stage_config *stage)
{
    if (stage->type == MIPK_STAGE_MODEL) return MIPK_VALUE_TENSORS;
    if (stage->type == MIPK_STAGE_POSTPROCESS) return MIPK_VALUE_RESULTS;
    const mipk_operator_ops *ops = mipk_operator_find(stage->name);
    return ops == NULL || ops->schema == NULL ? MIPK_VALUE_NONE :
        ops->schema->output_type;
}

static size_t output_consumer_count(const mipk_pipeline_config *config,
    size_t producer, bool *next_consumes)
{
    size_t count = 0;
    *next_consumes = false;
    const char *output = config->stages[producer].output;
    for (size_t stage_index = producer + 1;
        stage_index < config->stage_count; ++stage_index) {
        const mipk_stage_config *stage = &config->stages[stage_index];
        for (size_t binding = 0; binding < stage->input_count; ++binding) {
            if (strcmp(stage->inputs[binding].source, output) != 0) continue;
            ++count;
            if (stage_index == producer + 1) *next_consumes = true;
        }
    }
    return count;
}

static mipk_status validate_fusion_groups(const mipk_pipeline_config *config)
{
    for (size_t index = 0; index < config->stage_count; ++index) {
        const mipk_stage_config *stage = &config->stages[index];
        if (stage->execution != MIPK_EXECUTION_PER_ITEM) continue;
        const bool continues = index + 1 < config->stage_count &&
            config->stages[index + 1].execution ==
                MIPK_EXECUTION_PER_ITEM &&
            config->stages[index + 1].max_items == stage->max_items;
        bool next_consumes = false;
        const size_t consumers = output_consumer_count(config, index,
            &next_consumes);
        if (continues) {
            if (!next_consumes || consumers != 1) return MIPK_ERR_MODEL;
        } else if (stage_scalar_output_type(stage) == MIPK_VALUE_TENSORS) {
            return MIPK_ERR_UNSUPPORTED;
        }
    }
    return MIPK_OK;
}

static mipk_status create_stage(mipk_pipeline_runtime *runtime, size_t index)
{
    const mipk_stage_config *config = &runtime->config.stages[index];
    mipk_runtime_stage *stage = &runtime->stages[index];
    stage->config = config;
    stage->type = config->type;
    if (config->type == MIPK_STAGE_MODEL) {
        if (config->model_index >= runtime->model_count) {
            return MIPK_ERR_MODEL;
        }
        stage->model = runtime->models[config->model_index];
        const size_t input_count = mipk_core_model_input_count(stage->model);
        for (size_t input = 0; input < config->input_count; ++input) {
            errno = 0;
            char *end = NULL;
            const unsigned long tensor = strtoul(config->inputs[input].port,
                &end, 10);
            if (errno != 0 || end == config->inputs[input].port ||
                *end != '\0' || tensor >= input_count) return MIPK_ERR_MODEL;
        }
        return MIPK_OK;
    }

    /* Resolve tensor contracts from explicit graph edges, not stage proximity. */
    if (config->type == MIPK_STAGE_OPERATOR) {
        for (size_t next = index + 1; next < runtime->config.stage_count;
            ++next) {
            const mipk_stage_config *consumer = &runtime->config.stages[next];
            if (consumer->type != MIPK_STAGE_MODEL) continue;
            for (size_t input = 0; input < consumer->input_count; ++input) {
                if (strcmp(consumer->inputs[input].source, config->output) == 0) {
                    stage->model = runtime->models[consumer->model_index];
                    break;
                }
            }
            if (stage->model != NULL) break;
        }
    } else {
        for (size_t input = 0; input < config->input_count; ++input) {
            for (size_t previous = index; previous > 0; --previous) {
                const mipk_stage_config *producer =
                    &runtime->config.stages[previous - 1];
                if (producer->type == MIPK_STAGE_MODEL &&
                    strcmp(producer->output, config->inputs[input].source) == 0) {
                    stage->model = runtime->models[producer->model_index];
                    break;
                }
            }
            if (stage->model != NULL) break;
        }
    }

    stage->node_config = node_config(&runtime->config, config->label_set_id);
    if (config->type == MIPK_STAGE_OPERATOR && stage->model != NULL &&
        strcmp(config->name, "image_preprocess") == 0) {
        const size_t width = stage_size_option(config, "target_width",
            stage->node_config.input_width);
        const size_t height = stage_size_option(config, "target_height",
            stage->node_config.input_height);
        const mipk_mutable_tensor *model_inputs =
            mipk_core_model_inputs(stage->model);
        if (width == 0 || height == 0 || width > UINT32_MAX ||
            height > UINT32_MAX || model_inputs == NULL ||
            mipk_core_model_input_count(stage->model) == 0) {
            return MIPK_ERR_MODEL;
        }
        const bool manifest_geometry =
            mipk_preprocess_keeps_manifest_geometry(config->execution, width,
                height, runtime->config.model_input_width,
                runtime->config.model_input_height);
        stage->node_config.input_width = (uint32_t)width;
        stage->node_config.input_height = (uint32_t)height;
        stage->node_config.input_size = model_inputs[0].byte_size;
        if (!manifest_geometry) {
            stage->node_config.resized_width = (uint32_t)width;
            stage->node_config.resized_height = (uint32_t)height;
            stage->node_config.pad_x = 0;
            stage->node_config.pad_y = 0;
        }
    }
    const size_t options_size = strlen(config->options_json) + 1;
    stage->active_options_json = malloc(options_size);
    if (stage->active_options_json == NULL) return MIPK_ERR_MEMORY;
    memcpy(stage->active_options_json, config->options_json, options_size);
    mipk_status status = MIPK_OK;
    if (config->type == MIPK_STAGE_OPERATOR) {
        stage->operator_ops = mipk_operator_find(config->name);
        if (stage->operator_ops == NULL || stage->operator_ops->schema == NULL ||
            stage->operator_ops->schema->may_return_again !=
                stage->operator_ops->may_return_again) {
            return MIPK_ERR_UNSUPPORTED;
        }
        stage->context = stage->operator_ops->create(&stage->node_config,
            stage->active_options_json, &status);
    } else {
        stage->postprocess_ops = mipk_postprocess_find(config->name);
        if (stage->postprocess_ops == NULL ||
            stage->postprocess_ops->schema == NULL) {
            return MIPK_ERR_UNSUPPORTED;
        }
        stage->context = stage->postprocess_ops->create(&stage->node_config,
            stage->active_options_json, &status);
    }
    if (stage->context == NULL) return status;
    if (config->execution == MIPK_EXECUTION_PER_ITEM &&
        stage->operator_ops != NULL && stage->operator_ops->stateful) {
        return MIPK_ERR_UNSUPPORTED;
    }
    if (stage->operator_ops != NULL && stage->operator_ops->stateful) {
        if (stage->operator_ops->reset == NULL ||
            stage->operator_ops->checkpoint_size == 0 ||
            stage->operator_ops->checkpoint == NULL ||
            stage->operator_ops->restore == NULL) {
            return MIPK_ERR_UNSUPPORTED;
        }
        stage->checkpoint = calloc(1,
            stage->operator_ops->checkpoint_size);
        if (stage->checkpoint == NULL) return MIPK_ERR_MEMORY;
    }
    mipk_mutable_tensor *inputs = stage->model == NULL ? NULL :
        mipk_core_model_inputs(stage->model);
    const mipk_tensor *outputs = stage->model == NULL ? NULL :
        mipk_core_model_outputs(stage->model);
    const size_t input_count = stage->model == NULL ? 0 :
        mipk_core_model_input_count(stage->model);
    const size_t output_count = stage->model == NULL ? 0 :
        mipk_core_model_output_count(stage->model);
    if (config->type == MIPK_STAGE_OPERATOR) {
        status = stage->operator_ops->validate(stage->context, inputs,
            input_count, outputs, output_count);
        if (status != MIPK_OK || stage->model == NULL) return status;
        stage->prepared_tensors = calloc(input_count,
            sizeof(*stage->prepared_tensors));
        if (stage->prepared_tensors == NULL) return MIPK_ERR_MEMORY;
        stage->prepared_tensor_count = input_count;
        for (size_t tensor = 0; tensor < input_count; ++tensor) {
            stage->prepared_tensors[tensor] = inputs[tensor];
        }
        return MIPK_OK;
    }
    return stage->postprocess_ops->validate(stage->context, inputs, input_count,
        outputs, output_count);
}

mipk_status mipk_pipeline_runtime_create(const mipk_pipeline_config *config,
    mipk_pipeline_runtime **output)
{
    if (config == NULL || config->stages == NULL ||
        config->stage_count == 0 || config->output == NULL ||
        output == NULL) {
        return MIPK_ERR_ARGUMENT;
    }
    *output = NULL;
    mipk_pipeline_runtime *runtime = calloc(1, sizeof(*runtime));
    if (runtime == NULL) return MIPK_ERR_MEMORY;
    if (pthread_mutex_init(&runtime->run_lock, NULL) != 0) {
        free(runtime);
        return MIPK_ERR_RUNTIME;
    }
    runtime->run_lock_initialized = true;
    runtime->config = *config;
    runtime->collect = mipk_collect_results;
    mipk_status status = MIPK_OK;
    {
        runtime->output_stage_index = SIZE_MAX;
        for (size_t index = 0; index < config->stage_count; ++index) {
            if (strcmp(config->stages[index].output, config->output) == 0) {
                runtime->output_stage_index = index;
                break;
            }
        }
        if (runtime->output_stage_index == SIZE_MAX) status = MIPK_ERR_MODEL;
    }
    if (status == MIPK_OK) {
        status = create_models(runtime);
        if (status != MIPK_OK) {
            mipk_diag("mipk model backend setup failed: %s (%d)",
            mipk_status_string(status), (int)status);
        }
    }
    if (status == MIPK_OK) status = validate_typed_graph(runtime);
    if (status == MIPK_OK) status = validate_fusion_groups(config);
    if (status != MIPK_OK) {
        mipk_diag("mipk pipeline setup failed: %s (%d)",
            mipk_status_string(status), (int)status);
    }
    if (status == MIPK_OK) {
        runtime->stages = calloc(config->stage_count, sizeof(*runtime->stages));
        if (runtime->stages == NULL) status = MIPK_ERR_MEMORY;
        else runtime->stage_count = config->stage_count;
    }
    if (status == MIPK_OK) {
        runtime->execution_values = calloc(config->stage_count + 1,
            sizeof(*runtime->execution_values));
        runtime->execution_inputs = calloc(8,
            sizeof(*runtime->execution_inputs));
        if (runtime->execution_values == NULL ||
            runtime->execution_inputs == NULL) status = MIPK_ERR_MEMORY;
    }
    for (size_t index = 0; status == MIPK_OK && index < config->stage_count;
        ++index) {
        status = create_stage(runtime, index);
        if (status != MIPK_OK) {
            const mipk_stage_config *stage = &config->stages[index];
            mipk_diag("mipk pipeline stage[%zu] '%s' setup failed: "
                "%s (%d)", index, stage->name == NULL ? "model" :
                stage->name, mipk_status_string(status), (int)status);
        }
    }
    if (status == MIPK_OK) status = prepare_memory_plan(runtime);
    if (status != MIPK_OK) {
        mipk_pipeline_runtime_destroy(runtime);
        return status;
    }
    *output = runtime;
    return MIPK_OK;
}
