#include "mipk_pipeline_runtime_internal.h"

#include "mipk_memory.h"
#include "mipk_tensor.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

static const mipk_value *find_value(const mipk_pipeline_runtime *runtime,
    const mipk_value *values, const char *source)
{
    if (strcmp(source, "$input") == 0) return &values[0];
    for (size_t index = 0; index < runtime->stage_count; ++index) {
        if (strcmp(runtime->config.stages[index].output, source) == 0) {
            return &values[index + 1];
        }
    }
    return NULL;
}

static void apply_compatibility_value(mipk_execution_context *execution,
    const mipk_value *value)
{
    if (value == NULL) return;
    if (value->type == MIPK_VALUE_INPUT && execution->input == NULL) {
        execution->input = value->value.input;
    } else if (value->type == MIPK_VALUE_IMAGE_VIEW &&
        execution->image_view == NULL) {
        execution->image_view = &value->value.image_view;
        if (execution->input == NULL) {
            execution->input = value->value.image_view.root_input;
        }
    } else if (value->type == MIPK_VALUE_TENSORS &&
        execution->model_outputs == NULL) {
        execution->model_outputs = value->value.tensors.items;
        execution->model_output_count = value->value.tensors.count;
    } else if (value->type == MIPK_VALUE_RESULTS) {
        execution->items = value->value.results.items;
        execution->item_count = value->value.results.item_count;
        execution->item_capacity = value->value.results.item_capacity;
        execution->scores = value->value.results.scores;
        execution->score_count = value->value.results.score_count;
    }
}

static void assign_result_identity(mipk_value *value, uint64_t parent_id,
    uint64_t *next_result_id)
{
    if (value == NULL || value->type != MIPK_VALUE_RESULTS ||
        next_result_id == NULL) return;
    for (size_t index = 0; index < value->value.results.item_count; ++index) {
        mipk_result *result = &value->value.results.items[index];
        if (result->result_id == MIPK_INVALID_RESULT_ID) {
            result->result_id = (*next_result_id)++;
        }
        if (result->parent_id == MIPK_INVALID_RESULT_ID &&
            parent_id != MIPK_INVALID_RESULT_ID) {
            result->parent_id = parent_id;
        }
    }
}

static bool tensor_index(const char *port, size_t count, size_t *index)
{
    if (port == NULL || *port == '\0' || index == NULL) return false;
    errno = 0;
    char *end = NULL;
    const unsigned long parsed = strtoul(port, &end, 10);
    if (errno != 0 || end == port || *end != '\0' || parsed >= count) {
        return false;
    }
    *index = (size_t)parsed;
    return true;
}

static mipk_status copy_tensor(const mipk_tensor *source,
    mipk_mutable_tensor *destination)
{
    const void *source_data = mipk_tensor_data(source);
    void *destination_data = mipk_mutable_tensor_data(destination);
    if (source_data == NULL || destination_data == NULL ||
        !mipk_tensor_range_valid(source) ||
        !mipk_tensor_range_valid(destination) ||
        source->byte_size > destination->byte_size) {
        return MIPK_ERR_MODEL;
    }
    if (source_data == destination_data) return MIPK_OK;
    if (mipk_memory_zero(destination_data, destination->byte_size) != 0 ||
        mipk_memory_copy(destination_data, destination->byte_size,
            source_data, source->byte_size) != 0) return MIPK_ERR_RUNTIME;
    return MIPK_OK;
}

static mipk_status bind_model_inputs(const mipk_runtime_stage *stage,
    const mipk_named_value *inputs, size_t input_count)
{
    mipk_mutable_tensor *destinations = mipk_core_model_inputs(stage->model);
    const size_t destination_count = mipk_core_model_input_count(stage->model);
    if (destinations == NULL || destination_count == 0) return MIPK_ERR_MODEL;
    for (size_t binding = 0; binding < input_count; ++binding) {
        const mipk_value *source = inputs[binding].value;
        size_t destination_index = 0;
        if (!tensor_index(inputs[binding].port, destination_count,
                &destination_index)) return MIPK_ERR_MODEL;
        if (source->type == MIPK_VALUE_INPUT) {
            const mipk_input *input = source->value.input;
            if (input == NULL || input->type != MIPK_INPUT_BYTES) {
                return MIPK_ERR_UNSUPPORTED;
            }
            mipk_buffer input_buffer = {
                .desc = input->value.bytes.buffer,
                .sync_state = input->value.bytes.initial_sync_state,
            };
            const mipk_tensor tensor = {
                .buffer = &input_buffer,
                .byte_size = input_buffer.desc.size,
            };
            mipk_status status = copy_tensor(&tensor,
                &destinations[destination_index]);
            if (status != MIPK_OK) return status;
        } else if (source->type == MIPK_VALUE_TENSORS) {
            const mipk_tensor *tensors = source->value.tensors.items;
            const size_t tensor_count = source->value.tensors.count;
            if (tensors == NULL || tensor_count == 0) return MIPK_ERR_MODEL;
            const mipk_stage_binding *config = &stage->config->inputs[binding];
            if (config->select_tensor) {
                if (config->tensor_index >= tensor_count) return MIPK_ERR_MODEL;
                mipk_status status = copy_tensor(&tensors[config->tensor_index],
                    &destinations[destination_index]);
                if (status != MIPK_OK) return status;
            } else if (tensor_count == 1) {
                mipk_status status = copy_tensor(&tensors[0],
                    &destinations[destination_index]);
                if (status != MIPK_OK) return status;
            } else if (input_count == 1 && destination_index == 0 &&
                tensor_count == destination_count) {
                for (size_t index = 0; index < tensor_count; ++index) {
                    mipk_status status = copy_tensor(&tensors[index],
                        &destinations[index]);
                    if (status != MIPK_OK) return status;
                }
            } else {
                return MIPK_ERR_MODEL;
            }
        } else if (source->type == MIPK_VALUE_COLLECTION) {
            return MIPK_ERR_UNSUPPORTED;
        } else {
            return MIPK_ERR_MODEL;
        }
    }
    return MIPK_OK;
}

static void capture_result_value(const mipk_execution_context *execution,
    mipk_value *value)
{
    value->type = MIPK_VALUE_RESULTS;
    value->value.results = (mipk_result_value) {
        .items = execution->items,
        .item_count = execution->item_count,
        .item_capacity = execution->item_capacity,
        .scores = execution->scores,
        .score_count = execution->score_count,
    };
}

static mipk_status run_stage(const mipk_runtime_stage *stage,
    mipk_execution_context *execution)
{
    mipk_value *value = execution->stage_output;
    if (stage->type == MIPK_STAGE_MODEL) {
        mipk_status status = bind_model_inputs(stage, execution->stage_inputs,
            execution->stage_input_count);
        if (status != MIPK_OK) return status;
        status = mipk_core_model_execute(stage->model, execution);
        if (status == MIPK_OK) {
            value->type = MIPK_VALUE_TENSORS;
            value->value.tensors.items = mipk_core_model_outputs(stage->model);
            value->value.tensors.count =
                mipk_core_model_output_count(stage->model);
            status = mipk_value_validate(value);
        }
        return status;
    }
    if (stage->type == MIPK_STAGE_OPERATOR) {
        mipk_status status = stage->operator_ops->process(stage->context,
            execution);
        if (status != MIPK_OK) return status;
        if (value->type != MIPK_VALUE_NONE) return mipk_value_validate(value);
        if (execution->items != NULL || execution->scores != NULL) {
            capture_result_value(execution, value);
        } else if (stage->prepared_tensors != NULL) {
            value->type = MIPK_VALUE_TENSORS;
            value->value.tensors.items = stage->prepared_tensors;
            value->value.tensors.count = stage->prepared_tensor_count;
        } else if (execution->stage_input_count == 1) {
            *value = *execution->stage_inputs[0].value;
            value->release = NULL;
            value->release_context = NULL;
        } else {
            return MIPK_ERR_OPERATOR;
        }
        return mipk_value_validate(value);
    }
    mipk_postprocess_output output = {0};
    mipk_status status = stage->postprocess_ops->process(stage->context,
        execution->model_outputs, execution->model_output_count, &output);
    if (status == MIPK_OK &&
        (output.item_count > output.item_capacity ||
         (output.item_count != 0 && output.items == NULL) ||
         (output.score_count != 0 && output.scores == NULL))) {
        status = MIPK_ERR_OPERATOR;
    }
    if (status == MIPK_OK) {
        execution->items = output.items;
        execution->item_count = output.item_count;
        execution->item_capacity = output.item_capacity;
        execution->scores = output.scores;
        execution->score_count = output.score_count;
        capture_result_value(execution, value);
        status = mipk_value_validate(value);
    }
    return status;
}

static void release_values(mipk_value *values, size_t count)
{
    for (size_t index = count; index > 0; --index) {
        mipk_value *value = &values[index - 1];
        if (value->release != NULL) {
            value->release(value->release_context, value);
        }
    }
}

static size_t per_item_group_end(const mipk_pipeline_runtime *runtime,
    size_t start)
{
    const size_t max_items = runtime->config.stages[start].max_items;
    size_t end = start;
    while (end + 1 < runtime->stage_count &&
        runtime->config.stages[end + 1].execution ==
            MIPK_EXECUTION_PER_ITEM &&
        runtime->config.stages[end + 1].max_items == max_items) {
        ++end;
    }
    return end;
}

static mipk_status group_item_count(const mipk_pipeline_runtime *runtime,
    const mipk_value *values, size_t start, size_t end, size_t max_items,
    size_t *item_count)
{
    bool found = false;
    size_t count = 0;
    for (size_t stage_index = start; stage_index <= end; ++stage_index) {
        const mipk_stage_config *stage = &runtime->config.stages[stage_index];
        for (size_t binding = 0; binding < stage->input_count; ++binding) {
            const mipk_value *source = find_value(runtime, values,
                stage->inputs[binding].source);
            if (source == NULL || source->type != MIPK_VALUE_COLLECTION) {
                continue;
            }
            if (source->value.collection.count > max_items) {
                return MIPK_ERR_CAPACITY;
            }
            if (!found) {
                count = source->value.collection.count;
                found = true;
            } else if (source->value.collection.count != count) {
                return MIPK_ERR_MODEL;
            }
        }
    }
    if (!found) return MIPK_ERR_MODEL;
    *item_count = count;
    return MIPK_OK;
}

static mipk_status select_group_input(const mipk_value *source,
    size_t item_index, uint64_t *item_id, const mipk_value **selected)
{
    if (source == NULL || selected == NULL) return MIPK_ERR_MODEL;
    if (source->type != MIPK_VALUE_COLLECTION) {
        *selected = source;
        return MIPK_OK;
    }
    const mipk_value *item = mipk_value_collection_at(source, item_index);
    if (item == NULL) return MIPK_ERR_MODEL;
    if (*item_id == MIPK_INVALID_ITEM_ID) {
        *item_id = item->metadata.item_id;
    } else if (item->metadata.item_id != *item_id) {
        return MIPK_ERR_MODEL;
    }
    *selected = item;
    return MIPK_OK;
}

static mipk_status capture_group_item(mipk_runtime_stage *terminal,
    const mipk_value *source, size_t item_index, uint64_t parent_id,
    uint64_t *next_result_id)
{
    if (terminal == NULL || source == NULL ||
        item_index >= terminal->collection_capacity) {
        return MIPK_ERR_CAPACITY;
    }
    mipk_value *destination = &terminal->collection_items[item_index];
    *destination = *source;
    destination->release = NULL;
    destination->release_context = NULL;
    if (destination->metadata.item_id == MIPK_INVALID_ITEM_ID) {
        destination->metadata.item_id = parent_id;
    }
    if (destination->metadata.parent_item_id == MIPK_INVALID_ITEM_ID) {
        destination->metadata.parent_item_id = parent_id;
    }
    if (source->type == MIPK_VALUE_TENSORS) return MIPK_ERR_UNSUPPORTED;
    if (source->type != MIPK_VALUE_RESULTS) return mipk_value_validate(destination);
    if (source->value.results.item_count >
        terminal->collection_results_per_item) return MIPK_ERR_CAPACITY;
    mipk_result *storage = terminal->collection_results +
        item_index * terminal->collection_results_per_item;
    if (source->value.results.item_count != 0) {
        memcpy(storage, source->value.results.items,
            source->value.results.item_count * sizeof(*storage));
    }
    destination->value.results.items = storage;
    destination->value.results.item_capacity =
        terminal->collection_results_per_item;
    assign_result_identity(destination, parent_id, next_result_id);
    return mipk_value_validate(destination);
}

static mipk_status run_per_item_group(mipk_pipeline_runtime *runtime,
    mipk_value *values, mipk_named_value *inputs, size_t start, size_t end,
    mipk_result_buffer *results, uint64_t *next_result_id)
{
    mipk_runtime_stage *terminal = &runtime->stages[end];
    const size_t max_items = runtime->config.stages[start].max_items;
    if (terminal->collection_items == NULL ||
        terminal->collection_capacity < max_items) return MIPK_ERR_RUNTIME;
    size_t item_count = 0;
    mipk_status status = group_item_count(runtime, values, start, end,
        max_items, &item_count);
    if (status != MIPK_OK) return status;
    memset(terminal->collection_items, 0,
        terminal->collection_capacity * sizeof(*terminal->collection_items));
    for (size_t item_index = 0;
        status == MIPK_OK && item_index < item_count; ++item_index) {
        uint64_t item_id = MIPK_INVALID_ITEM_ID;
        for (size_t stage_index = start;
            status == MIPK_OK && stage_index <= end; ++stage_index) {
            mipk_runtime_stage *stage = &runtime->stages[stage_index];
            const mipk_stage_config *config = stage->config;
            memset(inputs, 0, 8 * sizeof(*inputs));
            for (size_t binding = 0; binding < config->input_count; ++binding) {
                const mipk_value *source = find_value(runtime, values,
                    config->inputs[binding].source);
                inputs[binding].port = config->inputs[binding].port;
                status = select_group_input(source, item_index, &item_id,
                    &inputs[binding].value);
                if (status != MIPK_OK ||
                    mipk_value_validate(inputs[binding].value) != MIPK_OK) {
                    status = MIPK_ERR_MODEL;
                    break;
                }
            }
            if (status != MIPK_OK) break;
            mipk_execution_context execution = {
                .stage_inputs = inputs,
                .stage_input_count = config->input_count,
                .stage_output = &values[stage_index + 1],
                .collection_items = stage->collection_items,
                .collection_capacity = stage->collection_capacity,
                .collection_results = stage->collection_results,
                .collection_result_capacity =
                    stage->collection_result_capacity,
                .results = results,
            };
            memset(execution.stage_output, 0, sizeof(*execution.stage_output));
            for (size_t binding = 0; binding < config->input_count; ++binding) {
                apply_compatibility_value(&execution, inputs[binding].value);
            }
            if (stage->model != NULL) {
                execution.model_inputs = mipk_core_model_inputs(stage->model);
                execution.model_input_count =
                    mipk_core_model_input_count(stage->model);
                if (execution.model_outputs == NULL) {
                    execution.model_outputs =
                        mipk_core_model_outputs(stage->model);
                    execution.model_output_count =
                        mipk_core_model_output_count(stage->model);
                }
            }
            status = run_stage(stage, &execution);
            if (status == MIPK_OK) {
                execution.stage_output->metadata.item_id = item_id;
                execution.stage_output->metadata.parent_item_id = item_id;
            }
        }
        if (status == MIPK_OK) {
            status = capture_group_item(terminal, &values[end + 1],
                item_index, item_id, next_result_id);
        }
        for (size_t stage_index = start; stage_index <= end; ++stage_index) {
            memset(&values[stage_index + 1], 0,
                sizeof(values[stage_index + 1]));
        }
    }
    if (status != MIPK_OK) return status;
    mipk_value_type item_type = MIPK_VALUE_NONE;
    if (item_count != 0) item_type = terminal->collection_items[0].type;
    else if (terminal->type == MIPK_STAGE_POSTPROCESS) {
        item_type = MIPK_VALUE_RESULTS;
    } else if (terminal->type == MIPK_STAGE_MODEL) {
        item_type = MIPK_VALUE_TENSORS;
    } else {
        item_type = terminal->operator_ops->schema->output_type;
    }
    values[end + 1] = (mipk_value) {
        .type = MIPK_VALUE_COLLECTION,
        .value.collection = {
            .items = terminal->collection_items,
            .count = item_count,
            .capacity = terminal->collection_capacity,
            .item_type = item_type,
        },
    };
    return mipk_value_validate(&values[end + 1]);
}

/*
 * Copies the graph's output value into the caller's buffer: a result set as
 * it is, a collection of result sets flattened with each item's parent id
 * filled in so a caller can put a cascade back together.
 */
mipk_status mipk_collect_results(const mipk_value *output,
    mipk_result_buffer *results)
{
    if (output == NULL || results == NULL) return MIPK_ERR_ARGUMENT;
    if (output->type == MIPK_VALUE_COLLECTION) {
        size_t count = 0;
        for (size_t item_index = 0;
            item_index < output->value.collection.count; ++item_index) {
            const mipk_value *item = mipk_value_collection_at(output,
                item_index);
            if (item == NULL || item->type != MIPK_VALUE_RESULTS) {
                return MIPK_ERR_MODEL;
            }
            if (item->value.results.item_count > results->capacity - count) {
                return MIPK_ERR_CAPACITY;
            }
            for (size_t result_index = 0;
                result_index < item->value.results.item_count;
                ++result_index) {
                mipk_result result = item->value.results.items[result_index];
                if (result.parent_id == MIPK_INVALID_RESULT_ID) {
                    result.parent_id = item->metadata.parent_item_id;
                }
                results->items[count++] = result;
            }
        }
        results->count = count;
        return MIPK_OK;
    }
    if (output->type != MIPK_VALUE_RESULTS) return MIPK_ERR_MODEL;
    const mipk_result_value *value = &output->value.results;
    if (value->item_count != 0 && value->items == NULL) return MIPK_ERR_MODEL;
    if (value->item_count > results->capacity) return MIPK_ERR_CAPACITY;
    if (value->item_count != 0) {
        memcpy(results->items, value->items,
            value->item_count * sizeof(*value->items));
    }
    results->count = value->item_count;
    return MIPK_OK;
}

mipk_status mipk_pipeline_runtime_run(mipk_pipeline_runtime *runtime,
    const mipk_input *input, mipk_result_buffer *results)
{
    if (runtime == NULL || input == NULL || results == NULL ||
        runtime->poisoned ||
        (results->capacity != 0 && results->items == NULL)) {
        return MIPK_ERR_ARGUMENT;
    }
    if (pthread_mutex_trylock(&runtime->run_lock) != 0) return MIPK_ERR_BUSY;
    results->count = 0;
    mipk_value *values = runtime->execution_values;
    mipk_named_value *inputs = runtime->execution_inputs;
    memset(values, 0, (runtime->stage_count + 1) * sizeof(*values));
    memset(inputs, 0, 8 * sizeof(*inputs));
    values[0].type = MIPK_VALUE_INPUT;
    values[0].value.input = input;
    uint64_t next_result_id = 1;
    mipk_status status = MIPK_OK;
    bool checkpointed[32] = {0};
    if (runtime->stage_count > sizeof(checkpointed) / sizeof(checkpointed[0])) {
        status = MIPK_ERR_CAPACITY;
    }
    for (size_t index = 0; status == MIPK_OK &&
        index < runtime->stage_count; ++index) {
        mipk_runtime_stage *stage = &runtime->stages[index];
        const mipk_stage_config *config = stage->config;
        if (config->execution == MIPK_EXECUTION_PER_ITEM) {
            const size_t end = per_item_group_end(runtime, index);
            status = run_per_item_group(runtime, values, inputs, index, end,
                results, &next_result_id);
            index = end;
            continue;
        }
        memset(inputs, 0, 8 * sizeof(*inputs));
        for (size_t binding = 0; binding < config->input_count; ++binding) {
            inputs[binding].port = config->inputs[binding].port;
            inputs[binding].value = find_value(runtime, values,
                config->inputs[binding].source);
            if (inputs[binding].value == NULL ||
                inputs[binding].value->type == MIPK_VALUE_NONE ||
                mipk_value_validate(inputs[binding].value) != MIPK_OK) {
                status = MIPK_ERR_MODEL;
                break;
            }
        }
        if (status == MIPK_AGAIN) break;
        if (status != MIPK_OK) break;
        mipk_execution_context execution = {
            .stage_inputs = inputs,
            .stage_input_count = config->input_count,
            .stage_output = &values[index + 1],
            .collection_items = stage->collection_items,
            .collection_capacity = stage->collection_capacity,
            .collection_results = stage->collection_results,
            .collection_result_capacity = stage->collection_result_capacity,
            .results = results,
        };
        for (size_t binding = 0; binding < config->input_count; ++binding) {
            apply_compatibility_value(&execution, inputs[binding].value);
        }
        if (stage->model != NULL) {
            execution.model_inputs = mipk_core_model_inputs(stage->model);
            execution.model_input_count =
                mipk_core_model_input_count(stage->model);
            if (execution.model_outputs == NULL) {
                execution.model_outputs = mipk_core_model_outputs(stage->model);
                execution.model_output_count =
                    mipk_core_model_output_count(stage->model);
            }
        }
        if (stage->operator_ops != NULL && stage->operator_ops->stateful) {
            status = stage->operator_ops->checkpoint(stage->context,
                stage->checkpoint, stage->operator_ops->checkpoint_size);
            if (status != MIPK_OK) break;
            checkpointed[index] = true;
        }
        status = run_stage(stage, &execution);
        if (status == MIPK_AGAIN && (stage->operator_ops == NULL ||
                !stage->operator_ops->may_return_again)) {
            status = MIPK_ERR_RUNTIME;
        }
        if (status != MIPK_OK) break;
        assign_result_identity(&values[index + 1],
            MIPK_INVALID_RESULT_ID, &next_result_id);
    }
    if (status == MIPK_OK) {
        status = runtime->collect(&values[runtime->output_stage_index + 1],
            results);
    }
    if (status != MIPK_OK && status != MIPK_AGAIN) {
        results->count = 0;
        for (size_t index = runtime->stage_count; index > 0; --index) {
            mipk_runtime_stage *stage = &runtime->stages[index - 1];
            if (!checkpointed[index - 1]) continue;
            if (stage->operator_ops->restore(stage->context,
                    stage->checkpoint,
                    stage->operator_ops->checkpoint_size) != MIPK_OK) {
                runtime->poisoned = true;
                status = MIPK_ERR_RUNTIME;
            }
        }
    }
    release_values(values + 1, runtime->stage_count);
    (void)pthread_mutex_unlock(&runtime->run_lock);
    return status;
}

const char *mipk_pipeline_runtime_class_name(
    const mipk_pipeline_runtime *runtime, int32_t class_id)
{
    if (runtime == NULL || class_id < 0 ||
        (size_t)class_id >= runtime->config.class_count) return "unknown";
    return runtime->config.classes[class_id].name;
}

mipk_status mipk_pipeline_runtime_reset(mipk_pipeline_runtime *runtime)
{
    if (runtime == NULL) return MIPK_ERR_ARGUMENT;
    if (pthread_mutex_trylock(&runtime->run_lock) != 0) return MIPK_ERR_BUSY;
    mipk_status status = runtime->poisoned ? MIPK_ERR_RUNTIME : MIPK_OK;
    for (size_t index = 0; index < runtime->stage_count; ++index) {
        if (status != MIPK_OK) break;
        mipk_runtime_stage *stage = &runtime->stages[index];
        if (stage->operator_ops == NULL || !stage->operator_ops->stateful) {
            continue;
        }
        if (stage->operator_ops->reset == NULL) {
            runtime->poisoned = true;
            status = MIPK_ERR_RUNTIME;
            break;
        }
        status = stage->operator_ops->reset(stage->context);
        if (status != MIPK_OK) {
            runtime->poisoned = true;
            break;
        }
    }
    (void)pthread_mutex_unlock(&runtime->run_lock);
    return status;
}

bool mipk_pipeline_runtime_busy(mipk_pipeline_runtime *runtime)
{
    if (runtime == NULL) return false;
    if (pthread_mutex_trylock(&runtime->run_lock) != 0) return true;
    (void)pthread_mutex_unlock(&runtime->run_lock);
    return false;
}
