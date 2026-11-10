#include "mipk_pipeline_runtime_internal.h"

#include "cJSON.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    mipk_runtime_stage *stage;
    char *options_json;
    void *context;
} mipk_stage_update;

static cJSON *parse_json(const char *text)
{
    const char *end = NULL;
    cJSON *json = cJSON_ParseWithOpts(text, &end, true);
    if (json == NULL || end == NULL || *end != '\0') {
        cJSON_Delete(json);
        return NULL;
    }
    return json;
}

static const mipk_runtime_option *runtime_options(
    const mipk_runtime_stage *stage, size_t *count)
{
    if (count == NULL) return NULL;
    *count = 0;
    if (stage == NULL) return NULL;
    const mipk_runtime_option *options = NULL;
    if (stage->operator_ops != NULL) {
        *count = stage->operator_ops->runtime_option_count;
        options = stage->operator_ops->runtime_options;
    } else if (stage->postprocess_ops != NULL) {
        *count = stage->postprocess_ops->runtime_option_count;
        options = stage->postprocess_ops->runtime_options;
    }
    if (*count != 0 && options == NULL) {
        *count = 0;
        return NULL;
    }
    return options;
}

static bool option_allowed(const mipk_runtime_stage *stage, const char *name)
{
    size_t count = 0;
    const mipk_runtime_option *options = runtime_options(stage, &count);
    for (size_t index = 0; index < count; ++index) {
        if (options[index].name != NULL &&
            strcmp(options[index].name, name) == 0) return true;
    }
    return false;
}

static mipk_runtime_stage *find_stage(mipk_pipeline_runtime *runtime,
    const char *node_id)
{
    for (size_t index = 0; index < runtime->stage_count; ++index) {
        mipk_runtime_stage *stage = &runtime->stages[index];
        if (stage->config != NULL && stage->config->id != NULL &&
            strcmp(stage->config->id, node_id) == 0) return stage;
    }
    return NULL;
}

static const char *stage_type_name(mipk_stage_type type)
{
    switch (type) {
        case MIPK_STAGE_OPERATOR: return "operator";
        case MIPK_STAGE_MODEL: return "model";
        case MIPK_STAGE_POSTPROCESS: return "postprocess";
        default: return "unknown";
    }
}

size_t mipk_pipeline_runtime_get_option_count(
    const mipk_pipeline_runtime *runtime)
{
    if (runtime == NULL) return 0;
    size_t total = 0;
    for (size_t stage_index = 0; stage_index < runtime->stage_count;
        ++stage_index) {
        size_t count = 0;
        const mipk_runtime_option *options = runtime_options(
            &runtime->stages[stage_index], &count);
        for (size_t option_index = 0; option_index < count; ++option_index) {
            if (options != NULL && options[option_index].name != NULL &&
                options[option_index].name[0] != '\0') ++total;
        }
    }
    return total;
}

static bool runtime_option_location(const mipk_pipeline_runtime *runtime,
    size_t index, const mipk_runtime_stage **selected_stage,
    const mipk_runtime_option **selected_option)
{
    size_t current = 0;
    for (size_t stage_index = 0; stage_index < runtime->stage_count;
        ++stage_index) {
        const mipk_runtime_stage *stage = &runtime->stages[stage_index];
        size_t count = 0;
        const mipk_runtime_option *options = runtime_options(stage, &count);
        for (size_t option_index = 0; option_index < count; ++option_index) {
            const mipk_runtime_option *option = options == NULL ? NULL :
                &options[option_index];
            if (option == NULL || option->name == NULL ||
                option->name[0] == '\0') continue;
            if (current++ != index) continue;
            *selected_stage = stage;
            *selected_option = option;
            return true;
        }
    }
    return false;
}

mipk_status mipk_pipeline_runtime_get_option_at(
    const mipk_pipeline_runtime *runtime, size_t index,
    mipk_runtime_option_info *info)
{
    if (runtime == NULL || info == NULL ||
        info->struct_size != sizeof(*info)) return MIPK_ERR_ARGUMENT;
    const mipk_runtime_stage *stage = NULL;
    const mipk_runtime_option *option = NULL;
    if (!runtime_option_location(runtime, index, &stage, &option)) {
        return MIPK_ERR_ARGUMENT;
    }
    info->node_id = stage->config->id;
    info->node_type = stage_type_name(stage->type);
    info->node_name = stage->config->name;
    info->option_name = option->name;
    info->option_type = option->type;
    info->unit = option->unit;
    info->description = option->description;
    return MIPK_OK;
}

mipk_status mipk_pipeline_runtime_get_option_value_at(
    mipk_pipeline_runtime *runtime, size_t index, char *buffer,
    size_t capacity, size_t *required_size)
{
    if (runtime == NULL || required_size == NULL ||
        (buffer == NULL && capacity != 0)) return MIPK_ERR_ARGUMENT;
    if (pthread_mutex_trylock(&runtime->run_lock) != 0) return MIPK_ERR_BUSY;
    mipk_status status = runtime->poisoned ? MIPK_ERR_RUNTIME : MIPK_OK;
    const mipk_runtime_stage *stage = NULL;
    const mipk_runtime_option *option = NULL;
    if (status == MIPK_OK &&
        !runtime_option_location(runtime, index, &stage, &option)) {
        status = MIPK_ERR_ARGUMENT;
    }
    cJSON *options = status == MIPK_OK ?
        parse_json(stage->active_options_json) : NULL;
    if (status == MIPK_OK && !cJSON_IsObject(options)) {
        status = MIPK_ERR_RUNTIME;
    }
    const cJSON *value = status == MIPK_OK ?
        cJSON_GetObjectItemCaseSensitive(options, option->name) : NULL;
    char *owned_value_json = status == MIPK_OK && value != NULL ?
        cJSON_PrintUnformatted(value) : NULL;
    const char *value_json = value != NULL ? owned_value_json :
        option == NULL ? NULL : option->default_json;
    if (status == MIPK_OK && value != NULL && owned_value_json == NULL) {
        status = MIPK_ERR_MEMORY;
    } else if (status == MIPK_OK && value_json == NULL) {
        status = MIPK_ERR_RUNTIME;
    }
    if (status == MIPK_OK) {
        *required_size = strlen(value_json) + 1U;
        if (buffer != NULL && capacity < *required_size) {
            status = MIPK_ERR_CAPACITY;
        } else if (buffer != NULL) {
            memcpy(buffer, value_json, *required_size);
        }
    }
    free(owned_value_json);
    cJSON_Delete(options);
    (void)pthread_mutex_unlock(&runtime->run_lock);
    return status;
}

static bool object_has_unique_fields(const cJSON *object)
{
    if (!cJSON_IsObject(object)) return false;
    for (const cJSON *field = object->child; field != NULL;
        field = field->next) {
        if (field->string == NULL || field->string[0] == '\0') return false;
        for (const cJSON *other = field->next; other != NULL;
            other = other->next) {
            if (other->string != NULL &&
                strcmp(field->string, other->string) == 0) return false;
        }
    }
    return true;
}

static mipk_status merge_options(const mipk_runtime_stage *stage,
    const cJSON *patch, char **output)
{
    *output = NULL;
    if (!object_has_unique_fields(patch) ||
        stage->active_options_json == NULL) return MIPK_ERR_ARGUMENT;
    cJSON *merged = parse_json(stage->active_options_json);
    if (!cJSON_IsObject(merged)) {
        cJSON_Delete(merged);
        return MIPK_ERR_RUNTIME;
    }
    mipk_status status = MIPK_OK;
    for (const cJSON *field = patch->child; field != NULL;
        field = field->next) {
        if (!option_allowed(stage, field->string)) {
            status = MIPK_ERR_UNSUPPORTED;
            break;
        }
        cJSON *replacement = cJSON_Duplicate(field, true);
        if (replacement == NULL) {
            status = MIPK_ERR_MEMORY;
            break;
        }
        if (cJSON_HasObjectItem(merged, field->string)) {
            cJSON_ReplaceItemInObjectCaseSensitive(merged, field->string,
                replacement);
            if (replacement->string == NULL ||
                cJSON_GetObjectItemCaseSensitive(merged, field->string) !=
                    replacement) {
                if (replacement->string != NULL) cJSON_Delete(replacement);
                status = MIPK_ERR_MEMORY;
                break;
            }
        } else {
            cJSON_AddItemToObject(merged, field->string, replacement);
            if (cJSON_GetObjectItemCaseSensitive(merged, field->string) !=
                replacement) {
                cJSON_Delete(replacement);
                status = MIPK_ERR_MEMORY;
                break;
            }
        }
    }
    if (status == MIPK_OK) {
        *output = cJSON_PrintUnformatted(merged);
        if (*output == NULL) status = MIPK_ERR_MEMORY;
    }
    cJSON_Delete(merged);
    return status;
}

static void destroy_update(mipk_stage_update *update)
{
    if (update->context != NULL) {
        if (update->stage->operator_ops != NULL) {
            update->stage->operator_ops->destroy(update->context);
        } else {
            update->stage->postprocess_ops->destroy(update->context);
        }
    }
    free(update->options_json);
    memset(update, 0, sizeof(*update));
}

static mipk_status prepare_update(mipk_stage_update *update)
{
    mipk_runtime_stage *stage = update->stage;
    mipk_status status = MIPK_OK;
    if (stage->operator_ops != NULL) {
        update->context = stage->operator_ops->create(&stage->node_config,
            update->options_json, &status);
    } else if (stage->postprocess_ops != NULL) {
        update->context = stage->postprocess_ops->create(&stage->node_config,
            update->options_json, &status);
    } else {
        return MIPK_ERR_UNSUPPORTED;
    }
    if (update->context == NULL) {
        return status == MIPK_OK ? MIPK_ERR_RUNTIME : status;
    }

    mipk_mutable_tensor *inputs = stage->model == NULL ? NULL :
        mipk_core_model_inputs(stage->model);
    const mipk_tensor *outputs = stage->model == NULL ? NULL :
        mipk_core_model_outputs(stage->model);
    const size_t input_count = stage->model == NULL ? 0 :
        mipk_core_model_input_count(stage->model);
    const size_t output_count = stage->model == NULL ? 0 :
        mipk_core_model_output_count(stage->model);
    if (stage->operator_ops != NULL) {
        status = stage->operator_ops->validate(update->context, inputs,
            input_count, outputs, output_count);
        if (status == MIPK_OK && stage->operator_ops->stateful) {
            status = stage->operator_ops->checkpoint(stage->context,
                stage->checkpoint, stage->operator_ops->checkpoint_size);
            if (status == MIPK_OK) {
                status = stage->operator_ops->restore(update->context,
                    stage->checkpoint, stage->operator_ops->checkpoint_size);
            }
        }
    } else {
        status = stage->postprocess_ops->validate(update->context, inputs,
            input_count, outputs, output_count);
    }
    return status;
}

static mipk_status append_update(mipk_pipeline_runtime *runtime,
    const char *node_id, const cJSON *patch, mipk_stage_update updates[32],
    size_t *update_count)
{
    mipk_runtime_stage *stage = find_stage(runtime, node_id);
    if (stage == NULL || stage->type == MIPK_STAGE_MODEL) {
        return MIPK_ERR_ARGUMENT;
    }
    size_t allowed_count = 0;
    (void)runtime_options(stage, &allowed_count);
    if (allowed_count == 0) return MIPK_ERR_UNSUPPORTED;
    for (size_t index = 0; index < *update_count; ++index) {
        if (updates[index].stage == stage) return MIPK_ERR_ARGUMENT;
    }
    if (*update_count >= 32) return MIPK_ERR_CAPACITY;
    mipk_stage_update *update = &updates[(*update_count)++];
    update->stage = stage;
    return merge_options(stage, patch, &update->options_json);
}

static mipk_status commit_updates(mipk_stage_update *updates, size_t count)
{
    mipk_status status = MIPK_OK;
    for (size_t index = 0; index < count; ++index) {
        status = prepare_update(&updates[index]);
        if (status != MIPK_OK) break;
    }
    if (status != MIPK_OK) return status;

    for (size_t index = 0; index < count; ++index) {
        mipk_stage_update *update = &updates[index];
        mipk_runtime_stage *stage = update->stage;
        void *old_context = stage->context;
        char *old_options_json = stage->active_options_json;
        stage->context = update->context;
        stage->active_options_json = update->options_json;
        update->context = old_context;
        update->options_json = old_options_json;
    }
    return MIPK_OK;
}

static mipk_status apply_nodes_locked(mipk_pipeline_runtime *runtime,
    const cJSON *nodes)
{
    if (!object_has_unique_fields(nodes)) return MIPK_ERR_ARGUMENT;
    mipk_stage_update updates[32] = {0};
    size_t update_count = 0;
    mipk_status status = MIPK_OK;
    for (const cJSON *node = nodes->child; node != NULL; node = node->next) {
        status = append_update(runtime, node->string, node, updates,
            &update_count);
        if (status != MIPK_OK) break;
    }
    if (status == MIPK_OK) status = commit_updates(updates, update_count);
    for (size_t index = 0; index < update_count; ++index) {
        destroy_update(&updates[index]);
    }
    return status;
}

mipk_status mipk_pipeline_runtime_update_node_options(
    mipk_pipeline_runtime *runtime, const char *node_id,
    const char *options_json)
{
    if (runtime == NULL || node_id == NULL || node_id[0] == '\0' ||
        options_json == NULL) return MIPK_ERR_ARGUMENT;
    if (pthread_mutex_trylock(&runtime->run_lock) != 0) return MIPK_ERR_BUSY;
    mipk_status status = runtime->poisoned ? MIPK_ERR_RUNTIME : MIPK_OK;
    cJSON *patch = status == MIPK_OK ? parse_json(options_json) : NULL;
    if (status == MIPK_OK && !object_has_unique_fields(patch)) {
        status = MIPK_ERR_ARGUMENT;
    }
    mipk_stage_update update = {0};
    if (status == MIPK_OK) {
        update.stage = find_stage(runtime, node_id);
        size_t allowed_count = 0;
        if (update.stage == NULL || update.stage->type == MIPK_STAGE_MODEL) {
            status = MIPK_ERR_ARGUMENT;
        } else if (runtime_options(update.stage, &allowed_count) == NULL ||
            allowed_count == 0) {
            status = MIPK_ERR_UNSUPPORTED;
        } else {
            status = merge_options(update.stage, patch,
                &update.options_json);
        }
    }
    if (status == MIPK_OK) status = commit_updates(&update, 1);
    destroy_update(&update);
    cJSON_Delete(patch);
    (void)pthread_mutex_unlock(&runtime->run_lock);
    return status;
}

mipk_status mipk_pipeline_runtime_update_node_options_checked(
    mipk_pipeline_runtime *runtime, const char *node_id,
    mipk_stage_type expected_type, const char *expected_name,
    const char *options_json)
{
    if (runtime == NULL || node_id == NULL || node_id[0] == '\0' ||
        expected_name == NULL || expected_name[0] == '\0' ||
        options_json == NULL) return MIPK_ERR_ARGUMENT;
    if (pthread_mutex_trylock(&runtime->run_lock) != 0) return MIPK_ERR_BUSY;
    mipk_status status = runtime->poisoned ? MIPK_ERR_RUNTIME : MIPK_OK;
    mipk_runtime_stage *stage = status == MIPK_OK ?
        find_stage(runtime, node_id) : NULL;
    if (status == MIPK_OK && (stage == NULL || stage->type != expected_type ||
        stage->config->name == NULL ||
        strcmp(stage->config->name, expected_name) != 0)) {
        status = MIPK_ERR_ARGUMENT;
    }
    cJSON *patch = status == MIPK_OK ? parse_json(options_json) : NULL;
    if (status == MIPK_OK && !object_has_unique_fields(patch)) {
        status = MIPK_ERR_ARGUMENT;
    }
    mipk_stage_update update = {.stage = stage};
    if (status == MIPK_OK) {
        status = merge_options(stage, patch, &update.options_json);
    }
    if (status == MIPK_OK) status = commit_updates(&update, 1);
    destroy_update(&update);
    cJSON_Delete(patch);
    (void)pthread_mutex_unlock(&runtime->run_lock);
    return status;
}

mipk_status mipk_pipeline_runtime_apply_overrides_json(
    mipk_pipeline_runtime *runtime, const char *overrides_json)
{
    if (runtime == NULL || overrides_json == NULL) return MIPK_ERR_ARGUMENT;
    if (pthread_mutex_trylock(&runtime->run_lock) != 0) return MIPK_ERR_BUSY;
    mipk_status status = runtime->poisoned ? MIPK_ERR_RUNTIME : MIPK_OK;
    cJSON *root = status == MIPK_OK ? parse_json(overrides_json) : NULL;
    const cJSON *schema = root == NULL ? NULL :
        cJSON_GetObjectItemCaseSensitive(root, "schema");
    const cJSON *nodes = root == NULL ? NULL :
        cJSON_GetObjectItemCaseSensitive(root, "nodes");
    if (status == MIPK_OK && (!object_has_unique_fields(root) ||
        cJSON_GetArraySize(root) != 2 || !cJSON_IsString(schema) ||
        schema->valuestring == NULL || strcmp(schema->valuestring,
            "mipk.runtime_overrides.v1") != 0 ||
        !object_has_unique_fields(nodes))) {
        status = MIPK_ERR_ARGUMENT;
    }
    if (status == MIPK_OK) status = apply_nodes_locked(runtime, nodes);
    cJSON_Delete(root);
    (void)pthread_mutex_unlock(&runtime->run_lock);
    return status;
}
