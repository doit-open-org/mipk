#define _XOPEN_SOURCE 700

#include "mipk_pipeline_private.h"

#include "mipk_config.h"
#include "cJSON.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static mipk_status manifest_error(const char *field, const char *message)
{
    mipk_diag("invalid manifest field '%s': %s", field, message);
    return MIPK_ERR_MODEL;
}

static bool json_string(const cJSON *object, const char *name,
    const cJSON **value)
{
    *value = cJSON_GetObjectItemCaseSensitive(object, name);
    return cJSON_IsString(*value) && (*value)->valuestring != NULL &&
        (*value)->valuestring[0] != '\0';
}

static bool json_size(const cJSON *object, const char *name,
    size_t default_value, size_t maximum, size_t *value)
{
    const cJSON *item = cJSON_GetObjectItemCaseSensitive(object, name);
    if (item == NULL) {
        *value = default_value;
        return true;
    }
    if (!cJSON_IsNumber(item) || !isfinite(item->valuedouble) ||
        item->valuedouble < 0.0 || item->valuedouble > (double)maximum ||
        floor(item->valuedouble) != item->valuedouble) return false;
    *value = (size_t)item->valuedouble;
    return true;
}

static bool object_has_exact_fields(const cJSON *object,
    const char *const *allowed, size_t allowed_count)
{
    if (!cJSON_IsObject(object)) return false;
    for (const cJSON *field = object->child; field != NULL;
        field = field->next) {
        bool found = false;
        if (field->string == NULL || field->string[0] == '\0') return false;
        for (size_t index = 0; index < allowed_count; ++index) {
            if (strcmp(field->string, allowed[index]) == 0) {
                found = true;
                break;
            }
        }
        if (!found) return false;
        for (const cJSON *other = field->next; other != NULL;
            other = other->next) {
            if (other->string != NULL &&
                strcmp(field->string, other->string) == 0) return false;
        }
    }
    return true;
}

static mipk_status require_exact_fields(const cJSON *object,
    const char *const *allowed, size_t allowed_count, const char *field)
{
    return object_has_exact_fields(object, allowed, allowed_count) ? MIPK_OK :
        manifest_error(field, "contains an unknown or duplicate field");
}

static bool safe_relative_path(const char *path)
{
    if (path == NULL || path[0] == '\0' || path[0] == '/' || path[0] == '\\') {
        return false;
    }
    const char *segment = path;
    for (const char *cursor = path;; ++cursor) {
        if (*cursor == '\\') return false;
        if (*cursor == '/' || *cursor == '\0') {
            const size_t length = (size_t)(cursor - segment);
            if (length == 0 || (length == 1 && segment[0] == '.') ||
                (length == 2 && segment[0] == '.' && segment[1] == '.')) {
                return false;
            }
            if (*cursor == '\0') break;
            segment = cursor + 1;
        }
    }
    return true;
}

#define MIPK_DEFAULT_RESULT_CAPACITY 100U

static mipk_status parse_task(const cJSON *root, mipk_pipeline *pipeline)
{
    const cJSON *task = cJSON_GetObjectItemCaseSensitive(root, "task");
    static const char *const fields[] = {
        "type", "result_types", "result_capacity", "attachment_capacity",
        "attachment_bytes",
    };
    if (task == NULL) {
        /* Every result type the graph may produce, a hundred of them. */
        pipeline->task_name = NULL;
        pipeline->config.result_capacity = MIPK_DEFAULT_RESULT_CAPACITY;
        pipeline->config.result_type_mask = (UINT64_C(1) << MIPK_RESULT_TEXT) |
            ((UINT64_C(1) << MIPK_RESULT_TEXT) - 1U);
        return MIPK_OK;
    }
    mipk_status field_status = require_exact_fields(task, fields,
        sizeof(fields) / sizeof(fields[0]), "task");
    if (field_status != MIPK_OK) return field_status;
    size_t capacity = MIPK_DEFAULT_RESULT_CAPACITY;
    size_t attachment_capacity = 0;
    size_t attachment_bytes = 0;
    if (!cJSON_IsObject(task) ||
        !json_size(task, "result_capacity", MIPK_DEFAULT_RESULT_CAPACITY,
            SIZE_MAX, &capacity) ||
        !json_size(task, "attachment_capacity", 0, SIZE_MAX,
            &attachment_capacity) ||
        !json_size(task, "attachment_bytes", 0, SIZE_MAX,
            &attachment_bytes) || capacity == 0) {
        return manifest_error("task", "invalid result or attachment capacity");
    }
    /*
     * "type" is informational: the graph is validated from its nodes and
     * the input format says whether frames or audio come in.  When present
     * it must agree with the input; parse_input() checks that.
     */
    const cJSON *type = cJSON_GetObjectItemCaseSensitive(task, "type");
    if (type != NULL && (!cJSON_IsString(type) || type->valuestring == NULL ||
        (strcmp(type->valuestring, "vision_pipeline") != 0 &&
            strcmp(type->valuestring, "audio_pipeline") != 0))) {
        return manifest_error("task.type",
            "must be vision_pipeline or audio_pipeline when given");
    }
    pipeline->task_name = type == NULL ? NULL :
        strcmp(type->valuestring, "audio_pipeline") == 0 ? "audio_pipeline" :
        "vision_pipeline";
    if (attachment_capacity != 0 || attachment_bytes != 0) {
        return manifest_error("task", "stage 1 does not accept attachments");
    }
    const cJSON *result_types = cJSON_GetObjectItemCaseSensitive(task,
        "result_types");
    if (result_types == NULL) {
        pipeline->config.result_capacity = capacity;
        pipeline->config.result_type_mask = (UINT64_C(1) << MIPK_RESULT_TEXT) |
            ((UINT64_C(1) << MIPK_RESULT_TEXT) - 1U);
        return MIPK_OK;
    }
    if (!cJSON_IsArray(result_types) || cJSON_GetArraySize(result_types) == 0) {
        return manifest_error("task.result_types", "must be a non-empty array");
    }
    uint64_t result_type_mask = 0;
    const int result_type_count = cJSON_GetArraySize(result_types);
    for (int index = 0; index < result_type_count; ++index) {
        const cJSON *item = cJSON_GetArrayItem(result_types, index);
        mipk_result_type result_type;
        if (!cJSON_IsString(item) || item->valuestring == NULL) {
            return manifest_error("task.result_types[]", "must be a string");
        }
        if (strcmp(item->valuestring, "detection") == 0) {
            result_type = MIPK_RESULT_DETECTION;
        } else if (strcmp(item->valuestring, "classification") == 0) {
            result_type = MIPK_RESULT_CLASSIFICATION;
        } else if (strcmp(item->valuestring, "keypoint") == 0) {
            result_type = MIPK_RESULT_KEYPOINT;
        } else if (strcmp(item->valuestring, "attribute") == 0) {
            result_type = MIPK_RESULT_ATTRIBUTE;
        } else if (strcmp(item->valuestring, "track") == 0) {
            result_type = MIPK_RESULT_TRACK;
        } else if (strcmp(item->valuestring, "text") == 0) {
            result_type = MIPK_RESULT_TEXT;
        } else {
            return manifest_error("task.result_types[]", "unknown result type");
        }
        const uint64_t bit = UINT64_C(1) << result_type;
        if ((result_type_mask & bit) != 0) {
            return manifest_error("task.result_types[]", "must be unique");
        }
        result_type_mask |= bit;
    }
    pipeline->config.result_capacity = capacity;
    pipeline->config.result_type_mask = result_type_mask;
    return MIPK_OK;
}

static bool parse_format(const char *value, mipk_data_format *format)
{
    if (strcmp(value, "nv12") == 0) *format = MIPK_FORMAT_NV12;
    else if (strcmp(value, "nv21") == 0) *format = MIPK_FORMAT_NV21;
    else if (strcmp(value, "bgr") == 0) *format = MIPK_FORMAT_BGR;
    else if (strcmp(value, "rgb") == 0) *format = MIPK_FORMAT_RGB;
    else if (strcmp(value, "pcm_s16le") == 0) *format = MIPK_FORMAT_PCM_S16LE;
    else return false;
    return true;
}

static mipk_status parse_label_item(const cJSON *item, int index,
    mipk_class_config *classes)
{
    const cJSON *name = item;
    int32_t source_id = index;
    bool enabled = true;
    float threshold = -1.0F;
    if (cJSON_IsObject(item)) {
        static const char *const fields[] = {
            "name", "source_id", "enabled", "score_threshold",
        };
        mipk_status field_status = require_exact_fields(item, fields,
            sizeof(fields) / sizeof(fields[0]), "label_sets.*[]");
        if (field_status != MIPK_OK) return field_status;
        if (!json_string(item, "name", &name)) {
            return manifest_error("label_sets.*[].name", "must be a string");
        }
        const cJSON *enabled_item = cJSON_GetObjectItemCaseSensitive(item,
            "enabled");
        const cJSON *threshold_item = cJSON_GetObjectItemCaseSensitive(item,
            "score_threshold");
        const cJSON *source_id_item = cJSON_GetObjectItemCaseSensitive(item,
            "source_id");
        if (enabled_item != NULL && !cJSON_IsBool(enabled_item)) {
            return manifest_error("label_sets.*[].enabled", "must be boolean");
        }
        if (enabled_item != NULL) enabled = cJSON_IsTrue(enabled_item);
        if (cJSON_IsNull(source_id_item)) source_id = -1;
        else if (source_id_item != NULL) {
            if (!cJSON_IsNumber(source_id_item) ||
                !isfinite(source_id_item->valuedouble) ||
                source_id_item->valuedouble < 0.0 ||
                source_id_item->valuedouble > INT32_MAX ||
                floor(source_id_item->valuedouble) !=
                    source_id_item->valuedouble) {
                return manifest_error("label_sets.*[].source_id",
                    "must be a non-negative integer or null");
            }
            source_id = (int32_t)source_id_item->valuedouble;
        }
        if (threshold_item != NULL && (!cJSON_IsNumber(threshold_item) ||
                !isfinite(threshold_item->valuedouble) ||
                threshold_item->valuedouble < 0.0 ||
                threshold_item->valuedouble > 1.0)) {
            return manifest_error("label_sets.*[].score_threshold",
                "must be in [0, 1]");
        }
        if (threshold_item != NULL) threshold = (float)threshold_item->valuedouble;
    }
    if (!cJSON_IsString(name) || name->valuestring == NULL ||
        name->valuestring[0] == '\0') {
        return manifest_error("label_sets.*[]", "must contain names");
    }
    for (int previous = 0; previous < index; ++previous) {
        if (strcmp(name->valuestring, classes[previous].name) == 0 ||
            (source_id >= 0 && source_id == classes[previous].source_id)) {
            return manifest_error("label_sets", "names and source IDs must be unique per set");
        }
    }
    classes[index] = (mipk_class_config) {
        .name = strdup(name->valuestring),
        .source_id = source_id,
        .enabled = enabled,
        .score_threshold = threshold,
    };
    return classes[index].name == NULL ? MIPK_ERR_MEMORY : MIPK_OK;
}

/*
 * "label_sets": {"labels": {"file": "coco80.txt"}} reads one name per line
 * from beside manifest.json.  Only a directory package can do this; the
 * packaging tool inlines the names when it builds a .mipk.
 */
static char *join_path(const char *directory, const char *file);

static cJSON *load_label_file(const mipk_pipeline *pipeline,
    const char *file)
{
    if (pipeline->model_sources != NULL || !safe_relative_path(file)) {
        manifest_error("label_sets.*.file",
            "only a recipe directory may reference a label file");
        return NULL;
    }
    char *path = join_path(pipeline->package_path, file);
    if (path == NULL) return NULL;
    FILE *stream = fopen(path, "rb");
    free(path);
    if (stream == NULL) {
        manifest_error("label_sets.*.file", "cannot open the label file");
        return NULL;
    }
    cJSON *labels = cJSON_CreateArray();
    char line[512];
    while (labels != NULL && fgets(line, sizeof(line), stream) != NULL) {
        size_t length = strlen(line);
        while (length != 0 && (line[length - 1] == '\n' ||
            line[length - 1] == '\r' || line[length - 1] == ' ')) {
            line[--length] = '\0';
        }
        if (length == 0 || line[0] == '#') continue;
        cJSON *item = cJSON_CreateString(line);
        if (item == NULL) {
            cJSON_Delete(labels);
            labels = NULL;
            break;
        }
        cJSON_AddItemToArray(labels, item);
    }
    fclose(stream);
    return labels;
}

static mipk_status parse_label_sets(const cJSON *root, mipk_pipeline *pipeline)
{
    const cJSON *sets = cJSON_GetObjectItemCaseSensitive(root, "label_sets");
    const int set_count = cJSON_GetArraySize(sets);
    if (!cJSON_IsObject(sets) || set_count <= 0) {
        return manifest_error("label_sets", "must be a non-empty object");
    }
    pipeline->label_sets = calloc((size_t)set_count,
        sizeof(*pipeline->label_sets));
    if (pipeline->label_sets == NULL) return MIPK_ERR_MEMORY;
    pipeline->label_set_count = (size_t)set_count;
    const cJSON *set = sets->child;
    for (int set_index = 0; set_index < set_count; ++set_index, set = set->next) {
        cJSON *loaded = NULL;
        const cJSON *file = cJSON_IsObject(set) ?
            cJSON_GetObjectItemCaseSensitive(set, "file") : NULL;
        if (file != NULL) {
            if (!cJSON_IsString(file) || cJSON_GetArraySize(set) != 1) {
                return manifest_error("label_sets.*", "file reference takes only \"file\"");
            }
            loaded = load_label_file(pipeline, file->valuestring);
            if (loaded == NULL) return MIPK_ERR_MODEL;
            /* Replace the reference in the document so the rest reads an array. */
            cJSON_ReplaceItemInObjectCaseSensitive((cJSON *)sets, set->string,
                loaded);
            set = loaded;
        }
        const int class_count = cJSON_GetArraySize(set);
        if (set->string == NULL || set->string[0] == '\0' ||
            !cJSON_IsArray(set) || class_count <= 0) {
            return manifest_error("label_sets.*", "must be a named non-empty array");
        }
        for (int previous = 0; previous < set_index; ++previous) {
            if (strcmp(set->string, pipeline->label_sets[previous].name) == 0) {
                return manifest_error("label_sets", "set names must be unique");
            }
        }
        mipk_label_set_config *target = &pipeline->label_sets[set_index];
        target->name = strdup(set->string);
        target->classes = calloc((size_t)class_count, sizeof(*target->classes));
        target->class_count = (size_t)class_count;
        if (target->name == NULL || target->classes == NULL) return MIPK_ERR_MEMORY;
        for (int index = 0; index < class_count; ++index) {
            mipk_status status = parse_label_item(cJSON_GetArrayItem(set, index),
                index, target->classes);
            if (status != MIPK_OK) return status;
        }
    }
    pipeline->config.label_sets = pipeline->label_sets;
    pipeline->config.label_set_count = pipeline->label_set_count;
    pipeline->config.classes = pipeline->label_sets[0].classes;
    pipeline->config.class_count = pipeline->label_sets[0].class_count;
    return MIPK_OK;
}

static mipk_status parse_input(const cJSON *root, mipk_pipeline *pipeline)
{
    const cJSON *input = cJSON_GetObjectItemCaseSensitive(root, "input");
    const cJSON *format = NULL;
    if (!cJSON_IsObject(input) || !json_string(input, "format", &format) ||
        !parse_format(format->valuestring, &pipeline->info.input_format)) {
        return manifest_error("input.format", "unsupported format");
    }
    if (pipeline->info.input_format == MIPK_FORMAT_PCM_S16LE) {
        static const char *const fields[] = {
            "format", "sample_rate", "channels", "frame_samples",
            "reset_gap_ms",
        };
        mipk_status field_status = require_exact_fields(input, fields,
            sizeof(fields) / sizeof(fields[0]), "input");
        if (field_status != MIPK_OK) return field_status;
        size_t sample_rate = 0;
        size_t channels = 0;
        size_t frame_samples = 0;
        size_t reset_gap_ms = 0;
        if (!json_size(input, "sample_rate", 0, UINT32_MAX, &sample_rate) ||
            !json_size(input, "channels", 0, UINT32_MAX, &channels) ||
            !json_size(input, "frame_samples", 0, UINT32_MAX,
                &frame_samples) ||
            !json_size(input, "reset_gap_ms", 0, UINT32_MAX,
                &reset_gap_ms) || sample_rate == 0 || channels == 0 ||
            frame_samples == 0 || frame_samples > SIZE_MAX / channels ||
            frame_samples * channels > SIZE_MAX / sizeof(int16_t) ||
            reset_gap_ms == 0) {
            return manifest_error("input", "invalid PCM configuration");
        }
        pipeline->info.input_sample_rate = (uint32_t)sample_rate;
        pipeline->info.input_channels = (uint32_t)channels;
        pipeline->info.input_frame_samples = (uint32_t)frame_samples;
        pipeline->info.input_size = frame_samples * channels * sizeof(int16_t);
        pipeline->audio_reset_gap_us = (uint64_t)reset_gap_ms * UINT64_C(1000);
    } else {
        static const char *const fields[] = {"format", "width", "height"};
        mipk_status field_status = require_exact_fields(input, fields,
            sizeof(fields) / sizeof(fields[0]), "input");
        if (field_status != MIPK_OK) return field_status;
        size_t width = 0;
        size_t height = 0;
        if (!json_size(input, "width", 0, UINT32_MAX, &width) ||
            !json_size(input, "height", 0, UINT32_MAX, &height) ||
            width == 0 || height == 0 || width > SIZE_MAX / height) {
            return manifest_error("input", "invalid image dimensions");
        }
        const size_t pixels = width * height;
        if (pipeline->info.input_format == MIPK_FORMAT_NV12 ||
            pipeline->info.input_format == MIPK_FORMAT_NV21) {
            if ((width & 1U) != 0 || (height & 1U) != 0 ||
                pixels > SIZE_MAX - pixels / 2U) {
                return manifest_error("input", "invalid YUV dimensions");
            }
            pipeline->info.input_size = pixels + pixels / 2U;
        } else {
            if (pixels > SIZE_MAX / 3U) {
                return manifest_error("input", "image dimensions overflow");
            }
            pipeline->info.input_size = pixels * 3U;
        }
        pipeline->info.input_width = (uint32_t)width;
        pipeline->info.input_height = (uint32_t)height;
    }
    pipeline->config.input_width = pipeline->info.input_width;
    pipeline->config.input_height = pipeline->info.input_height;
    pipeline->config.input_size = pipeline->info.input_size;
    pipeline->config.input_sample_rate = pipeline->info.input_sample_rate;
    pipeline->config.input_channels = pipeline->info.input_channels;
    pipeline->config.input_frame_samples = pipeline->info.input_frame_samples;
    pipeline->audio = pipeline->info.input_format == MIPK_FORMAT_PCM_S16LE;
    if (pipeline->task_name != NULL &&
        (strcmp(pipeline->task_name, "audio_pipeline") == 0) != pipeline->audio) {
        return manifest_error("task.type", "does not match the input format");
    }
    pipeline->task_name = pipeline->audio ? "audio_pipeline" : "vision_pipeline";
    pipeline->info.task = pipeline->task_name;
    pipeline->config.input_format = pipeline->info.input_format;
    pipeline->config.model_input_width = pipeline->info.input_width;
    pipeline->config.model_input_height = pipeline->info.input_height;
    pipeline->config.model_input_size = pipeline->info.input_size;
    pipeline->config.resized_width = pipeline->info.input_width;
    pipeline->config.resized_height = pipeline->info.input_height;
    pipeline->config.to_source[0] = 1.0F;
    pipeline->config.to_source[4] = 1.0F;
    pipeline->config.to_source[8] = 1.0F;
    return MIPK_OK;
}

static char *join_path(const char *directory, const char *file)
{
    const size_t directory_length = strlen(directory);
    const size_t file_length = strlen(file);
    char *path = malloc(directory_length + file_length + 2U);
    if (path == NULL) return NULL;
    memcpy(path, directory, directory_length);
    path[directory_length] = '/';
    memcpy(path + directory_length + 1U, file, file_length + 1U);
    return path;
}

static mipk_status parse_models(const cJSON *root, mipk_pipeline *pipeline)
{
    const cJSON *models = cJSON_GetObjectItemCaseSensitive(root, "models");
    const int count = cJSON_GetArraySize(models);
    const bool packed = pipeline->model_sources != NULL;
    if (!cJSON_IsArray(models) || count <= 0 || count > 64 ||
        (packed && (size_t)count != pipeline->model_source_count)) {
        return manifest_error("models",
            "must match the package model entries");
    }
    pipeline->models = calloc((size_t)count, sizeof(*pipeline->models));
    if (pipeline->models == NULL) return MIPK_ERR_MEMORY;
    pipeline->model_count = (size_t)count;
    pipeline->config.models = pipeline->models;
    pipeline->config.model_count = pipeline->model_count;
    size_t total_model_bytes = 0;
    for (int index = 0; index < count; ++index) {
        const cJSON *model = cJSON_GetArrayItem(models, index);
        static const char *const fields[] = {"id", "backend", "file", "kind"};
        mipk_status field_status = require_exact_fields(model, fields,
            sizeof(fields) / sizeof(fields[0]), "models[]");
        if (field_status != MIPK_OK) return field_status;
        const cJSON *id = NULL;
        const cJSON *backend = NULL;
        const cJSON *file = NULL;
        const cJSON *kind = cJSON_GetObjectItemCaseSensitive(model, "kind");
        if (!cJSON_IsObject(model) || !json_string(model, "id", &id) ||
            !json_string(model, "backend", &backend) ||
            !json_string(model, "file", &file) ||
            !safe_relative_path(file->valuestring) ||
            (kind != NULL && (!cJSON_IsString(kind) ||
                kind->valuestring == NULL || kind->valuestring[0] == '\0'))) {
            return manifest_error("models[]",
                "requires id, backend, safe file, and optional kind");
        }
        for (int previous = 0; previous < index; ++previous) {
            if (strcmp(id->valuestring, pipeline->models[previous].id) == 0) {
                return manifest_error("models[].id", "must be unique");
            }
        }
        mipk_model_config *config = &pipeline->models[index];
        config->id = strdup(id->valuestring);
        config->backend_name = strdup(backend->valuestring);
        config->kind = kind == NULL ? NULL : strdup(kind->valuestring);
        config->options_json = cJSON_PrintUnformatted(model);
        if (packed) {
            config->model_size = pipeline->model_sources[index].size;
            config->model_loader = mipk_container_load_model;
            config->model_loader_context = &pipeline->model_sources[index];
        } else {
            struct stat attributes;
            config->model_path = join_path(pipeline->package_path,
                file->valuestring);
            if (config->model_path == NULL) return MIPK_ERR_MEMORY;
            if (stat(config->model_path, &attributes) != 0 ||
                !S_ISREG(attributes.st_mode) || attributes.st_size <= 0) {
                return manifest_error("models[].file",
                    "must name a model file next to manifest.json");
            }
            config->model_size = (size_t)attributes.st_size;
        }
        if (config->model_size > MIPK_MAX_MODEL_BYTES - total_model_bytes) {
            return manifest_error("models[]", "total model bytes exceed device limit");
        }
        total_model_bytes += config->model_size;
        if (config->id == NULL || config->backend_name == NULL ||
            (kind != NULL && config->kind == NULL) ||
            config->options_json == NULL) return MIPK_ERR_MEMORY;
    }
    return MIPK_OK;
}

static bool value_available(const mipk_pipeline *pipeline, size_t stage_count,
    const char *name)
{
    if (strcmp(name, "$input") == 0) return true;
    for (size_t index = 0; index < stage_count; ++index) {
        if (strcmp(name, pipeline->stages[index].output) == 0) return true;
    }
    return false;
}

static bool find_model(const mipk_pipeline *pipeline, const char *id,
    size_t *model_index)
{
    for (size_t index = 0; index < pipeline->model_count; ++index) {
        if (strcmp(id, pipeline->models[index].id) == 0) {
            *model_index = index;
            return true;
        }
    }
    return false;
}

static bool find_label_set(const mipk_pipeline *pipeline, const char *name,
    uint32_t *label_set_id)
{
    for (size_t index = 0; index < pipeline->label_set_count; ++index) {
        if (strcmp(name, pipeline->label_sets[index].name) == 0) {
            *label_set_id = (uint32_t)index;
            return true;
        }
    }
    return false;
}

static bool decimal_port(const char *value)
{
    if (value == NULL || value[0] == '\0') return false;
    for (const char *cursor = value; *cursor != '\0'; ++cursor) {
        if (*cursor < '0' || *cursor > '9') return false;
    }
    return true;
}

static mipk_status parse_stage_bindings(const cJSON *stage_json,
    mipk_pipeline *pipeline, size_t stage_index)
{
    mipk_stage_config *stage = &pipeline->stages[stage_index];
    const cJSON *inputs = cJSON_GetObjectItemCaseSensitive(stage_json, "inputs");
    const cJSON *output = NULL;
    const int count = cJSON_GetArraySize(inputs);
    if (!cJSON_IsObject(inputs) || count <= 0 || count > 8 ||
        !json_string(stage_json, "output", &output) ||
        strcmp(output->valuestring, "$input") == 0 ||
        value_available(pipeline, stage_index, output->valuestring)) {
        return manifest_error("pipeline[].inputs/output",
            "requires available inputs and a unique output");
    }
    stage->inputs = calloc((size_t)count, sizeof(*stage->inputs));
    if (stage->inputs == NULL) return MIPK_ERR_MEMORY;
    stage->input_count = (size_t)count;
    for (int index = 0; index < count; ++index) {
        const cJSON *input = cJSON_GetArrayItem(inputs, index);
        const char *port_name = input == NULL ? NULL : input->string;
        const char *source_name = NULL;
        bool select_tensor = false;
        size_t source_tensor = 0;
        if (cJSON_IsString(input)) {
            source_name = input->valuestring;
        } else if (cJSON_IsObject(input)) {
            static const char *const fields[] = {"source", "tensor_index"};
            const cJSON *source = NULL;
            if (!object_has_exact_fields(input, fields,
                    sizeof(fields) / sizeof(fields[0])) ||
                cJSON_GetArraySize(input) != 2 ||
                !json_string(input, "source", &source) ||
                !json_size(input, "tensor_index", 0, SIZE_MAX,
                    &source_tensor)) {
                return manifest_error("pipeline[].inputs",
                    "tensor binding requires source and tensor_index");
            }
            source_name = source->valuestring;
            select_tensor = true;
        }
        if (source_name == NULL || port_name == NULL || port_name[0] == '\0' ||
            !value_available(pipeline, stage_index, source_name)) {
            return manifest_error("pipeline[].inputs",
                "must reference $input or an earlier stage output");
        }
        for (int previous = 0; previous < index; ++previous) {
            if (strcmp(port_name, stage->inputs[previous].port) == 0) {
                return manifest_error("pipeline[].inputs", "ports must be unique");
            }
        }
        stage->inputs[index].port = strdup(port_name);
        stage->inputs[index].source = strdup(source_name);
        stage->inputs[index].select_tensor = select_tensor;
        stage->inputs[index].tensor_index = source_tensor;
        if (stage->inputs[index].port == NULL ||
            stage->inputs[index].source == NULL) return MIPK_ERR_MEMORY;
    }
    stage->output = strdup(output->valuestring);
    return stage->output == NULL ? MIPK_ERR_MEMORY : MIPK_OK;
}

static mipk_status parse_image_preprocess(const cJSON *stage,
    mipk_pipeline *pipeline, bool *configured)
{
    size_t width = pipeline->info.input_width;
    size_t height = pipeline->info.input_height;
    if (!json_size(stage, "target_width", width, UINT32_MAX, &width) ||
        !json_size(stage, "target_height", height, UINT32_MAX, &height) ||
        width == 0 || height == 0) {
        return manifest_error("pipeline[].image_preprocess",
            "invalid target dimensions");
    }
    const cJSON *mode_json = cJSON_GetObjectItemCaseSensitive(stage, "mode");
    const char *mode = mode_json == NULL ? "stretch" :
        cJSON_IsString(mode_json) ? mode_json->valuestring : NULL;
    const bool stretch = mode != NULL && strcmp(mode, "stretch") == 0;
    const bool letterbox = mode != NULL && strcmp(mode, "letterbox") == 0;
    const bool yuv = pipeline->info.input_format == MIPK_FORMAT_NV12 ||
        pipeline->info.input_format == MIPK_FORMAT_NV21;
    if ((!stretch && !letterbox) ||
        (yuv && (((width | height) & 1U) != 0)) ||
        (!yuv && (width != pipeline->info.input_width ||
            height != pipeline->info.input_height))) {
        return manifest_error("pipeline[].image_preprocess",
            "unsupported mode, format, or target dimensions");
    }
    if (width > SIZE_MAX / height) {
        return manifest_error("pipeline[].image_preprocess",
            "target size overflow");
    }
    const size_t pixels = width * height;
    if (
        (yuv && pixels > SIZE_MAX - pixels / 2U) ||
        (!yuv && pixels > SIZE_MAX / 3U)) {
        return manifest_error("pipeline[].image_preprocess",
            "target size overflow");
    }
    uint32_t resized_width = (uint32_t)width;
    uint32_t resized_height = (uint32_t)height;
    uint32_t pad_x = 0;
    uint32_t pad_y = 0;
    if (letterbox && !mipk_preprocess_letterbox(pipeline->info.input_width,
        pipeline->info.input_height, (uint32_t)width, (uint32_t)height, yuv,
        &resized_width, &resized_height, &pad_x, &pad_y)) {
        return manifest_error("pipeline[].image_preprocess",
            "letterbox result is empty");
    }
    if (*configured) return MIPK_OK;
    pipeline->config.model_input_width = (uint32_t)width;
    pipeline->config.model_input_height = (uint32_t)height;
    pipeline->config.model_input_size = yuv ? pixels + pixels / 2U :
        pixels * 3U;
    pipeline->config.resized_width = resized_width;
    pipeline->config.resized_height = resized_height;
    pipeline->config.pad_x = pad_x;
    pipeline->config.pad_y = pad_y;
    memset(pipeline->config.to_source, 0, sizeof(pipeline->config.to_source));
    pipeline->config.to_source[0] = (float)pipeline->info.input_width /
        resized_width;
    pipeline->config.to_source[2] =
        -(float)pad_x * pipeline->config.to_source[0];
    pipeline->config.to_source[4] = (float)pipeline->info.input_height /
        resized_height;
    pipeline->config.to_source[5] =
        -(float)pad_y * pipeline->config.to_source[4];
    pipeline->config.to_source[8] = 1.0F;
    *configured = true;
    return MIPK_OK;
}

static bool field_name_in(const char *name, const char *const *fields,
    size_t count)
{
    for (size_t index = 0; index < count; ++index) {
        if (strcmp(name, fields[index]) == 0) return true;
    }
    return false;
}

static const mipk_node_schema *manifest_stage_schema(mipk_stage_type type,
    const char *name)
{
    if (type == MIPK_STAGE_OPERATOR) {
        const mipk_operator_ops *ops = mipk_operator_find(name);
        return ops == NULL ? NULL : ops->schema;
    }
    if (type == MIPK_STAGE_POSTPROCESS) {
        const mipk_postprocess_ops *ops = mipk_postprocess_find(name);
        return ops == NULL ? NULL : ops->schema;
    }
    return NULL;
}

static mipk_status validate_stage_fields(const cJSON *object,
    mipk_stage_type type, const char *name)
{
    static const char *const common[] = {
        "id", "type", "inputs", "output", "execution",
    };
    static const char *const model[] = {"model", "metadata_from"};
    const mipk_node_schema *schema = manifest_stage_schema(type, name);
    if (type != MIPK_STAGE_MODEL && schema == NULL) {
        mipk_diag("MIPK %s '%s' is not registered; enable its module "
            "(see `mipk_tool.py modules`) or register it before opening the "
            "pipeline",
            type == MIPK_STAGE_OPERATOR ? "operator" : "postprocess", name);
        return MIPK_ERR_UNSUPPORTED;
    }
    for (const cJSON *field = object->child; field != NULL;
        field = field->next) {
        if (field->string == NULL || field->string[0] == '\0') {
            return manifest_error("pipeline[]", "contains an unnamed field");
        }
        bool allowed = field_name_in(field->string, common,
            sizeof(common) / sizeof(common[0]));
        if (type == MIPK_STAGE_MODEL) {
            allowed = allowed || field_name_in(field->string, model,
                sizeof(model) / sizeof(model[0]));
        } else {
            const char *identity = type == MIPK_STAGE_OPERATOR ? "operator" :
                "postprocess";
            allowed = allowed || strcmp(field->string, identity) == 0 ||
                (type == MIPK_STAGE_POSTPROCESS &&
                    strcmp(field->string, "label_set") == 0) ||
                field_name_in(field->string, schema->option_names,
                    schema->option_count);
        }
        if (!allowed) {
            return manifest_error("pipeline[]", "contains an unknown field");
        }
        for (const cJSON *other = field->next; other != NULL;
            other = other->next) {
            if (other->string != NULL &&
                strcmp(field->string, other->string) == 0) {
                return manifest_error("pipeline[]", "contains a duplicate field");
            }
        }
    }
    return MIPK_OK;
}

static mipk_status parse_pipeline(const cJSON *root, mipk_pipeline *pipeline)
{
    const cJSON *stages = cJSON_GetObjectItemCaseSensitive(root, "pipeline");
    const int count = cJSON_GetArraySize(stages);
    if (!cJSON_IsArray(stages) || count < 1 || count > 32) {
        return manifest_error("pipeline", "must contain 1 to 32 stages");
    }
    pipeline->stages = calloc((size_t)count, sizeof(*pipeline->stages));
    if (pipeline->stages == NULL) return MIPK_ERR_MEMORY;
    pipeline->stage_count = (size_t)count;
    pipeline->config.stages = pipeline->stages;
    pipeline->config.stage_count = pipeline->stage_count;
    bool image_preprocess_configured = false;
    for (int index = 0; index < count; ++index) {
        const cJSON *stage_json = cJSON_GetArrayItem(stages, index);
        const cJSON *id = NULL;
        const cJSON *type = NULL;
        const cJSON *name = NULL;
        if (!cJSON_IsObject(stage_json) ||
            (cJSON_GetObjectItemCaseSensitive(stage_json, "id") != NULL &&
                !json_string(stage_json, "id", &id)) ||
            !json_string(stage_json, "type", &type)) {
            return manifest_error("pipeline[]", "stage requires a type");
        }
        /* A node is named after its output unless the manifest says otherwise. */
        if (id == NULL && !json_string(stage_json, "output", &id)) {
            return manifest_error("pipeline[]", "stage requires an output name");
        }
        for (int previous = 0; previous < index; ++previous) {
            if (strcmp(pipeline->stages[previous].id, id->valuestring) == 0) {
                return manifest_error("pipeline[].id", "must be unique");
            }
        }
        mipk_stage_config *stage = &pipeline->stages[index];
        stage->id = strdup(id->valuestring);
        if (stage->id == NULL) return MIPK_ERR_MEMORY;
        if (strcmp(type->valuestring, "operator") == 0 &&
            json_string(stage_json, "operator", &name)) {
            stage->type = MIPK_STAGE_OPERATOR;
        } else if (strcmp(type->valuestring, "model") == 0 &&
            json_string(stage_json, "model", &name) &&
            find_model(pipeline, name->valuestring, &stage->model_index)) {
            stage->type = MIPK_STAGE_MODEL;
            stage->backend_name =
                pipeline->models[stage->model_index].backend_name;
            stage->model_kind = pipeline->models[stage->model_index].kind;
        } else if (strcmp(type->valuestring, "postprocess") == 0 &&
            json_string(stage_json, "postprocess", &name)) {
            stage->type = MIPK_STAGE_POSTPROCESS;
        } else {
            return manifest_error("pipeline[]", "invalid stage or model reference");
        }
        stage->name = strdup(name->valuestring);
        mipk_status status = validate_stage_fields(stage_json, stage->type,
            name->valuestring);
        if (status != MIPK_OK) return status;
        stage->options_json = cJSON_PrintUnformatted(stage_json);
        if (stage->name == NULL || stage->options_json == NULL) {
            return MIPK_ERR_MEMORY;
        }
        status = parse_stage_bindings(stage_json, pipeline, (size_t)index);
        if (status != MIPK_OK) return status;
        stage->label_set_id = MIPK_INVALID_LABEL_SET_ID;
        if (stage->type == MIPK_STAGE_POSTPROCESS) {
            const cJSON *label_set = NULL;
            if (!json_string(stage_json, "label_set", &label_set) ||
                !find_label_set(pipeline, label_set->valuestring,
                    &stage->label_set_id)) {
                return manifest_error("pipeline[].label_set",
                    "postprocess must reference a label set");
            }
        }
        const cJSON *execution =
            cJSON_GetObjectItemCaseSensitive(stage_json, "execution");
        if (execution == NULL) {
            stage->execution = MIPK_EXECUTION_ONCE;
        } else {
            const cJSON *mode = NULL;
            if (!cJSON_IsObject(execution) ||
                !json_string(execution, "mode", &mode)) {
                return manifest_error("pipeline[].execution",
                    "must be an object with mode");
            }
            static const char *const fields[] = {"mode", "max_items"};
            status = require_exact_fields(execution, fields,
                sizeof(fields) / sizeof(fields[0]), "pipeline[].execution");
            if (status != MIPK_OK) return status;
            if (strcmp(mode->valuestring, "once") == 0 &&
                cJSON_GetObjectItemCaseSensitive(execution,
                    "max_items") == NULL) {
                stage->execution = MIPK_EXECUTION_ONCE;
            } else if (strcmp(mode->valuestring, "per_item") == 0 &&
                json_size(execution, "max_items", 0, SIZE_MAX,
                    &stage->max_items) && stage->max_items != 0) {
                stage->execution = MIPK_EXECUTION_PER_ITEM;
            } else {
                return manifest_error("pipeline[].execution",
                    "invalid mode or max_items");
            }
        }
        if (stage->type == MIPK_STAGE_MODEL) {
            for (size_t input = 0; input < stage->input_count; ++input) {
                if (!decimal_port(stage->inputs[input].port)) {
                    return manifest_error("pipeline[].inputs",
                        "model input ports must be tensor indices");
                }
            }
        } else {
            for (size_t input = 0; input < stage->input_count; ++input) {
                if (stage->inputs[input].select_tensor) {
                    return manifest_error("pipeline[].inputs",
                        "tensor_index is only valid for model stages");
                }
            }
        }
        if (stage->type == MIPK_STAGE_OPERATOR &&
            strcmp(stage->name, "tfmicrofrontend") == 0) {
            size_t step_ms = 0;
            size_t inference_stride = 0;
            if (!json_size(stage_json, "step_ms", 10, UINT32_MAX, &step_ms) ||
                !json_size(stage_json, "inference_stride", 1, UINT32_MAX,
                    &inference_stride) || step_ms == 0 ||
                inference_stride == 0 ||
                step_ms > UINT32_MAX / inference_stride) {
                return manifest_error("pipeline[].tfmicrofrontend",
                    "invalid inference interval");
            }
            pipeline->config.inference_interval_ms =
                (uint32_t)(step_ms * inference_stride);
        } else if (stage->type == MIPK_STAGE_OPERATOR &&
            strcmp(stage->name, "image_preprocess") == 0) {
            status = parse_image_preprocess(stage_json, pipeline,
                &image_preprocess_configured);
            if (status != MIPK_OK) return status;
        }
    }
    size_t *model_references = calloc(pipeline->model_count,
        sizeof(*model_references));
    if (model_references == NULL) return MIPK_ERR_MEMORY;
    for (size_t index = 0; index < pipeline->stage_count; ++index) {
        if (pipeline->stages[index].type != MIPK_STAGE_MODEL) continue;
        ++model_references[pipeline->stages[index].model_index];
    }
    for (size_t index = 0; index < pipeline->model_count; ++index) {
        if (model_references[index] != 1) {
            free(model_references);
            return manifest_error("pipeline[].model",
                "must reference every model exactly once");
        }
    }
    free(model_references);
    const cJSON *output = NULL;
    if (!json_string(root, "output", &output) ||
        !value_available(pipeline, pipeline->stage_count, output->valuestring) ||
        strcmp(output->valuestring, "$input") == 0) {
        return manifest_error("output", "must select a pipeline stage output");
    }
    pipeline->output = strdup(output->valuestring);
    if (pipeline->output == NULL) return MIPK_ERR_MEMORY;
    pipeline->config.output = pipeline->output;
    return MIPK_OK;
}

static bool capability_kind_from_string(const char *name,
    mipk_capability_kind *kind)
{
    if (strcmp(name, "backend") == 0) *kind = MIPK_CAPABILITY_BACKEND;
    else if (strcmp(name, "operator") == 0) *kind = MIPK_CAPABILITY_OPERATOR;
    else if (strcmp(name, "postprocess") == 0) {
        *kind = MIPK_CAPABILITY_POSTPROCESS;
    } else {
        return false;
    }
    return true;
}

static bool capability_used(const mipk_pipeline *pipeline,
    mipk_capability_kind kind, const char *name)
{
    if (kind == MIPK_CAPABILITY_BACKEND) {
        for (size_t index = 0; index < pipeline->model_count; ++index) {
            if (strcmp(pipeline->models[index].backend_name, name) == 0) {
                return true;
            }
        }
        return false;
    }
    for (size_t index = 0; index < pipeline->stage_count; ++index) {
        const mipk_stage_config *stage = &pipeline->stages[index];
        if ((kind == MIPK_CAPABILITY_OPERATOR &&
                stage->type == MIPK_STAGE_OPERATOR) ||
            (kind == MIPK_CAPABILITY_POSTPROCESS &&
                stage->type == MIPK_STAGE_POSTPROCESS)) {
            if (strcmp(stage->name, name) == 0) return true;
        }
    }
    return false;
}

static bool requirement_fields_valid(const cJSON *requirement)
{
    if (!cJSON_IsObject(requirement) ||
        cJSON_GetArraySize(requirement) != 4) return false;
    for (const cJSON *field = requirement->child; field != NULL;
        field = field->next) {
        if (field->string == NULL ||
            (strcmp(field->string, "kind") != 0 &&
             strcmp(field->string, "name") != 0 &&
             strcmp(field->string, "abi_version") != 0 &&
             strcmp(field->string, "min_implementation_version") != 0)) {
            return false;
        }
    }
    return true;
}

static bool requirement_identity(const cJSON *requirement,
    mipk_capability_kind *kind, const char **name)
{
    const cJSON *kind_json = NULL;
    const cJSON *name_json = NULL;
    if (!requirement_fields_valid(requirement) ||
        !json_string(requirement, "kind", &kind_json) ||
        !json_string(requirement, "name", &name_json) ||
        !capability_kind_from_string(kind_json->valuestring, kind)) {
        return false;
    }
    *name = name_json->valuestring;
    return true;
}

static mipk_status validate_capability_requirements(const cJSON *root,
    const mipk_pipeline *pipeline)
{
    const cJSON *requires = cJSON_GetObjectItemCaseSensitive(root, "requires");
    const cJSON *requirements = requires == NULL ? NULL :
        cJSON_GetObjectItemCaseSensitive(requires, "capabilities");
    const int requirement_count = requirements == NULL ? 0 :
        cJSON_GetArraySize(requirements);
    for (int index = 0; index < requirement_count; ++index) {
        const cJSON *requirement = cJSON_GetArrayItem(requirements, index);
        mipk_capability_kind kind;
        const char *name = NULL;
        if (!requirement_identity(requirement, &kind, &name)) {
            return manifest_error("requires.capabilities[]",
                "invalid fields or capability identity");
        }
        for (int previous = 0; previous < index; ++previous) {
            mipk_capability_kind previous_kind;
            const char *previous_name = NULL;
            if (!requirement_identity(cJSON_GetArrayItem(requirements,
                    previous), &previous_kind, &previous_name) ||
                (previous_kind == kind && strcmp(previous_name, name) == 0)) {
                return manifest_error("requires.capabilities[]",
                    "capability identities must be unique");
            }
        }
        const cJSON *abi = cJSON_GetObjectItemCaseSensitive(requirement,
            "abi_version");
        const cJSON *minimum = NULL;
        const mipk_capability_descriptor *capability =
            mipk_capability_find(kind, name);
        if (!cJSON_IsNumber(abi) || !isfinite(abi->valuedouble) ||
            abi->valuedouble < 1.0 || abi->valuedouble > UINT32_MAX ||
            floor(abi->valuedouble) != abi->valuedouble ||
            !json_string(requirement, "min_implementation_version",
                &minimum) || capability == NULL ||
            capability->abi_version != (uint32_t)abi->valuedouble ||
            !mipk_capability_version_at_least(
                capability->implementation_version, minimum->valuestring)) {
            return manifest_error("requires.capabilities[]",
                "capability is missing or has an incompatible version");
        }
        if (!capability_used(pipeline, kind, name)) {
            return manifest_error("requires.capabilities[]",
                "unused capabilities must not be declared");
        }
    }
    /*
     * Every node and backend the pipeline names has to be registered.  A
     * requirement entry pins its version; a node without one only has to
     * exist, so a manifest stays valid across module version bumps it does
     * not care about.
     */
    for (size_t index = 0; index < pipeline->model_count; ++index) {
        if (mipk_capability_find(MIPK_CAPABILITY_BACKEND,
                pipeline->models[index].backend_name) == NULL) {
            mipk_diag("MIPK backend '%s' is not registered",
                pipeline->models[index].backend_name);
            return MIPK_ERR_UNSUPPORTED;
        }
    }
    for (size_t stage_index = 0; stage_index < pipeline->stage_count;
        ++stage_index) {
        const mipk_stage_config *stage = &pipeline->stages[stage_index];
        const mipk_capability_kind kind = stage->type == MIPK_STAGE_OPERATOR ?
            MIPK_CAPABILITY_OPERATOR : MIPK_CAPABILITY_POSTPROCESS;
        if (stage->type == MIPK_STAGE_MODEL) continue;
        if (mipk_capability_find(kind, stage->name) == NULL) {
            mipk_diag("MIPK %s '%s' is not registered; enable its "
                "module or register it before opening the pipeline",
                kind == MIPK_CAPABILITY_OPERATOR ? "operator" : "postprocess",
                stage->name);
            return MIPK_ERR_UNSUPPORTED;
        }
    }
    return MIPK_OK;
}

mipk_status mipk_manifest_parse(const char *content, mipk_pipeline *pipeline)
{
    const char *end = NULL;
    cJSON *root = cJSON_ParseWithOpts(content, &end, true);
    if (!cJSON_IsObject(root)) {
        cJSON_Delete(root);
        return manifest_error("$", "invalid JSON document");
    }
    const cJSON *name = NULL;
    const cJSON *version = NULL;
    const cJSON *schema = NULL;
    const cJSON *runtime_version = NULL;
    const cJSON *requires = cJSON_GetObjectItemCaseSensitive(root, "requires");
    size_t format_version = 0;
    mipk_status status = MIPK_OK;
    static const char *const root_fields[] = {
        "format_version", "schema", "name", "version", "requires", "task",
        "models", "input", "label_sets", "pipeline", "output",
    };
    static const char *const requires_fields[] = {
        "runtime_min_version", "capabilities",
    };
    /*
     * Only the graph itself is mandatory.  name, version, requires and task
     * have defaults so a hand-written manifest carries no boilerplate; the
     * packaging tool fills them in when it builds a .mipk.
     */
    status = require_exact_fields(root, root_fields,
        sizeof(root_fields) / sizeof(root_fields[0]), "$");
    if (status == MIPK_OK && requires != NULL) {
        status = require_exact_fields(requires, requires_fields,
            sizeof(requires_fields) / sizeof(requires_fields[0]), "requires");
    }
    if (status != MIPK_OK) {
        /* Error already reported by require_exact_fields(). */
    } else if (!json_size(root, "format_version", 0, UINT32_MAX,
            &format_version) ||
        format_version != MIPK_FORMAT_VERSION) {
        status = manifest_error("format_version", "must be 2");
    } else if (!json_string(root, "schema", &schema) ||
        strcmp(schema->valuestring, "mipk.unified_pipeline.v2") != 0) {
        status = manifest_error("schema", "must be mipk.unified_pipeline.v2");
    } else if (requires != NULL && (!cJSON_IsObject(requires) ||
        (cJSON_GetObjectItemCaseSensitive(requires, "runtime_min_version") != NULL &&
            (!json_string(requires, "runtime_min_version", &runtime_version) ||
             !mipk_capability_version_at_least(MIPK_RUNTIME_VERSION,
                runtime_version->valuestring))) ||
        (cJSON_GetObjectItemCaseSensitive(requires, "capabilities") != NULL &&
            !cJSON_IsArray(cJSON_GetObjectItemCaseSensitive(requires,
                "capabilities"))))) {
        status = manifest_error("requires", "invalid runtime or capabilities");
    } else if ((cJSON_GetObjectItemCaseSensitive(root, "name") != NULL &&
            !json_string(root, "name", &name)) ||
        (cJSON_GetObjectItemCaseSensitive(root, "version") != NULL &&
            (!json_string(root, "version", &version) ||
             !mipk_capability_version_at_least(version->valuestring,
                version->valuestring)))) {
        status = manifest_error("name/version", "invalid metadata");
    }
    if (status == MIPK_OK) {
        pipeline->name = strdup(name != NULL ? name->valuestring : "pipeline");
        pipeline->version = strdup(version != NULL ? version->valuestring :
            "1.0.0");
        if (pipeline->name == NULL || pipeline->version == NULL) {
            status = MIPK_ERR_MEMORY;
        }
    }
    if (status == MIPK_OK) status = parse_task(root, pipeline);
    if (status == MIPK_OK) status = parse_input(root, pipeline);
    if (status == MIPK_OK &&
        cJSON_GetObjectItemCaseSensitive(root, "classes") != NULL) {
        status = manifest_error("classes", "removed; use label_sets");
    }
    if (status == MIPK_OK) status = parse_label_sets(root, pipeline);
    if (status == MIPK_OK) status = parse_models(root, pipeline);
    if (status == MIPK_OK) status = parse_pipeline(root, pipeline);
    if (status == MIPK_OK) status = mipk_capability_registry_freeze();
    if (status == MIPK_OK) {
        status = validate_capability_requirements(root, pipeline);
    }
    if (status == MIPK_OK) {
        pipeline->info.name = pipeline->name;
        pipeline->info.version = pipeline->version;
    }
    cJSON_Delete(root);
    return status;
}

bool mipk_manifest_find_label_set(const mipk_pipeline *pipeline,
    const char *name, uint32_t *label_set_id)
{
    return find_label_set(pipeline, name, label_set_id);
}

void mipk_manifest_free(mipk_pipeline *pipeline)
{
    for (size_t set = 0; set < pipeline->label_set_count; ++set) {
        for (size_t index = 0;
            index < pipeline->label_sets[set].class_count; ++index) {
            free((char *)pipeline->label_sets[set].classes[index].name);
        }
        free(pipeline->label_sets[set].classes);
        free((char *)pipeline->label_sets[set].name);
    }
    free(pipeline->label_sets);
    pipeline->label_sets = NULL;
    pipeline->label_set_count = 0;
    for (size_t index = 0; index < pipeline->stage_count; ++index) {
        for (size_t input = 0; input < pipeline->stages[index].input_count;
            ++input) {
            free((char *)pipeline->stages[index].inputs[input].port);
            free((char *)pipeline->stages[index].inputs[input].source);
        }
        free(pipeline->stages[index].inputs);
        free((char *)pipeline->stages[index].id);
        free((char *)pipeline->stages[index].output);
        free((char *)pipeline->stages[index].options_json);
        free((char *)pipeline->stages[index].name);
    }
    free(pipeline->stages);
    pipeline->stages = NULL;
    pipeline->stage_count = 0;
    free(pipeline->output);
    pipeline->task_name = NULL;
    pipeline->output = NULL;
    for (size_t index = 0; index < pipeline->model_count; ++index) {
        free((char *)pipeline->models[index].options_json);
        free((char *)pipeline->models[index].model_path);
        free((char *)pipeline->models[index].kind);
        free((char *)pipeline->models[index].backend_name);
        free((char *)pipeline->models[index].id);
    }
    free(pipeline->models);
    pipeline->models = NULL;
    pipeline->model_count = 0;
    free(pipeline->version);
    free(pipeline->name);
    pipeline->version = NULL;
    pipeline->name = NULL;
}
