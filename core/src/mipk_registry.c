#include "mipk_pipeline_internal.h"

#include "mipk_backend.h"
#include "mipk_module.h"
#include "cJSON.h"

#include <math.h>
#include <pthread.h>
#include <string.h>

/*
 * Every backend, operator and postprocess the runtime knows about is
 * registered here, by the module that implements it.  The runtime itself
 * ships nothing: the build generates mipk_register_builtin_modules() from
 * the enabled module list, and applications may register more before the
 * first pipeline opens.  Registered ops and their capability descriptors
 * have static lifetime.
 */
#define MIPK_MAX_POSTPROCESSES 32U
#define MIPK_MAX_OPERATORS 32U
#define MIPK_MAX_BACKENDS 8U

static pthread_mutex_t g_registry_lock = PTHREAD_MUTEX_INITIALIZER;
static const mipk_postprocess_ops *g_postprocessors[MIPK_MAX_POSTPROCESSES];
static size_t g_postprocess_count;
static const mipk_operator_ops *g_operators[MIPK_MAX_OPERATORS];
static size_t g_operator_count;
static const mipk_backend_ops *g_backends[MIPK_MAX_BACKENDS];
static size_t g_backend_count;
static bool g_registry_frozen;

static bool semver_component(const char **cursor, uint32_t *value)
{
    const char *text = *cursor;
    if (text[0] < '0' || text[0] > '9' ||
        (text[0] == '0' && text[1] >= '0' && text[1] <= '9')) {
        return false;
    }
    uint32_t parsed = 0;
    do {
        const uint32_t digit = (uint32_t)(*text - '0');
        if (parsed > (UINT32_MAX - digit) / 10U) return false;
        parsed = parsed * 10U + digit;
        ++text;
    } while (*text >= '0' && *text <= '9');
    *cursor = text;
    *value = parsed;
    return true;
}

static bool parse_semver(const char *version, uint32_t values[3])
{
    if (version == NULL) return false;
    const char *cursor = version;
    for (size_t index = 0; index < 3; ++index) {
        if (!semver_component(&cursor, &values[index])) return false;
        if (index < 2) {
            if (*cursor != '.') return false;
            ++cursor;
        }
    }
    return *cursor == '\0';
}

bool mipk_capability_version_at_least(const char *actual,
    const char *minimum)
{
    uint32_t actual_values[3];
    uint32_t minimum_values[3];
    if (!parse_semver(actual, actual_values) ||
        !parse_semver(minimum, minimum_values)) return false;
    for (size_t index = 0; index < 3; ++index) {
        if (actual_values[index] != minimum_values[index]) {
            return actual_values[index] > minimum_values[index];
        }
    }
    return true;
}

static bool descriptor_valid(const mipk_capability_descriptor *capability,
    mipk_capability_kind kind, const char *name)
{
    uint32_t version[3];
    const uint32_t expected_abi = kind == MIPK_CAPABILITY_BACKEND ?
        MIPK_BACKEND_ABI_VERSION : kind == MIPK_CAPABILITY_OPERATOR ?
        MIPK_OPERATOR_ABI_VERSION : MIPK_POSTPROCESS_ABI_VERSION;
    return capability != NULL &&
        capability->struct_size == sizeof(*capability) &&
        capability->kind == kind && capability->name != NULL &&
        capability->name[0] != '\0' && name != NULL &&
        strcmp(capability->name, name) == 0 &&
        capability->abi_version == expected_abi &&
        parse_semver(capability->implementation_version, version);
}

static bool option_default_valid(const mipk_runtime_option *option)
{
    switch (option->type) {
        case MIPK_RUNTIME_OPTION_BOOL:
        case MIPK_RUNTIME_OPTION_UINT32:
        case MIPK_RUNTIME_OPTION_SIZE:
        case MIPK_RUNTIME_OPTION_FLOAT:
        case MIPK_RUNTIME_OPTION_STRING:
        case MIPK_RUNTIME_OPTION_STRING_ARRAY:
        case MIPK_RUNTIME_OPTION_JSON:
            break;
        default:
            return false;
    }
    if (option->default_json == NULL) return true;
    const char *end = NULL;
    cJSON *json = cJSON_ParseWithOpts(option->default_json, &end, true);
    bool valid = json != NULL && end != NULL && *end == '\0';
    if (valid) {
        switch (option->type) {
            case MIPK_RUNTIME_OPTION_BOOL:
                valid = cJSON_IsBool(json);
                break;
            case MIPK_RUNTIME_OPTION_UINT32:
                valid = cJSON_IsNumber(json) && isfinite(json->valuedouble) &&
                    json->valuedouble >= 0.0 &&
                    json->valuedouble <= (double)UINT32_MAX &&
                    floor(json->valuedouble) == json->valuedouble;
                break;
            case MIPK_RUNTIME_OPTION_SIZE:
                valid = cJSON_IsNumber(json) && isfinite(json->valuedouble) &&
                    json->valuedouble >= 0.0 &&
                    json->valuedouble <= (double)SIZE_MAX &&
                    floor(json->valuedouble) == json->valuedouble;
                break;
            case MIPK_RUNTIME_OPTION_FLOAT:
                valid = cJSON_IsNumber(json) && isfinite(json->valuedouble);
                break;
            case MIPK_RUNTIME_OPTION_STRING:
                valid = cJSON_IsString(json) && json->valuestring != NULL;
                break;
            case MIPK_RUNTIME_OPTION_STRING_ARRAY:
                valid = cJSON_IsArray(json);
                for (const cJSON *item = json->child; valid && item != NULL;
                    item = item->next) {
                    valid = cJSON_IsString(item) && item->valuestring != NULL;
                }
                break;
            case MIPK_RUNTIME_OPTION_JSON:
                break;
            default:
                valid = false;
                break;
        }
    }
    cJSON_Delete(json);
    return valid;
}

static bool runtime_options_valid(const mipk_runtime_option *options,
    size_t count)
{
    if (count == 0) return options == NULL;
    if (options == NULL) return false;
    for (size_t index = 0; index < count; ++index) {
        const mipk_runtime_option *option = &options[index];
        if (option->struct_size != sizeof(*option) || option->name == NULL ||
            option->name[0] == '\0' || !option_default_valid(option)) {
            return false;
        }
        for (size_t previous = 0; previous < index; ++previous) {
            if (strcmp(options[previous].name, option->name) == 0) return false;
        }
    }
    return true;
}

static const mipk_capability_descriptor *registered_capability_at(
    size_t index)
{
    if (index < g_backend_count) return g_backends[index]->capability;
    index -= g_backend_count;
    if (index < g_operator_count) return g_operators[index]->capability;
    index -= g_operator_count;
    return index < g_postprocess_count ?
        g_postprocessors[index]->capability : NULL;
}

static size_t registered_capability_count(void)
{
    return g_backend_count + g_operator_count + g_postprocess_count;
}

static bool duplicate_capability(const mipk_capability_descriptor *capability)
{
    for (size_t index = 0; index < registered_capability_count(); ++index) {
        const mipk_capability_descriptor *existing =
            registered_capability_at(index);
        if (existing->kind == capability->kind &&
            strcmp(existing->name, capability->name) == 0) return true;
    }
    return false;
}

mipk_status mipk_capability_registry_freeze(void)
{
    if (pthread_mutex_lock(&g_registry_lock) != 0) return MIPK_ERR_RUNTIME;
    g_registry_frozen = true;
    (void)pthread_mutex_unlock(&g_registry_lock);
    return MIPK_OK;
}

size_t mipk_capability_count(void)
{
    if (pthread_mutex_lock(&g_registry_lock) != 0) return 0;
    const size_t count = registered_capability_count();
    (void)pthread_mutex_unlock(&g_registry_lock);
    return count;
}

const mipk_capability_descriptor *mipk_capability_at(size_t index)
{
    if (pthread_mutex_lock(&g_registry_lock) != 0) return NULL;
    const mipk_capability_descriptor *capability =
        registered_capability_at(index);
    (void)pthread_mutex_unlock(&g_registry_lock);
    return capability;
}

const mipk_capability_descriptor *mipk_capability_find(
    mipk_capability_kind kind, const char *name)
{
    if (name == NULL || pthread_mutex_lock(&g_registry_lock) != 0) return NULL;
    const mipk_capability_descriptor *found = NULL;
    const size_t total = registered_capability_count();
    for (size_t index = 0; index < total; ++index) {
        const mipk_capability_descriptor *candidate =
            registered_capability_at(index);
        if (candidate->kind == kind && strcmp(candidate->name, name) == 0) {
            found = candidate;
            break;
        }
    }
    (void)pthread_mutex_unlock(&g_registry_lock);
    return found;
}

mipk_status mipk_operator_register(const mipk_operator_ops *ops)
{
    if (ops == NULL || !descriptor_valid(ops->capability,
            MIPK_CAPABILITY_OPERATOR, ops->name) ||
        ops->create == NULL || ops->destroy == NULL || ops->validate == NULL ||
        ops->process == NULL || !runtime_options_valid(ops->runtime_options,
            ops->runtime_option_count)) return MIPK_ERR_ARGUMENT;
    if (pthread_mutex_lock(&g_registry_lock) != 0) return MIPK_ERR_RUNTIME;
    mipk_status status = g_registry_frozen ? MIPK_ERR_BUSY : MIPK_OK;
    if (status != MIPK_OK) goto done;
    for (size_t index = 0; index < g_operator_count; ++index) {
        if (strcmp(ops->name, g_operators[index]->name) != 0) continue;
        status = g_operators[index] == ops ? MIPK_OK : MIPK_ERR_ARGUMENT;
        goto done;
    }
    if (duplicate_capability(ops->capability)) {
        status = MIPK_ERR_ARGUMENT;
    } else if (g_operator_count == MIPK_MAX_OPERATORS) {
        status = MIPK_ERR_CAPACITY;
    } else {
        g_operators[g_operator_count++] = ops;
    }
done:
    (void)pthread_mutex_unlock(&g_registry_lock);
    return status;
}

const mipk_operator_ops *mipk_operator_find(const char *name)
{
    if (name == NULL || pthread_mutex_lock(&g_registry_lock) != 0) return NULL;
    const mipk_operator_ops *found = NULL;
    for (size_t index = 0; index < g_operator_count; ++index) {
        if (strcmp(name, g_operators[index]->name) == 0) {
            found = g_operators[index];
            break;
        }
    }
    (void)pthread_mutex_unlock(&g_registry_lock);
    return found;
}

mipk_status mipk_postprocess_register(const mipk_postprocess_ops *ops)
{
    if (ops == NULL || !descriptor_valid(ops->capability,
            MIPK_CAPABILITY_POSTPROCESS, ops->name) ||
        ops->create == NULL || ops->destroy == NULL || ops->validate == NULL ||
        ops->process == NULL || !runtime_options_valid(ops->runtime_options,
            ops->runtime_option_count)) return MIPK_ERR_ARGUMENT;
    if (pthread_mutex_lock(&g_registry_lock) != 0) return MIPK_ERR_RUNTIME;
    mipk_status status = g_registry_frozen ? MIPK_ERR_BUSY : MIPK_OK;
    if (status != MIPK_OK) goto done;
    for (size_t index = 0; index < g_postprocess_count; ++index) {
        if (strcmp(ops->name, g_postprocessors[index]->name) != 0) continue;
        status = g_postprocessors[index] == ops ? MIPK_OK : MIPK_ERR_ARGUMENT;
        goto done;
    }
    if (duplicate_capability(ops->capability)) {
        status = MIPK_ERR_ARGUMENT;
    } else if (g_postprocess_count == MIPK_MAX_POSTPROCESSES) {
        status = MIPK_ERR_CAPACITY;
    } else {
        g_postprocessors[g_postprocess_count++] = ops;
    }
done:
    (void)pthread_mutex_unlock(&g_registry_lock);
    return status;
}

const mipk_postprocess_ops *mipk_postprocess_find(const char *name)
{
    if (name == NULL || pthread_mutex_lock(&g_registry_lock) != 0) return NULL;
    const mipk_postprocess_ops *found = NULL;
    for (size_t index = 0; index < g_postprocess_count; ++index) {
        if (strcmp(name, g_postprocessors[index]->name) == 0) {
            found = g_postprocessors[index];
            break;
        }
    }
    (void)pthread_mutex_unlock(&g_registry_lock);
    return found;
}


mipk_status mipk_backend_register(const mipk_backend_ops *ops)
{
    if (ops == NULL || !descriptor_valid(ops->capability,
            MIPK_CAPABILITY_BACKEND, ops->name) ||
        ops->create == NULL || ops->destroy == NULL ||
        ops->input_count == NULL || ops->output_count == NULL ||
        ops->inputs == NULL || ops->outputs == NULL ||
        ops->execute == NULL) return MIPK_ERR_ARGUMENT;
    if (pthread_mutex_lock(&g_registry_lock) != 0) return MIPK_ERR_RUNTIME;
    mipk_status status = g_registry_frozen ? MIPK_ERR_BUSY : MIPK_OK;
    if (status != MIPK_OK) goto done;
    for (size_t index = 0; index < g_backend_count; ++index) {
        if (strcmp(ops->name, g_backends[index]->name) != 0) continue;
        status = g_backends[index] == ops ? MIPK_OK : MIPK_ERR_ARGUMENT;
        goto done;
    }
    if (duplicate_capability(ops->capability)) {
        status = MIPK_ERR_ARGUMENT;
    } else if (g_backend_count == MIPK_MAX_BACKENDS) {
        status = MIPK_ERR_CAPACITY;
    } else {
        g_backends[g_backend_count++] = ops;
    }
done:
    (void)pthread_mutex_unlock(&g_registry_lock);
    return status;
}

const mipk_backend_ops *mipk_backend_find(const char *name)
{
    if (name == NULL || pthread_mutex_lock(&g_registry_lock) != 0) return NULL;
    const mipk_backend_ops *found = NULL;
    for (size_t index = 0; index < g_backend_count; ++index) {
        if (strcmp(name, g_backends[index]->name) == 0) {
            found = g_backends[index];
            break;
        }
    }
    (void)pthread_mutex_unlock(&g_registry_lock);
    return found;
}

static pthread_once_t g_init_once = PTHREAD_ONCE_INIT;
static mipk_status g_init_status = MIPK_OK;

static void init_once(void)
{
    g_init_status = mipk_register_builtin_modules();
}

mipk_status mipk_init(void)
{
    if (pthread_once(&g_init_once, init_once) != 0) return MIPK_ERR_RUNTIME;
    return g_init_status;
}

static cJSON *parse_options(const char *options_json)
{
    cJSON *options = options_json == NULL ? NULL : cJSON_Parse(options_json);
    if (!cJSON_IsObject(options)) {
        cJSON_Delete(options);
        return NULL;
    }
    return options;
}

mipk_status mipk_postprocess_option_get_size(const char *options_json,
    const char *name, size_t default_value, size_t *value)
{
    if (name == NULL || name[0] == '\0' || value == NULL) {
        return MIPK_ERR_ARGUMENT;
    }
    cJSON *options = parse_options(options_json);
    if (options == NULL) return MIPK_ERR_MODEL;
    const cJSON *item = cJSON_GetObjectItemCaseSensitive(options, name);
    mipk_status status = MIPK_OK;
    if (item == NULL) {
        *value = default_value;
    } else if (!cJSON_IsNumber(item) || !isfinite(item->valuedouble) ||
        item->valuedouble < 0.0 || item->valuedouble > (double)SIZE_MAX ||
        floor(item->valuedouble) != item->valuedouble) {
        status = MIPK_ERR_MODEL;
    } else {
        *value = (size_t)item->valuedouble;
    }
    cJSON_Delete(options);
    return status;
}

mipk_status mipk_postprocess_option_get_bool(const char *options_json,
    const char *name, bool default_value, bool *value)
{
    if (name == NULL || name[0] == '\0' || value == NULL) {
        return MIPK_ERR_ARGUMENT;
    }
    cJSON *options = parse_options(options_json);
    if (options == NULL) return MIPK_ERR_MODEL;
    const cJSON *item = cJSON_GetObjectItemCaseSensitive(options, name);
    mipk_status status = MIPK_OK;
    if (item == NULL) {
        *value = default_value;
    } else if (!cJSON_IsBool(item)) {
        status = MIPK_ERR_MODEL;
    } else {
        *value = cJSON_IsTrue(item);
    }
    cJSON_Delete(options);
    return status;
}

mipk_status mipk_postprocess_option_get_float(const char *options_json,
    const char *name, float default_value, float *value)
{
    if (name == NULL || name[0] == '\0' || value == NULL) {
        return MIPK_ERR_ARGUMENT;
    }
    cJSON *options = parse_options(options_json);
    if (options == NULL) return MIPK_ERR_MODEL;
    const cJSON *item = cJSON_GetObjectItemCaseSensitive(options, name);
    mipk_status status = MIPK_OK;
    if (item == NULL) {
        *value = default_value;
    } else if (!cJSON_IsNumber(item) || !isfinite(item->valuedouble)) {
        status = MIPK_ERR_MODEL;
    } else {
        *value = (float)item->valuedouble;
    }
    cJSON_Delete(options);
    return status;
}
