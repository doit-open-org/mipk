#include "mipk_value.h"

#include <math.h>
#include <string.h>

static bool value_type_valid(mipk_value_type type)
{
    return type >= MIPK_VALUE_NONE && type <= MIPK_VALUE_OPAQUE;
}

static bool collection_item_type_valid(mipk_value_type type)
{
    return value_type_valid(type) && type != MIPK_VALUE_NONE &&
        type != MIPK_VALUE_COLLECTION;
}

mipk_value_type mipk_value_descriptor_payload_type(
    const mipk_value_descriptor *descriptor)
{
    if (descriptor == NULL || !value_type_valid(descriptor->value_type)) {
        return MIPK_VALUE_NONE;
    }
    return descriptor->value_type == MIPK_VALUE_COLLECTION ?
        descriptor->collection_item_type : descriptor->value_type;
}

bool mipk_value_descriptor_valid(const mipk_value_descriptor *descriptor)
{
    if (descriptor == NULL || !value_type_valid(descriptor->value_type)) {
        return false;
    }
    if (descriptor->value_type == MIPK_VALUE_COLLECTION) {
        if (!collection_item_type_valid(descriptor->collection_item_type) ||
            descriptor->max_items == 0) return false;
    } else if (descriptor->collection_item_type != MIPK_VALUE_NONE ||
        descriptor->max_items != 0) {
        return false;
    }

    const mipk_value_type payload =
        mipk_value_descriptor_payload_type(descriptor);
    if (payload == MIPK_VALUE_TENSORS) {
        if (descriptor->tensor_count != 0 && descriptor->tensors == NULL) {
            return false;
        }
    } else if (descriptor->tensor_count != 0 || descriptor->tensors != NULL) {
        return false;
    }
    if (payload == MIPK_VALUE_RESULTS) {
        return descriptor->results.result_type_mask != 0 &&
            descriptor->results.max_results != 0;
    }
    return descriptor->results.result_type_mask == 0 &&
        descriptor->results.max_results == 0;
}

static mipk_status validate_collection(const mipk_collection_value *collection)
{
    if (collection == NULL ||
        !collection_item_type_valid(collection->item_type) ||
        collection->count > collection->capacity ||
        (collection->capacity != 0 && collection->items == NULL)) {
        return MIPK_ERR_ARGUMENT;
    }
    for (size_t index = 0; index < collection->count; ++index) {
        const mipk_value *item = &collection->items[index];
        if (item->type != collection->item_type ||
            item->metadata.item_id == MIPK_INVALID_ITEM_ID) {
            return MIPK_ERR_ARGUMENT;
        }
        for (size_t previous = 0; previous < index; ++previous) {
            if (collection->items[previous].metadata.item_id ==
                item->metadata.item_id) return MIPK_ERR_ARGUMENT;
        }
        mipk_status status = mipk_value_validate(item);
        if (status != MIPK_OK) return status;
    }
    return MIPK_OK;
}

mipk_status mipk_value_validate(const mipk_value *value)
{
    if (value == NULL || !value_type_valid(value->type)) {
        return MIPK_ERR_ARGUMENT;
    }
    switch (value->type) {
    case MIPK_VALUE_NONE:
        return MIPK_OK;
    case MIPK_VALUE_INPUT:
        return value->value.input == NULL ? MIPK_ERR_ARGUMENT : MIPK_OK;
    case MIPK_VALUE_IMAGE_VIEW:
        if (value->value.image_view.root_input == NULL ||
            (value->value.image_view.point_count != 2 &&
             value->value.image_view.point_count != 4)) {
            return MIPK_ERR_ARGUMENT;
        }
        for (size_t index = 0;
            index < value->value.image_view.point_count * 2U; ++index) {
            if (!isfinite(value->value.image_view.region[index])) {
                return MIPK_ERR_ARGUMENT;
            }
        }
        return MIPK_OK;
    case MIPK_VALUE_TENSORS:
        return value->value.tensors.count == 0 ||
            value->value.tensors.items == NULL ?
            MIPK_ERR_ARGUMENT : MIPK_OK;
    case MIPK_VALUE_RESULTS: {
        const mipk_result_value *results = &value->value.results;
        return results->item_count > results->item_capacity ||
            (results->item_capacity != 0 && results->items == NULL) ||
            (results->score_count != 0 && results->scores == NULL) ?
            MIPK_ERR_ARGUMENT : MIPK_OK;
    }
    case MIPK_VALUE_COLLECTION:
        return validate_collection(&value->value.collection);
    case MIPK_VALUE_OPAQUE:
        return value->value.opaque.type_name == NULL ||
            value->value.opaque.type_name[0] == '\0' ||
            (value->value.opaque.size != 0 &&
                value->value.opaque.data == NULL) ?
            MIPK_ERR_ARGUMENT : MIPK_OK;
    default:
        return MIPK_ERR_ARGUMENT;
    }
}

const mipk_value *mipk_value_collection_at(const mipk_value *value,
    size_t index)
{
    if (value == NULL || value->type != MIPK_VALUE_COLLECTION ||
        index >= value->value.collection.count) return NULL;
    return &value->value.collection.items[index];
}

const mipk_value *mipk_named_value_find(const mipk_named_value *values,
    size_t count, const char *port)
{
    if (values == NULL || port == NULL) return NULL;
    for (size_t index = 0; index < count; ++index) {
        if (values[index].port != NULL &&
            strcmp(values[index].port, port) == 0) {
            return values[index].value;
        }
    }
    return NULL;
}
