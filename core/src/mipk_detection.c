#include "mipk_detection.h"

#include <math.h>
#include <stdlib.h>

static float intersection_over_union(const mipk_bbox *left,
    const mipk_bbox *right)
{
    const float x1 = fmaxf(left->x1, right->x1);
    const float y1 = fmaxf(left->y1, right->y1);
    const float x2 = fminf(left->x2, right->x2);
    const float y2 = fminf(left->y2, right->y2);
    const float intersection = fmaxf(0.0F, x2 - x1) *
        fmaxf(0.0F, y2 - y1);
    const float left_area = fmaxf(0.0F, left->x2 - left->x1) *
        fmaxf(0.0F, left->y2 - left->y1);
    const float right_area = fmaxf(0.0F, right->x2 - right->x1) *
        fmaxf(0.0F, right->y2 - right->y1);
    const float union_area = left_area + right_area - intersection;
    return union_area > 0.0F ? intersection / union_area : 0.0F;
}

static int compare_score(const void *left, const void *right)
{
    const mipk_result *a = left;
    const mipk_result *b = right;
    return a->score < b->score ? 1 : (a->score > b->score ? -1 : 0);
}

void mipk_detection_retain(mipk_result *items, size_t capacity,
    size_t *count, const mipk_result *candidate)
{
    if (items == NULL || count == NULL || candidate == NULL || capacity == 0) return;
    if (*count < capacity) {
        items[(*count)++] = *candidate;
        return;
    }
    size_t lowest = 0;
    for (size_t i = 1; i < capacity; ++i) {
        if (items[i].score < items[lowest].score) lowest = i;
    }
    if (candidate->score > items[lowest].score) items[lowest] = *candidate;
}

mipk_status mipk_detection_nms(mipk_result *candidates,
    size_t candidate_count, bool apply_nms, bool class_agnostic,
    float nms_threshold, mipk_result_buffer *results)
{
    if ((candidate_count != 0 && candidates == NULL) || results == NULL ||
        (results->capacity != 0 && results->items == NULL) ||
        nms_threshold < 0.0F || nms_threshold > 1.0F) return MIPK_ERR_ARGUMENT;
    results->count = 0;
    qsort(candidates, candidate_count, sizeof(*candidates), compare_score);
    for (size_t i = 0; i < candidate_count; ++i) {
        bool suppressed = false;
        if (apply_nms) {
            for (size_t selected = 0; selected < results->count; ++selected) {
                if ((class_agnostic || candidates[i].class_id ==
                        results->items[selected].class_id) &&
                    intersection_over_union(&candidates[i].value.bbox,
                        &results->items[selected].value.bbox) > nms_threshold) {
                    suppressed = true;
                    break;
                }
            }
        }
        if (suppressed) continue;
        if (results->count == results->capacity) return MIPK_ERR_CAPACITY;
        results->items[results->count++] = candidates[i];
    }
    return MIPK_OK;
}
