#ifndef MIPK_DETECTION_H
#define MIPK_DETECTION_H

#include "mipk_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Keep the highest-scoring candidates in caller-owned storage. */
void mipk_detection_retain(mipk_result *items, size_t capacity,
    size_t *count, const mipk_result *candidate);

/* Sort candidates in place and copy the NMS result into caller-owned results. */
mipk_status mipk_detection_nms(mipk_result *candidates,
    size_t candidate_count, bool apply_nms, bool class_agnostic,
    float nms_threshold, mipk_result_buffer *results);

#ifdef __cplusplus
}
#endif
#endif
