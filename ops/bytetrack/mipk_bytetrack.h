#ifndef MIPK_BYTETRACK_H
#define MIPK_BYTETRACK_H

#include "mipk_module.h"

#ifdef __cplusplus
extern "C" {
#endif

const mipk_operator_ops *mipk_bytetrack_operator(void);
#define MIPK_BYTETRACK_UPDATE_HIGH_THRESHOLD UINT32_C(1)
#define MIPK_BYTETRACK_UPDATE_LOW_THRESHOLD UINT32_C(2)
#define MIPK_BYTETRACK_UPDATE_NEW_THRESHOLD UINT32_C(4)
#define MIPK_BYTETRACK_UPDATE_MIN_MATCH_IOU UINT32_C(8)
#define MIPK_BYTETRACK_UPDATE_LOST_TIMEOUT UINT32_C(16)
#define MIPK_BYTETRACK_UPDATE_MAX_PREDICTION_DT UINT32_C(32)
#define MIPK_BYTETRACK_UPDATE_RESET_GAP UINT32_C(64)
#define MIPK_BYTETRACK_UPDATE_CLASS_AWARE UINT32_C(128)
#define MIPK_BYTETRACK_UPDATE_EMIT_LOST UINT32_C(256)
#define MIPK_BYTETRACK_UPDATE_EMIT_REMOVED UINT32_C(512)
#define MIPK_BYTETRACK_UPDATE_ALL UINT32_C(1023)

typedef struct {
    uint32_t struct_size;
    uint32_t field_mask;
    float track_high_threshold;
    float track_low_threshold;
    float new_track_threshold;
    float min_match_iou;
    uint32_t lost_timeout_ms;
    uint32_t max_prediction_dt_ms;
    uint32_t reset_gap_ms;
    bool class_aware;
    bool emit_lost;
    bool emit_removed;
} mipk_bytetrack_runtime_options;

/* Typed hot update of a bytetrack node; tracker state survives the change. */
mipk_status mipk_pipeline_update_bytetrack(mipk_pipeline *pipeline,
    const char *node_id, const mipk_bytetrack_runtime_options *options);

/* Registers the module with the runtime; called by the generated registry. */
mipk_status mipk_bytetrack_register(void);

#ifdef __cplusplus
}
#endif
#endif
