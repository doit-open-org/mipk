#ifndef MIPK_NMS_H
#define MIPK_NMS_H

#include "mipk_module.h"

#ifdef __cplusplus
extern "C" {
#endif

const mipk_operator_ops *mipk_nms_operator(void);
#define MIPK_NMS_UPDATE_IOU_THRESHOLD UINT32_C(1)
#define MIPK_NMS_UPDATE_CLASS_AGNOSTIC UINT32_C(2)
#define MIPK_NMS_UPDATE_ALL (MIPK_NMS_UPDATE_IOU_THRESHOLD | \
    MIPK_NMS_UPDATE_CLASS_AGNOSTIC)

typedef struct {
    uint32_t struct_size;
    uint32_t field_mask;
    float iou_threshold;
    bool class_agnostic;
} mipk_nms_runtime_options;

/* Typed hot update of an nms node; the node must be an nms operator. */
mipk_status mipk_pipeline_update_nms(mipk_pipeline *pipeline,
    const char *node_id, const mipk_nms_runtime_options *options);

/* Registers the module with the runtime; called by the generated registry. */
mipk_status mipk_nms_register(void);

#ifdef __cplusplus
}
#endif
#endif
