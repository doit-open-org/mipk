#ifndef MIPK_YOLO_POSE_H
#define MIPK_YOLO_POSE_H

#include "mipk_module.h"

#ifdef __cplusplus
extern "C" {
#endif

const mipk_postprocess_ops *mipk_yolo_pose_postprocess(void);
/* Registers the module with the runtime; called by the generated registry. */
mipk_status mipk_yolo_pose_register(void);

#ifdef __cplusplus
}
#endif
#endif
