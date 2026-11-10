#ifndef MIPK_YOLO_H
#define MIPK_YOLO_H

#include "mipk_module.h"

#ifdef __cplusplus
extern "C" {
#endif

const mipk_postprocess_ops *mipk_yolo_postprocess(void);
/* Registers the module with the runtime; called by the generated registry. */
mipk_status mipk_yolo_register(void);

#ifdef __cplusplus
}
#endif
#endif
