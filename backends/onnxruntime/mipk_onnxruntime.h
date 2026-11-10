#ifndef MIPK_ONNXRUNTIME_H
#define MIPK_ONNXRUNTIME_H

#include "mipk_module.h"

#ifdef __cplusplus
extern "C" {
#endif

const mipk_backend_ops *mipk_onnxruntime_backend(void);
/* Registers the module with the runtime; called by the generated registry. */
mipk_status mipk_onnxruntime_register(void);

#ifdef __cplusplus
}
#endif
#endif
