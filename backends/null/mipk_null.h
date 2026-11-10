#ifndef MIPK_NULL_H
#define MIPK_NULL_H

#include "mipk_module.h"

#ifdef __cplusplus
extern "C" {
#endif

const mipk_backend_ops *mipk_null_backend(void);
/* Registers the module with the runtime; called by the generated registry. */
mipk_status mipk_null_backend_register(void);

#ifdef __cplusplus
}
#endif
#endif
