#ifndef MIPK_TFMICROFRONTEND_H
#define MIPK_TFMICROFRONTEND_H

#include "mipk_module.h"

#ifdef __cplusplus
extern "C" {
#endif

const mipk_operator_ops *mipk_tfmicrofrontend_operator(void);
/* Registers the module with the runtime; called by the generated registry. */
mipk_status mipk_tfmicrofrontend_register(void);

#ifdef __cplusplus
}
#endif
#endif
