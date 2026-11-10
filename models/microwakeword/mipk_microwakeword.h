#ifndef MIPK_MICROWAKEWORD_H
#define MIPK_MICROWAKEWORD_H

#include "mipk_module.h"

#ifdef __cplusplus
extern "C" {
#endif

const mipk_postprocess_ops *mipk_microwakeword_postprocess(void);
/* Registers the module with the runtime; called by the generated registry. */
mipk_status mipk_microwakeword_register(void);

#ifdef __cplusplus
}
#endif
#endif
