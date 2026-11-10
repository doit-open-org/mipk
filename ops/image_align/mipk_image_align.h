#ifndef MIPK_IMAGE_ALIGN_H
#define MIPK_IMAGE_ALIGN_H

#include "mipk_module.h"

#ifdef __cplusplus
extern "C" {
#endif

const mipk_operator_ops *mipk_image_align_operator(void);
/* Registers the module with the runtime; called by the generated registry. */
mipk_status mipk_image_align_register(void);

#ifdef __cplusplus
}
#endif
#endif
