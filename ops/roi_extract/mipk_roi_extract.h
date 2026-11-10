#ifndef MIPK_ROI_EXTRACT_H
#define MIPK_ROI_EXTRACT_H

#include "mipk_module.h"

#ifdef __cplusplus
extern "C" {
#endif

const mipk_operator_ops *mipk_roi_extract_operator(void);
/* Registers the module with the runtime; called by the generated registry. */
mipk_status mipk_roi_extract_register(void);

#ifdef __cplusplus
}
#endif
#endif
