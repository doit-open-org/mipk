#ifndef MIPK_RESULT_FILTER_H
#define MIPK_RESULT_FILTER_H

#include "mipk_module.h"

#ifdef __cplusplus
extern "C" {
#endif

const mipk_operator_ops *mipk_result_filter_operator(void);
/* Registers the module with the runtime; called by the generated registry. */
mipk_status mipk_result_filter_register(void);

#ifdef __cplusplus
}
#endif
#endif
