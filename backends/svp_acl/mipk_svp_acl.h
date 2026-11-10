#ifndef MIPK_SVP_ACL_H
#define MIPK_SVP_ACL_H

#include "mipk_module.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * HiSilicon SVP ACL backend (Hi3516CV610 and relatives): runs .om models on
 * the NPU. models[] options: "runtime_config" names an ACL config file.
 */
const mipk_backend_ops *mipk_svp_acl_backend(void);
/* Registers the module with the runtime; called by the generated registry. */
mipk_status mipk_svp_acl_register(void);

#ifdef __cplusplus
}
#endif
#endif
