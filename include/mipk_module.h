#ifndef MIPK_MODULE_H
#define MIPK_MODULE_H

/*
 * Everything an operator, postprocess or backend module needs to
 * describe itself to the runtime.  Modules include only this header and the
 * public headers it pulls in; nothing from core/ is visible to them.
 */

#include "mipk.h"
#include "mipk_backend.h"
#include "mipk_capability.h"
#include "mipk_detection.h"
#include "mipk_memory.h"
#include "mipk_operator.h"
#include "mipk_postprocess.h"
#include "mipk_task.h"
#include "mipk_tensor.h"
#include "mipk_value.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MIPK_VALUE_MASK(type) (UINT32_C(1) << (type))
#define MIPK_DOMAIN_MASK(domain) (UINT32_C(1) << (domain))
#define MIPK_RESULT_MASK(type) (UINT64_C(1) << (type))

#define MIPK_CAPABILITY_DESCRIPTOR(capability_kind, capability_name, \
    capability_abi, capability_version) \
    { \
        .struct_size = sizeof(mipk_capability_descriptor), \
        .kind = (capability_kind), \
        .name = (capability_name), \
        .abi_version = (capability_abi), \
        .implementation_version = (capability_version), \
    }

#define MIPK_OPERATOR_CAPABILITY(capability_name, capability_version) \
    MIPK_CAPABILITY_DESCRIPTOR(MIPK_CAPABILITY_OPERATOR, capability_name, \
        MIPK_OPERATOR_ABI_VERSION, capability_version)
#define MIPK_POSTPROCESS_CAPABILITY(capability_name, capability_version) \
    MIPK_CAPABILITY_DESCRIPTOR(MIPK_CAPABILITY_POSTPROCESS, capability_name, \
        MIPK_POSTPROCESS_ABI_VERSION, capability_version)
#define MIPK_BACKEND_CAPABILITY(capability_name, capability_version) \
    MIPK_CAPABILITY_DESCRIPTOR(MIPK_CAPABILITY_BACKEND, capability_name, \
        MIPK_BACKEND_ABI_VERSION, capability_version)

/* An input port that accepts the tensors of a model stage. */
#define MIPK_TENSOR_INPUT_PORT(port_name) \
    { \
        .port = (port_name), \
        .accepted_value_mask = MIPK_VALUE_MASK(MIPK_VALUE_TENSORS), \
        .accepted_domain_mask = MIPK_DOMAIN_MASK(MIPK_DOMAIN_IMAGE) | \
            MIPK_DOMAIN_MASK(MIPK_DOMAIN_AUDIO), \
        .required = true, \
    }

/*
 * Registers every module compiled into the mipk library.  The build generates
 * the definition from the enabled module list and compiles it into core; a
 * library built without any module registers nothing.  mipk_pipeline_open()
 * calls this once through mipk_init(), so applications only need it when
 * they register their own modules first and want the built-in set to take
 * precedence.
 */
mipk_status mipk_register_builtin_modules(void);

/* Idempotent. Registers the built-in modules once. */
mipk_status mipk_init(void);

#ifdef __cplusplus
}
#endif
#endif
