#ifndef MIPK_CAPABILITY_H
#define MIPK_CAPABILITY_H

#include "mipk_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    MIPK_CAPABILITY_BACKEND = 1,
    MIPK_CAPABILITY_OPERATOR,
    MIPK_CAPABILITY_POSTPROCESS,
} mipk_capability_kind;

#define MIPK_BACKEND_ABI_VERSION UINT32_C(1)
#define MIPK_OPERATOR_ABI_VERSION UINT32_C(2)
#define MIPK_POSTPROCESS_ABI_VERSION UINT32_C(2)

typedef struct {
    uint32_t struct_size;
    mipk_capability_kind kind;
    const char *name;
    uint32_t abi_version;
    const char *implementation_version;
} mipk_capability_descriptor;

size_t mipk_capability_count(void);
const mipk_capability_descriptor *mipk_capability_at(size_t index);

#ifdef __cplusplus
}
#endif
#endif
