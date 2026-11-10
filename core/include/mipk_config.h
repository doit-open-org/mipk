#ifndef MIPK_CONFIG_H
#define MIPK_CONFIG_H

/*
 * Resource ceilings the runtime enforces when a pipeline opens.  The defaults
 * suit a desktop host; a constrained device profile overrides them from the
 * build (for example -DMIPK_MAX_MODEL_BYTES=12582912 for a 32 MiB SoC).
 */

#ifndef MIPK_CONTAINER_MANIFEST_MAX_SIZE
#define MIPK_CONTAINER_MANIFEST_MAX_SIZE (1024U * 1024U)
#endif

#ifndef MIPK_MAX_MODEL_BYTES
#define MIPK_MAX_MODEL_BYTES (1024U * 1024U * 1024U)
#endif

#ifndef MIPK_MAX_OUTPUT_BYTES
#define MIPK_MAX_OUTPUT_BYTES (16U * 1024U * 1024U)
#endif

#ifndef MIPK_MAX_SCRATCH_BYTES
#define MIPK_MAX_SCRATCH_BYTES (64U * 1024U * 1024U)
#endif

#ifndef MIPK_MAX_MMZ_BYTES
#define MIPK_MAX_MMZ_BYTES (256U * 1024U * 1024U)
#endif

#endif
