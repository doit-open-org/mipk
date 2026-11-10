#ifndef MIPK_WAKE_WORD_TRIGGER_H
#define MIPK_WAKE_WORD_TRIGGER_H

#include "mipk_module.h"

#ifdef __cplusplus
extern "C" {
#endif

const mipk_operator_ops *mipk_wake_word_trigger_operator(void);
#define MIPK_WAKE_WORD_UPDATE_SCORE_THRESHOLD UINT32_C(1)
#define MIPK_WAKE_WORD_UPDATE_CONFIRMATION_HITS UINT32_C(2)
#define MIPK_WAKE_WORD_UPDATE_CONFIRMATION_GAPS UINT32_C(4)
#define MIPK_WAKE_WORD_UPDATE_COOLDOWN UINT32_C(8)
#define MIPK_WAKE_WORD_UPDATE_ALL UINT32_C(15)

typedef struct {
    uint32_t struct_size;
    uint32_t field_mask;
    float score_threshold;
    uint32_t confirmation_hits;
    uint32_t confirmation_gaps;
    uint32_t cooldown_ms;
} mipk_wake_word_runtime_options;

/* Typed hot update of a wake_word_trigger node. */
mipk_status mipk_pipeline_update_wake_word(mipk_pipeline *pipeline,
    const char *node_id, const mipk_wake_word_runtime_options *options);

/* Registers the module with the runtime; called by the generated registry. */
mipk_status mipk_wake_word_trigger_register(void);

#ifdef __cplusplus
}
#endif
#endif
