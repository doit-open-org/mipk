#ifndef MIPK_MICROFRONTEND_INTERNAL_H
#define MIPK_MICROFRONTEND_INTERNAL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t sample_rate;
    size_t audio_frame_samples;
    size_t feature_count;
    int window_size_ms;
    int window_step_ms;
    float lower_band_limit;
    float upper_band_limit;
    int noise_smoothing_bits;
    float noise_even_smoothing;
    float noise_odd_smoothing;
    float noise_min_signal_remaining;
    bool enable_pcan;
    float pcan_strength;
    float pcan_offset;
    int pcan_gain_bits;
    bool enable_log;
    int log_scale_shift;
    float output_scale;
} mipk_microfrontend_config;

typedef struct mipk_microfrontend mipk_microfrontend;

mipk_microfrontend *mipk_microfrontend_create(
    const mipk_microfrontend_config *config);
void mipk_microfrontend_destroy(mipk_microfrontend *frontend);
void mipk_microfrontend_reset(mipk_microfrontend *frontend);
size_t mipk_microfrontend_checkpoint_size(const mipk_microfrontend *frontend);
bool mipk_microfrontend_checkpoint(const mipk_microfrontend *frontend,
    void *buffer, size_t size);
bool mipk_microfrontend_restore(mipk_microfrontend *frontend,
    const void *buffer, size_t size);
/* Returns 1 for one feature frame, 0 during warm-up, or -1 on error. */
int mipk_microfrontend_process(mipk_microfrontend *frontend,
    const int16_t *samples, size_t sample_count,
    float *features, size_t feature_count);

#ifdef __cplusplus
}
#endif

#endif
