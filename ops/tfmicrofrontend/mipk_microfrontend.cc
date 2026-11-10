#include "mipk_microfrontend.h"

#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>

extern "C" {
#include "tensorflow/lite/experimental/microfrontend/lib/frontend.h"
#include "tensorflow/lite/experimental/microfrontend/lib/frontend_util.h"
}

struct mipk_microfrontend {
    FrontendState state;
    mipk_microfrontend_config config;
};

static bool config_valid(const mipk_microfrontend_config *config)
{
    return config != nullptr && config->sample_rate > 0 &&
        config->sample_rate <= INT_MAX && config->audio_frame_samples > 0 &&
        config->feature_count > 0 && config->feature_count <= INT_MAX &&
        config->window_size_ms > 0 && config->window_step_ms > 0 &&
        config->lower_band_limit >= 0.0F &&
        config->upper_band_limit > config->lower_band_limit &&
        config->upper_band_limit <= (float)config->sample_rate / 2.0F &&
        std::isfinite(config->output_scale) && config->output_scale > 0.0F;
}

static void configure(const mipk_microfrontend_config *source,
    FrontendConfig *destination)
{
    std::memset(destination, 0, sizeof(*destination));
    destination->window.size_ms = source->window_size_ms;
    destination->window.step_size_ms = source->window_step_ms;
    destination->filterbank.num_channels =
        static_cast<int>(source->feature_count);
    destination->filterbank.lower_band_limit = source->lower_band_limit;
    destination->filterbank.upper_band_limit = source->upper_band_limit;
    destination->noise_reduction.smoothing_bits =
        source->noise_smoothing_bits;
    destination->noise_reduction.even_smoothing =
        source->noise_even_smoothing;
    destination->noise_reduction.odd_smoothing =
        source->noise_odd_smoothing;
    destination->noise_reduction.min_signal_remaining =
        source->noise_min_signal_remaining;
    destination->pcan_gain_control.enable_pcan = source->enable_pcan ? 1 : 0;
    destination->pcan_gain_control.strength = source->pcan_strength;
    destination->pcan_gain_control.offset = source->pcan_offset;
    destination->pcan_gain_control.gain_bits = source->pcan_gain_bits;
    destination->log_scale.enable_log = source->enable_log ? 1 : 0;
    destination->log_scale.scale_shift = source->log_scale_shift;
}

extern "C" mipk_microfrontend *mipk_microfrontend_create(
    const mipk_microfrontend_config *config)
{
    if (!config_valid(config)) {
        return nullptr;
    }
    mipk_microfrontend *frontend = static_cast<mipk_microfrontend *>(
        std::calloc(1, sizeof(*frontend)));
    if (frontend == nullptr) {
        return nullptr;
    }
    frontend->config = *config;
    FrontendConfig tensorflow_config;
    configure(config, &tensorflow_config);
    if (!FrontendPopulateState(&tensorflow_config, &frontend->state,
            static_cast<int>(config->sample_rate))) {
        FrontendFreeStateContents(&frontend->state);
        std::free(frontend);
        return nullptr;
    }
    return frontend;
}

extern "C" void mipk_microfrontend_destroy(mipk_microfrontend *frontend)
{
    if (frontend == nullptr) {
        return;
    }
    FrontendFreeStateContents(&frontend->state);
    std::free(frontend);
}

extern "C" void mipk_microfrontend_reset(mipk_microfrontend *frontend)
{
    if (frontend != nullptr) {
        FrontendReset(&frontend->state);
    }
}

static size_t snapshot_size(const mipk_microfrontend *frontend)
{
    if (frontend == nullptr) return 0;
    const FrontendState &state = frontend->state;
    return sizeof(state.window.input_used) +
        state.window.size * sizeof(*state.window.input) +
        state.window.size * sizeof(*state.window.output) +
        state.fft.input_size * sizeof(*state.fft.input) +
        (state.fft.fft_size / 2 + 1) * 2 * sizeof(*state.fft.output) +
        state.fft.scratch_size +
        static_cast<size_t>(state.filterbank.num_channels + 1) *
            sizeof(*state.filterbank.work) +
        static_cast<size_t>(state.noise_reduction.num_channels) *
            sizeof(*state.noise_reduction.estimate);
}

extern "C" size_t mipk_microfrontend_checkpoint_size(
    const mipk_microfrontend *frontend)
{
    return snapshot_size(frontend);
}

static bool transfer_snapshot(const mipk_microfrontend *frontend,
    uint8_t *buffer, size_t size, bool restore)
{
    if (frontend == nullptr || buffer == nullptr ||
        size != snapshot_size(frontend)) return false;
    FrontendState &state = const_cast<FrontendState &>(frontend->state);
    size_t offset = 0;
    auto transfer = [&](void *data, size_t bytes) {
        if (bytes > size - offset) return false;
        if (restore) std::memcpy(data, buffer + offset, bytes);
        else std::memcpy(buffer + offset, data, bytes);
        offset += bytes;
        return true;
    };
    return transfer(&state.window.input_used, sizeof(state.window.input_used)) &&
        transfer(state.window.input,
            state.window.size * sizeof(*state.window.input)) &&
        transfer(state.window.output,
            state.window.size * sizeof(*state.window.output)) &&
        transfer(state.fft.input,
            state.fft.input_size * sizeof(*state.fft.input)) &&
        transfer(state.fft.output, (state.fft.fft_size / 2 + 1) * 2 *
            sizeof(*state.fft.output)) &&
        transfer(state.fft.scratch, state.fft.scratch_size) &&
        transfer(state.filterbank.work,
            static_cast<size_t>(state.filterbank.num_channels + 1) *
                sizeof(*state.filterbank.work)) &&
        transfer(state.noise_reduction.estimate,
            static_cast<size_t>(state.noise_reduction.num_channels) *
                sizeof(*state.noise_reduction.estimate)) && offset == size;
}

extern "C" bool mipk_microfrontend_checkpoint(
    const mipk_microfrontend *frontend, void *buffer, size_t size)
{
    return transfer_snapshot(frontend, static_cast<uint8_t *>(buffer), size,
        false);
}

extern "C" bool mipk_microfrontend_restore(mipk_microfrontend *frontend,
    const void *buffer, size_t size)
{
    return transfer_snapshot(frontend,
        const_cast<uint8_t *>(static_cast<const uint8_t *>(buffer)), size,
        true);
}

extern "C" int mipk_microfrontend_process(mipk_microfrontend *frontend,
    const int16_t *samples, size_t sample_count, float *features,
    size_t feature_count)
{
    if (frontend == nullptr || samples == nullptr || features == nullptr ||
        sample_count != frontend->config.audio_frame_samples ||
        feature_count != frontend->config.feature_count) {
        return -1;
    }

    size_t samples_read = 0;
    const FrontendOutput output = FrontendProcessSamples(&frontend->state,
        samples, sample_count, &samples_read);
    if (samples_read != sample_count ||
        (output.size != 0 && output.size != feature_count)) {
        return -1;
    }
    if (output.size == 0) {
        return 0;
    }
    for (size_t index = 0; index < feature_count; ++index) {
        features[index] = static_cast<float>(output.values[index]) *
            frontend->config.output_scale;
    }
    return 1;
}
