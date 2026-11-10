#include "mipk_pipeline_internal.h"

#include <math.h>

/*
 * How an image_preprocess stage fits a frame into the model's input.
 *
 * Two places need this and must agree: the manifest, which records the
 * geometry and derives from it the transform that maps results back to the
 * frame, and the builder, which hands each stage the geometry it will resize
 * with.  When they disagreed, results came back stretched about the centre of
 * the frame -- correct in the middle, further out the nearer the edge.
 */

bool mipk_preprocess_letterbox(uint32_t source_width, uint32_t source_height,
    uint32_t target_width, uint32_t target_height, bool yuv,
    uint32_t *resized_width, uint32_t *resized_height, uint32_t *pad_x,
    uint32_t *pad_y)
{
    if (source_width == 0 || source_height == 0 || target_width == 0 ||
        target_height == 0 || resized_width == NULL ||
        resized_height == NULL || pad_x == NULL || pad_y == NULL) {
        return false;
    }
    /* Whichever axis runs out first decides the scale; the other is padded. */
    const double scale = fmin((double)target_width / source_width,
        (double)target_height / source_height);
    const uint32_t width = (uint32_t)floor(source_width * scale) &
        ~UINT32_C(1);
    const uint32_t height = (uint32_t)floor(source_height * scale) &
        ~UINT32_C(1);
    if (width == 0 || height == 0 || width > target_width ||
        height > target_height) {
        return false;
    }
    uint32_t x = (target_width - width) / 2U;
    uint32_t y = (target_height - height) / 2U;
    if (yuv) {
        /* Chroma is shared between pixel pairs, so padding starts even. */
        x &= ~UINT32_C(1);
        y &= ~UINT32_C(1);
    }
    *resized_width = width;
    *resized_height = height;
    *pad_x = x;
    *pad_y = y;
    return true;
}

bool mipk_preprocess_keeps_manifest_geometry(mipk_execution_mode execution,
    size_t target_width, size_t target_height, uint32_t manifest_width,
    uint32_t manifest_height)
{
    return execution == MIPK_EXECUTION_ONCE &&
        target_width == manifest_width && target_height == manifest_height;
}
