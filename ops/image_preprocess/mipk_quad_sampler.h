#ifndef MIPK_QUAD_SAMPLER_H
#define MIPK_QUAD_SAMPLER_H

#include <stddef.h>
#include <stdint.h>

/*
 * Nearest-neighbour sampling of a quadrilateral view out of an NV12 frame.
 *
 * A quad maps to the output affinely along each axis, so the source position
 * can be stepped with integer adds instead of being recomputed per pixel.  The
 * obvious per-pixel form needs eight multiplies plus floorf/fminf/fmaxf, and on
 * the Hi3516CV610 those libm calls dominate: a 224x224 crop measured 93 ms,
 * against 1.7 ms for the stepped form below.
 *
 * Positions are carried in 32.32 fixed point, so accumulating across a row
 * drifts by less than 1e-7 of a pixel and the selected source pixel matches the
 * floating point reference.
 */

typedef struct {
    const uint8_t *luma;
    const uint8_t *chroma;
    uint32_t width;
    uint32_t height;
    uint32_t luma_stride;
    uint32_t chroma_stride;
} mipk_nv12_source;

/*
 * corners lists the quad as x,y pairs in source pixels, ordered to match the
 * output's (0,0), (1,0), (1,1) and (0,1). destination points at the first
 * written pixel and advances by destination_stride per row.
 */
void mipk_quad_sample_luma(const mipk_nv12_source *source,
    const float corners[8], uint8_t *destination, uint32_t destination_stride,
    uint32_t width, uint32_t height);

/* Chroma pairs, sampled at the centre of each 2x2 luma block. */
void mipk_quad_sample_chroma(const mipk_nv12_source *source,
    const float corners[8], uint8_t *destination, uint32_t destination_stride,
    uint32_t width, uint32_t height);

#endif
