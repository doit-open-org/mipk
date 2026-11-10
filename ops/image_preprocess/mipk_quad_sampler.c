#include "mipk_quad_sampler.h"

#define MIPK_QUAD_FRACTION_BITS 32
#define MIPK_QUAD_ONE (1.0 * 4294967296.0)

typedef struct {
    int64_t x;
    int64_t y;
    int64_t step_x;
    int64_t step_y;
} quad_row;

static int64_t to_fixed(double value)
{
    return (int64_t)(value * MIPK_QUAD_ONE);
}

/*
 * Builds the source position for the first sample of a row and the per-step
 * delta.  At a fixed v the quad's edges interpolate to a segment, and the
 * output walks that segment at a constant rate.
 */
static quad_row quad_row_init(const float corners[8], double v,
    double first_u, double step_u)
{
    const double complement = 1.0 - v;
    const double ax = complement * corners[0] + v * corners[6];
    const double ay = complement * corners[1] + v * corners[7];
    const double bx = complement * corners[2] + v * corners[4];
    const double by = complement * corners[3] + v * corners[5];
    quad_row row;
    row.x = to_fixed(ax + (bx - ax) * first_u);
    row.y = to_fixed(ay + (by - ay) * first_u);
    row.step_x = to_fixed((bx - ax) * step_u);
    row.step_y = to_fixed((by - ay) * step_u);
    return row;
}

/* Arithmetic shift floors towards minus infinity, matching floorf. */
static int32_t quad_floor(int64_t value)
{
    return (int32_t)(value >> MIPK_QUAD_FRACTION_BITS);
}

static int32_t clamp_i32(int32_t value, int32_t maximum)
{
    if (value < 0) return 0;
    return value > maximum ? maximum : value;
}

void mipk_quad_sample_luma(const mipk_nv12_source *source,
    const float corners[8], uint8_t *destination, uint32_t destination_stride,
    uint32_t width, uint32_t height)
{
    if (source == NULL || corners == NULL || destination == NULL ||
        width == 0 || height == 0 || source->width == 0 ||
        source->height == 0) {
        return;
    }
    const int32_t last_x = (int32_t)source->width - 1;
    const int32_t last_y = (int32_t)source->height - 1;
    const double inverse_width = 1.0 / (double)width;
    const double inverse_height = 1.0 / (double)height;
    for (uint32_t row = 0; row < height; ++row) {
        quad_row walk = quad_row_init(corners,
            ((double)row + 0.5) * inverse_height,
            0.5 * inverse_width, inverse_width);
        uint8_t *target = destination + (size_t)row * destination_stride;
        for (uint32_t column = 0; column < width; ++column) {
            const int32_t x = clamp_i32(quad_floor(walk.x), last_x);
            const int32_t y = clamp_i32(quad_floor(walk.y), last_y);
            target[column] =
                source->luma[(size_t)y * source->luma_stride + (uint32_t)x];
            walk.x += walk.step_x;
            walk.y += walk.step_y;
        }
    }
}

void mipk_quad_sample_chroma(const mipk_nv12_source *source,
    const float corners[8], uint8_t *destination, uint32_t destination_stride,
    uint32_t width, uint32_t height)
{
    if (source == NULL || corners == NULL || destination == NULL ||
        width == 0 || height == 0 || source->width < 2 ||
        source->height < 2) {
        return;
    }
    /* The luma grid is twice the chroma grid in each direction. */
    const int32_t last_x = (int32_t)source->width - 2;
    const int32_t last_y = (int32_t)source->height / 2 - 1;
    const double inverse_width = 1.0 / (double)(width * 2U);
    const double inverse_height = 1.0 / (double)(height * 2U);
    for (uint32_t row = 0; row < height; ++row) {
        quad_row walk = quad_row_init(corners,
            ((double)(row * 2U) + 1.0) * inverse_height,
            inverse_width, 2.0 * inverse_width);
        uint8_t *target = destination + (size_t)row * destination_stride;
        for (uint32_t pair = 0; pair < width; ++pair) {
            int32_t x = clamp_i32(quad_floor(walk.x), last_x);
            x &= ~INT32_C(1);
            /* floor(y / 2) for a non-negative y is the floor of y shifted. */
            const int32_t y =
                clamp_i32(quad_floor(walk.y) < 0 ? 0 : quad_floor(walk.y) / 2,
                    last_y);
            const uint8_t *sample = source->chroma +
                (size_t)y * source->chroma_stride + (uint32_t)x;
            target[pair * 2U] = sample[0];
            target[pair * 2U + 1U] = sample[1];
            walk.x += walk.step_x;
            walk.y += walk.step_y;
        }
    }
}
