/*
 * mipk_run: open a package (a .mipk file or a recipe directory), run one
 * image through it, print the results and optionally draw them.
 *
 *   mipk_run <package> <image> [--out annotated.png] [--loops N]
 *
 * The image is letterboxed into the pipeline's input size so nothing is
 * distorted, and results are mapped back to the image's own pixels.
 */
#define _POSIX_C_SOURCE 200809L

#include "mipk.h"
#include "mipk_platform_posix.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    uint8_t *pixels;
    int width;
    int height;
} image;

typedef struct {
    float scale;
    int pad_x;
    int pad_y;
} letterbox;

static double now_ms(void)
{
    struct timespec time;
#ifdef _WIN32
    (void)timespec_get(&time, TIME_UTC);
#else
    clock_gettime(CLOCK_MONOTONIC, &time);
#endif
    return (double)time.tv_sec * 1000.0 + (double)time.tv_nsec / 1.0e6;
}

/* Fits the image into width x height, grey around it, and reports how. */
static uint8_t *letterbox_rgb(const image *source, int width, int height,
    letterbox *geometry)
{
    uint8_t *canvas = malloc((size_t)width * height * 3U);
    if (canvas == NULL) return NULL;
    memset(canvas, 114, (size_t)width * height * 3U);
    const float scale = fminf((float)width / source->width,
        (float)height / source->height);
    const int resized_width = (int)(source->width * scale);
    const int resized_height = (int)(source->height * scale);
    geometry->scale = scale;
    geometry->pad_x = (width - resized_width) / 2;
    geometry->pad_y = (height - resized_height) / 2;
    for (int row = 0; row < resized_height; ++row) {
        const int source_row = (int)(row / scale);
        for (int column = 0; column < resized_width; ++column) {
            const int source_column = (int)(column / scale);
            const uint8_t *from = source->pixels +
                ((size_t)source_row * source->width + source_column) * 3U;
            uint8_t *to = canvas + ((size_t)(row + geometry->pad_y) * width +
                column + geometry->pad_x) * 3U;
            to[0] = from[0];
            to[1] = from[1];
            to[2] = from[2];
        }
    }
    return canvas;
}

static uint8_t *rgb_to_nv12(const uint8_t *rgb, int width, int height)
{
    const size_t pixels = (size_t)width * height;
    uint8_t *nv12 = malloc(pixels + pixels / 2U);
    if (nv12 == NULL) return NULL;
    uint8_t *chroma = nv12 + pixels;
    for (int row = 0; row < height; ++row) {
        for (int column = 0; column < width; ++column) {
            const uint8_t *p = rgb + ((size_t)row * width + column) * 3U;
            const float r = p[0], g = p[1], b = p[2];
            nv12[(size_t)row * width + column] = (uint8_t)fminf(255.0F,
                fmaxf(0.0F, 16.0F + 0.257F * r + 0.504F * g + 0.098F * b));
            if ((row & 1) == 0 && (column & 1) == 0) {
                uint8_t *pair = chroma + (size_t)(row / 2) * width + column;
                pair[0] = (uint8_t)fminf(255.0F, fmaxf(0.0F,
                    128.0F - 0.148F * r - 0.291F * g + 0.439F * b));
                pair[1] = (uint8_t)fminf(255.0F, fmaxf(0.0F,
                    128.0F + 0.439F * r - 0.368F * g - 0.071F * b));
            }
        }
    }
    return nv12;
}

static void draw_rectangle(image *target, int x1, int y1, int x2, int y2,
    const uint8_t color[3])
{
    for (int x = x1; x <= x2; ++x) {
        for (int y = y1; y <= y2; ++y) {
            const bool edge = x - x1 < 2 || x2 - x < 2 || y - y1 < 2 ||
                y2 - y < 2;
            if (!edge || x < 0 || y < 0 || x >= target->width ||
                y >= target->height) continue;
            uint8_t *p = target->pixels + ((size_t)y * target->width + x) * 3U;
            p[0] = color[0];
            p[1] = color[1];
            p[2] = color[2];
        }
    }
}

static void draw_point(image *target, int x, int y, const uint8_t color[3])
{
    draw_rectangle(target, x - 2, y - 2, x + 2, y + 2, color);
}

static const char *result_type_name(mipk_result_type type)
{
    switch (type) {
        case MIPK_RESULT_DETECTION: return "detection";
        case MIPK_RESULT_CLASSIFICATION: return "classification";
        case MIPK_RESULT_KEYPOINT: return "keypoint";
        case MIPK_RESULT_ATTRIBUTE: return "attribute";
        case MIPK_RESULT_TRACK: return "track";
        case MIPK_RESULT_TEXT: return "text";
        default: return "unknown";
    }
}

static void report(const mipk_pipeline *pipeline, const mipk_output *output,
    const letterbox *geometry, image *annotated)
{
    static const uint8_t box_color[3] = {255, 64, 32};
    static const uint8_t point_color[3] = {32, 220, 64};
    const size_t count = mipk_output_result_count(output);
    printf("results: %zu\n", count);
    for (size_t index = 0; index < count; ++index) {
        const mipk_result *result = mipk_output_result_at(output, index);
        const char *label = mipk_pipeline_label_name(pipeline,
            result->label_set_id, result->class_id);
        if (result->type == MIPK_RESULT_DETECTION ||
            result->type == MIPK_RESULT_TRACK) {
            const mipk_bbox box = result->type == MIPK_RESULT_DETECTION ?
                result->value.bbox : result->value.track.bbox;
            const float x1 = (box.x1 - geometry->pad_x) / geometry->scale;
            const float y1 = (box.y1 - geometry->pad_y) / geometry->scale;
            const float x2 = (box.x2 - geometry->pad_x) / geometry->scale;
            const float y2 = (box.y2 - geometry->pad_y) / geometry->scale;
            printf("  [%zu] %s %s (%d) score=%.3f box=[%.1f, %.1f, %.1f, %.1f]",
                index, result_type_name(result->type),
                label != NULL ? label : "?", result->class_id, result->score,
                x1, y1, x2, y2);
            if (result->type == MIPK_RESULT_TRACK) {
                printf(" track=%lld", (long long)result->track_id);
            }
            printf("\n");
            if (annotated != NULL) {
                draw_rectangle(annotated, (int)x1, (int)y1, (int)x2, (int)y2,
                    box_color);
            }
        } else if (result->type == MIPK_RESULT_KEYPOINT) {
            const float x = (result->value.keypoint.x - geometry->pad_x) /
                geometry->scale;
            const float y = (result->value.keypoint.y - geometry->pad_y) /
                geometry->scale;
            printf("  [%zu] keypoint id=%d parent=%llu score=%.3f at (%.1f, %.1f)\n",
                index, result->value.keypoint.keypoint_id,
                (unsigned long long)result->parent_id, result->score, x, y);
            if (annotated != NULL) {
                draw_point(annotated, (int)x, (int)y, point_color);
            }
        } else {
            printf("  [%zu] %s %s (%d) score=%.3f\n", index,
                result_type_name(result->type), label != NULL ? label : "?",
                result->class_id, result->score);
        }
    }
}

int main(int argc, char **argv)
{
    const char *package = NULL;
    const char *image_path = NULL;
    const char *out_path = NULL;
    int loops = 1;
    for (int index = 1; index < argc; ++index) {
        if (strcmp(argv[index], "--out") == 0 && index + 1 < argc) {
            out_path = argv[++index];
        } else if (strcmp(argv[index], "--loops") == 0 && index + 1 < argc) {
            loops = atoi(argv[++index]);
        } else if (package == NULL) {
            package = argv[index];
        } else if (image_path == NULL) {
            image_path = argv[index];
        }
    }
    if (package == NULL || image_path == NULL || loops < 1) {
        fprintf(stderr, "usage: mipk_run <package|recipe-dir> <image> "
            "[--out annotated.png] [--loops N]\n");
        return 2;
    }

    image source = {0};
    int channels = 0;
    source.pixels = stbi_load(image_path, &source.width, &source.height,
        &channels, 3);
    if (source.pixels == NULL) {
        fprintf(stderr, "cannot decode %s: %s\n", image_path,
            stbi_failure_reason());
        return 1;
    }

    mipk_posix_platform *platform = NULL;
    const mipk_posix_platform_options platform_options = {
        .struct_size = sizeof(platform_options),
    };
    mipk_status status = mipk_platform_posix_create(&platform_options,
        &platform);
    if (status != MIPK_OK) {
        fprintf(stderr, "platform: %s\n", mipk_status_string(status));
        return 1;
    }
    static char diagnostic[2048];
    const mipk_pipeline_open_options options = {
        .struct_size = sizeof(options),
        .max_output_handles = 1,
        .platform = mipk_platform_posix_context(platform),
        .diagnostic = diagnostic,
        .diagnostic_size = sizeof(diagnostic),
    };
    mipk_pipeline *pipeline = NULL;
    const double open_start = now_ms();
    status = mipk_pipeline_open(package, &options, &pipeline);
    if (status != MIPK_OK) {
        fprintf(stderr, "open %s: %s\n%s\n", package,
            mipk_status_string(status), diagnostic);
        return 1;
    }
    const mipk_pipeline_info *info = mipk_pipeline_get_info(pipeline);
    printf("pipeline: %s %s (%s), input %s %ux%u, open %.1f ms\n", info->name,
        info->version, info->task,
        mipk_data_format_string(info->input_format), info->input_width,
        info->input_height, now_ms() - open_start);
    if (info->input_format == MIPK_FORMAT_PCM_S16LE) {
        fprintf(stderr, "this example feeds images; the package wants audio\n");
        return 1;
    }

    letterbox geometry;
    uint8_t *canvas = letterbox_rgb(&source, (int)info->input_width,
        (int)info->input_height, &geometry);
    if (canvas == NULL) return 1;
    const size_t canvas_pixels = (size_t)info->input_width * info->input_height;
    uint8_t *nv12 = NULL;
    mipk_input input;
    if (info->input_format == MIPK_FORMAT_NV12 ||
        info->input_format == MIPK_FORMAT_NV21) {
        nv12 = rgb_to_nv12(canvas, (int)info->input_width,
            (int)info->input_height);
        if (nv12 == NULL) return 1;
        if (info->input_format == MIPK_FORMAT_NV21) {
            for (size_t index = 0; index < canvas_pixels / 2U; index += 2) {
                const uint8_t u = nv12[canvas_pixels + index];
                nv12[canvas_pixels + index] = nv12[canvas_pixels + index + 1];
                nv12[canvas_pixels + index + 1] = u;
            }
        }
        input = mipk_input_nv12(info->input_format, nv12, 0,
            nv12 + canvas_pixels, 0, info->input_width, info->input_height, 0);
    } else {
        if (info->input_format == MIPK_FORMAT_BGR) {
            for (size_t index = 0; index < canvas_pixels; ++index) {
                const uint8_t r = canvas[index * 3U];
                canvas[index * 3U] = canvas[index * 3U + 2U];
                canvas[index * 3U + 2U] = r;
            }
        }
        input = mipk_input_packed_image(canvas, info->input_width,
            info->input_height, 0);
    }

    mipk_output *output = NULL;
    status = mipk_pipeline_output_create(pipeline, &output);
    if (status != MIPK_OK) {
        fprintf(stderr, "output: %s\n", mipk_status_string(status));
        return 1;
    }
    double total_ms = 0.0;
    for (int loop = 0; loop < loops; ++loop) {
        input.pts_us = (uint64_t)loop * 33333U;
        const double start = now_ms();
        status = mipk_pipeline_run_output(pipeline, &input, output);
        total_ms += now_ms() - start;
        if (status != MIPK_OK && status != MIPK_AGAIN) {
            fprintf(stderr, "run: %s\n%s\n", mipk_status_string(status),
                diagnostic);
            return 1;
        }
    }
    printf("run: %.2f ms per frame over %d frame(s)\n", total_ms / loops, loops);
    report(pipeline, output, &geometry, out_path != NULL ? &source : NULL);
    if (out_path != NULL) {
        if (!stbi_write_png(out_path, source.width, source.height, 3,
                source.pixels, source.width * 3)) {
            fprintf(stderr, "cannot write %s\n", out_path);
            return 1;
        }
        printf("annotated: %s\n", out_path);
    }
    mipk_output_destroy(output);
    mipk_pipeline_close(pipeline);
    mipk_platform_posix_destroy(platform);
    free(nv12);
    free(canvas);
    stbi_image_free(source.pixels);
    return 0;
}
