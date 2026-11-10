#ifndef MIPK_PIPELINE_PRIVATE_H
#define MIPK_PIPELINE_PRIVATE_H

/*
 * The pipeline object, shared by mipk_manifest.c (which fills it from the
 * manifest) and mipk_pipeline.c (which opens, runs and closes it).
 */

#include "mipk.h"
#include "mipk_container.h"
#include "mipk_pipeline_internal.h"

struct mipk_pipeline {
    char *package_path;
    char *name;
    char *version;
    /* "vision_pipeline" or "audio_pipeline", derived from the input. */
    const char *task_name;
    bool audio;
    char *output;
    mipk_model_config *models;
    size_t model_count;
    mipk_stage_config *stages;
    size_t stage_count;
    mipk_label_set_config *label_sets;
    size_t label_set_count;
    mipk_container_model_source *model_sources;
    size_t model_source_count;
    mipk_pipeline_config config;
    mipk_pipeline_info info;
    mipk_pipeline_runtime *runtime;
    mipk_output *outputs;
    mipk_result *output_results;
    uint32_t max_output_handles;
    uint32_t active_output_handles;
    uint64_t audio_reset_gap_us;
    uint64_t last_audio_pts_us;
    bool has_last_audio_pts;
    const mipk_platform_context *platform;
    bool platform_retained;
    char *diagnostic;
    size_t diagnostic_size;
};

struct mipk_output {
    mipk_pipeline *owner;
    mipk_result_buffer results;
    uint64_t pts_us;
    bool leased;
};

/* Parses and validates the manifest JSON into the pipeline's config. */
mipk_status mipk_manifest_parse(const char *content, mipk_pipeline *pipeline);
/* Frees everything mipk_manifest_parse() allocated. */
void mipk_manifest_free(mipk_pipeline *pipeline);
bool mipk_manifest_find_label_set(const mipk_pipeline *pipeline,
    const char *name, uint32_t *label_set_id);

#endif
