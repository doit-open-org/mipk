#ifndef MIPK_RUNTIME_H
#define MIPK_RUNTIME_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    MIPK_RUNTIME_OPTION_UNKNOWN = 0,
    MIPK_RUNTIME_OPTION_BOOL,
    MIPK_RUNTIME_OPTION_UINT32,
    MIPK_RUNTIME_OPTION_SIZE,
    MIPK_RUNTIME_OPTION_FLOAT,
    MIPK_RUNTIME_OPTION_STRING,
    MIPK_RUNTIME_OPTION_STRING_ARRAY,
    MIPK_RUNTIME_OPTION_JSON,
} mipk_runtime_option_type;

typedef struct {
    uint32_t struct_size;
    const char *name;
    mipk_runtime_option_type type;
    /* NULL for dimensionless values; otherwise a short unit such as "ms". */
    const char *unit;
    const char *description;
    /* Canonical JSON used when the option is omitted; NULL means required. */
    const char *default_json;
} mipk_runtime_option;

#define MIPK_MAX_TENSOR_DIMS 128U
#define MIPK_VIDEO_MAX_PLANES 3U
#define MIPK_PHYSICAL_ADDRESS_INVALID UINT64_MAX
#define MIPK_PTS_UNKNOWN UINT64_MAX
#define MIPK_INVALID_RESULT_ID UINT64_C(0)
#define MIPK_INVALID_ITEM_ID UINT64_C(0)
#define MIPK_INVALID_TRACK_ID INT64_C(-1)
#define MIPK_INVALID_ATTACHMENT_INDEX INT32_C(-1)
#define MIPK_INVALID_LABEL_SET_ID UINT32_MAX

typedef enum {
    MIPK_OK = 0,
    MIPK_AGAIN = 1,
    MIPK_ERR_ARGUMENT = -1,
    MIPK_ERR_MEMORY = -2,
    MIPK_ERR_IO = -3,
    MIPK_ERR_RUNTIME = -4,
    MIPK_ERR_MODEL = -5,
    MIPK_ERR_OPERATOR = -6,
    MIPK_ERR_CAPACITY = -7,
    MIPK_ERR_UNSUPPORTED = -8,
    MIPK_ERR_SECURITY = -9,
    MIPK_ERR_BUSY = -10,
} mipk_status;

/* Element type of a model tensor. Values match the Ascend ACL numbering. */
typedef enum {
    MIPK_DTYPE_FLOAT32 = 0,
    MIPK_DTYPE_FLOAT16 = 1,
    MIPK_DTYPE_INT8 = 2,
    MIPK_DTYPE_INT32 = 3,
    MIPK_DTYPE_UINT8 = 4,
    MIPK_DTYPE_INT16 = 6,
    MIPK_DTYPE_UINT16 = 7,
    MIPK_DTYPE_UINT32 = 8,
    MIPK_DTYPE_INT64 = 9,
    MIPK_DTYPE_UINT64 = 10,
    MIPK_DTYPE_DOUBLE = 11,
    MIPK_DTYPE_BOOL = 12,
} mipk_data_type;

/* Memory layout of a model tensor. Values match the Ascend ACL numbering. */
typedef enum {
    MIPK_LAYOUT_NCHW = 0,
    MIPK_LAYOUT_NHWC = 1,
    MIPK_LAYOUT_ND = 2,
    MIPK_LAYOUT_NC1HWC0 = 3,
} mipk_tensor_layout;

typedef enum {
    MIPK_MEMORY_CPU = 0,
    MIPK_MEMORY_PHYS_CACHED,
    MIPK_MEMORY_PHYS_UNCACHED,
} mipk_memory_type;

typedef enum {
    MIPK_BUFFER_OWNER_APPLICATION = 0,
    MIPK_BUFFER_OWNER_RUNTIME,
    MIPK_BUFFER_OWNER_BACKEND,
} mipk_buffer_owner;

typedef enum {
    MIPK_BUFFER_SYNC_UNKNOWN = 0,
    MIPK_BUFFER_SYNCED,
    MIPK_BUFFER_CPU_DIRTY,
    MIPK_BUFFER_DEVICE_DIRTY,
} mipk_buffer_sync_state;

typedef struct {
    void *virtual_address;
    uint64_t physical_address;
    size_t size;
    uint32_t alignment;
    mipk_memory_type memory_type;
    mipk_buffer_owner owner;
} mipk_buffer_desc;

typedef struct {
    mipk_buffer_desc desc;
    mipk_buffer_sync_state sync_state;
} mipk_buffer;

typedef struct {
    mipk_buffer *buffer;
    size_t offset;
    size_t byte_size;
    int32_t data_type;
    int32_t format;
    int64_t dims[MIPK_MAX_TENSOR_DIMS];
    size_t strides[MIPK_MAX_TENSOR_DIMS];
    size_t dim_count;
} mipk_tensor;

typedef mipk_tensor mipk_mutable_tensor;

typedef enum {
    MIPK_RESULT_DETECTION = 0,
    MIPK_RESULT_CLASSIFICATION,
    MIPK_RESULT_KEYPOINT,
    MIPK_RESULT_ATTRIBUTE,
    MIPK_RESULT_TRACK,
    MIPK_RESULT_TEXT,
} mipk_result_type;

typedef struct {
    float x1;
    float y1;
    float x2;
    float y2;
} mipk_bbox;

typedef struct { float x; float y; } mipk_point;
typedef struct { mipk_point points[4]; } mipk_quad;

typedef struct {
    float x;
    float y;
    float visibility;
    int32_t keypoint_id;
} mipk_keypoint;

typedef enum {
    MIPK_TRACK_NEW = 0,
    MIPK_TRACK_UPDATE,
    MIPK_TRACK_LOST,
    MIPK_TRACK_REMOVED,
} mipk_track_state;

typedef struct { mipk_bbox bbox; mipk_track_state state; } mipk_track;
typedef struct { mipk_quad quad; } mipk_text;

typedef struct {
    mipk_result_type type;
    uint64_t result_id;
    uint64_t parent_id;
    int64_t track_id;
    uint32_t label_set_id;
    int32_t class_id;
    float score;
    int32_t attachment_index;
    union {
        mipk_bbox bbox;
        mipk_keypoint keypoint;
        mipk_track track;
        mipk_text text;
    } value;
} mipk_result;

/* Runtime-owned staging buffer used by node implementations. */
typedef struct {
    mipk_result *items;
    size_t capacity;
    size_t count;
} mipk_result_buffer;

typedef enum {
    MIPK_INPUT_BYTES = 0,
    MIPK_INPUT_VIDEO_FRAME,
    MIPK_INPUT_AUDIO,
} mipk_input_type;

typedef struct {
    mipk_buffer_desc buffer;
    mipk_buffer_sync_state initial_sync_state;
    uint32_t stride;
} mipk_video_plane;

typedef struct {
    uint32_t format;
    uint32_t width;
    uint32_t height;
    uint32_t plane_count;
    mipk_video_plane planes[MIPK_VIDEO_MAX_PLANES];
} mipk_video_frame;

typedef struct {
    mipk_buffer_desc buffer;
    mipk_buffer_sync_state initial_sync_state;
    uint32_t sample_rate;
    uint32_t channels;
    uint32_t sample_count;
} mipk_audio_frame;

typedef struct {
    mipk_input_type type;
    uint64_t pts_us;
    union {
        struct {
            mipk_buffer_desc buffer;
            mipk_buffer_sync_state initial_sync_state;
        } bytes;
        mipk_video_frame video;
        mipk_audio_frame audio;
    } value;
} mipk_input;

static inline const void *mipk_input_data(const mipk_input *input)
{
    if (input == NULL) return NULL;
    if (input->type == MIPK_INPUT_BYTES) {
        return input->value.bytes.buffer.virtual_address;
    }
    if (input->type == MIPK_INPUT_AUDIO) {
        return input->value.audio.buffer.virtual_address;
    }
    if (input->type == MIPK_INPUT_VIDEO_FRAME &&
        input->value.video.plane_count != 0) {
        return input->value.video.planes[0].buffer.virtual_address;
    }
    return NULL;
}

static inline size_t mipk_input_data_size(const mipk_input *input)
{
    if (input == NULL) return 0;
    if (input->type == MIPK_INPUT_BYTES) return input->value.bytes.buffer.size;
    if (input->type == MIPK_INPUT_AUDIO) return input->value.audio.buffer.size;
    if (input->type == MIPK_INPUT_VIDEO_FRAME &&
        input->value.video.plane_count != 0) {
        size_t total = 0;
        for (uint32_t index = 0; index < input->value.video.plane_count;
            ++index) {
            const size_t size = input->value.video.planes[index].buffer.size;
            if (size > SIZE_MAX - total) return 0;
            total += size;
        }
        return total;
    }
    return 0;
}

const char *mipk_status_string(mipk_status status);

/*
 * Reports why something was refused.  Inside a pipeline call that was given
 * a diagnostic buffer (see mipk_pipeline_open_options) the text is appended
 * to that buffer; otherwise it goes to stderr.  Modules call this from
 * create() and validate() so an application can show the real reason a
 * package would not open.
 */
void mipk_diag(const char *format, ...)
#if defined(__GNUC__)
    __attribute__((format(printf, 1, 2)))
#endif
    ;

#ifdef __cplusplus
}
#endif
#endif
