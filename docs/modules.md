# 模块参考

由 `tools/gen_module_docs.py` 从各模块的 `module.json` 与 `recipe.json` 生成，请勿手改。

## 能力一览

| 种类 | 名称 | 模块 | 版本 | 兼容自 | 输入端口 | 输出 |
| --- | --- | --- | --- | --- | --- | --- |
| operator | `bytetrack` | `ops/bytetrack` | 2.0.0 | 2.0.0 | `frame`, `detections` | results (track) |
| operator | `image_align` | `ops/image_align` | 2.0.0 | 2.0.0 | `image`, `landmarks` | image_view |
| operator | `image_preprocess` | `ops/image_preprocess` | 2.0.0 | 2.0.0 | `input` | tensors |
| operator | `nms` | `ops/nms` | 2.0.0 | 2.0.0 | `results` | results (inherit:results) |
| operator | `result_filter` | `ops/result_filter` | 2.0.0 | 2.0.0 | `results` | results (inherit:results) |
| operator | `roi_extract` | `ops/roi_extract` | 2.0.0 | 2.0.0 | `frame`, `results` | collection of image_view |
| operator | `tfmicrofrontend` | `ops/tfmicrofrontend` | 2.0.0 | 2.0.0 | `audio` | tensors |
| operator | `wake_word_trigger` | `ops/wake_word_trigger` | 2.0.0 | 2.0.0 | `scores` | results (classification) |
| postprocess | `classification` | `models/classification` | 2.0.0 | 2.0.0 | `tensors` | results (classification) |
| postprocess | `keypoints` | `models/keypoints` | 2.0.0 | 2.0.0 | `tensors` | results (keypoint) |
| postprocess | `microwakeword` | `models/microwakeword` | 2.0.0 | 2.0.0 | `tensors` | results (classification) |
| postprocess | `yolo` | `models/yolo` | 2.0.0 | 2.0.0 | `tensors` | results (detection) |
| postprocess | `yolo_pose` | `models/yolo_pose` | 2.0.0 | 2.0.0 | `tensors` | results (detection, keypoint) |
| backend | `null` | `backends/null` | 2.0.0 | 2.0.0 | - | - |
| backend | `onnxruntime` | `backends/onnxruntime` | 2.0.0 | 2.0.0 | - | - |
| backend | `svp_acl` | `backends/svp_acl` | 2.0.0 | 2.0.0 | - | - |

## 节点选项

### `bytetrack` (operator, `ops/bytetrack`)

| 选项 | 类型与范围 | 必填 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `track_high_threshold` | number, >= 0 <= 1 | 是 | `-` |  |
| `track_low_threshold` | number, >= 0 <= 1 | 是 | `-` |  |
| `new_track_threshold` | number, >= 0 <= 1 | 是 | `-` |  |
| `min_match_iou` | number, >= 0 <= 1 | 是 | `-` |  |
| `lost_timeout_ms` | number, > 0 | 是 | `-` |  |
| `max_prediction_dt_ms` | number, > 0 | 是 | `-` |  |
| `reset_gap_ms` | number, > 0 | 是 | `-` |  |
| `class_aware` | bool | 是 | `-` |  |
| `max_tracks` | int, >= 1 <= result_capacity | 是 | `-` |  |
| `emit_lost` | bool | 是 | `-` |  |
| `emit_removed` | bool | 是 | `-` |  |
| 约束 | order: `track_low_threshold`, `track_high_threshold`, `new_track_threshold` | | | 非递减 |
| 约束 | le: `max_prediction_dt_ms`, `reset_gap_ms` | | | 前者不大于后者 |
| 端口 `detections` | 接受 detection 结果 | | | |

### `image_align` (operator, `ops/image_align`)

| 选项 | 类型与范围 | 必填 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `max_items` | int, >= 1 |  | `-` |  |
| `output_width` | int, >= 1 | 是 | `-` |  |
| `output_height` | int, >= 1 | 是 | `-` |  |
| `point_ids` | array, length 3, items int, unique | 是 | `-` |  |
| `target_points` | array, length 3, items point | 是 | `-` |  |
| `min_score` | number, >= 0 <= 1 |  | `-` |  |

### `image_preprocess` (operator, `ops/image_preprocess`)

| 选项 | 类型与范围 | 必填 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `mode` | string, one of `stretch`, `letterbox` |  | `"stretch"` |  |
| `target_width` | int, >= 1 |  | `-` |  |
| `target_height` | int, >= 1 |  | `-` |  |
| `format` | string, one of `auto`, `nv12`, `float` |  | `"auto"` |  |
| `layout` | string, one of `nchw` |  | `"nchw"` |  |
| `input_order` | string, one of `rgb`, `bgr` |  | `-` |  |
| `output_order` | string, one of `rgb`, `bgr` |  | `-` |  |
| `mean` | number_or_array, length 3 |  | `0` |  |
| `scale` | number_or_array, length 3 |  | `1` |  |
| `letterbox_value` | number, >= 0 <= 255 |  | `114` |  |
| `interpolation` | string, one of `bilinear`, `nearest` |  | `"bilinear"` | float output only; NV12 output is always a direct copy |
| 约束 | together: `target_width`, `target_height` | | | 要么都写要么都不写 |

### `nms` (operator, `ops/nms`)

| 选项 | 类型与范围 | 必填 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `iou_threshold` | number, >= 0 <= 1 | 是 | `-` |  |
| `class_agnostic` | bool | 是 | `-` |  |
| `max_results` | int, >= 1 <= result_capacity | 是 | `-` |  |
| 端口 `results` | 接受 detection 结果 | | | |

### `result_filter` (operator, `ops/result_filter`)

| 选项 | 类型与范围 | 必填 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `class_allowlist` | array, items string |  | `[]` |  |
| `score_threshold` | number, >= 0 <= 1 |  | `-` |  |
| `top_k` | int, >= 1 | 是 | `-` |  |
| `max_items` | int, >= 1 <= result_capacity | 是 | `-` |  |
| 约束 | le: `top_k`, `max_items` | | | 前者不大于后者 |

### `roi_extract` (operator, `ops/roi_extract`)

| 选项 | 类型与范围 | 必填 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `max_items` | int, >= 1 | 是 | `-` |  |
| `scale` | number, >= 1 |  | `1` |  |
| `square` | bool |  | `false` |  |
| `min_size` | number, > 0 |  | `1` |  |

### `tfmicrofrontend` (operator, `ops/tfmicrofrontend`)

| 选项 | 类型与范围 | 必填 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `window_ms` | int, >= 1 |  | `30` |  |
| `step_ms` | int, >= 1 |  | `10` |  |
| `feature_channels` | int, >= 1 | 是 | `-` |  |
| `feature_frames` | int, >= 1 | 是 | `-` |  |
| `inference_stride` | int, >= 1 |  | `1` |  |
| `input_index` | int, >= 0 |  | `0` |  |
| `lower_band_hz` | number |  | `125` |  |
| `upper_band_hz` | number |  | `7500` |  |
| `noise_smoothing_bits` | int |  | `10` |  |
| `noise_even_smoothing` | number |  | `0.025` |  |
| `noise_odd_smoothing` | number |  | `0.06` |  |
| `noise_min_signal_remaining` | number |  | `0.05` |  |
| `pcan` | bool |  | `true` |  |
| `pcan_strength` | number |  | `0.95` |  |
| `pcan_offset` | number |  | `80` |  |
| `pcan_gain_bits` | int |  | `21` |  |
| `log_scale` | bool |  | `true` |  |
| `log_scale_shift` | int |  | `6` |  |
| `output_scale` | number |  | `0.0390625` |  |
| `layout` | string, one of `time_major`, `feature_major` |  | `"time_major"` |  |

### `wake_word_trigger` (operator, `ops/wake_word_trigger`)

| 选项 | 类型与范围 | 必填 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `score_threshold` | number, >= 0 <= 1 |  | `0.8` |  |
| `smoothing` | string, one of `none`, `mean`, `max` |  | `"mean"` |  |
| `smoothing_window` | int, >= 1 |  | `5` |  |
| `confirmation_hits` | int, >= 1 |  | `1` |  |
| `confirmation_gaps` | int, >= 0 |  | `0` |  |
| `cooldown_ms` | int, >= 0 |  | `2000` |  |
| 端口 `scores` | 接受 classification 结果 | | | |

### `classification` (postprocess, `models/classification`)

| 选项 | 类型与范围 | 必填 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `output_index` | int, >= 0 |  | `0` |  |
| `transform` | string, one of `probability`, `sigmoid`, `softmax` |  | `"probability"` |  |
| `class_output_indices` | array, items int, unique |  | `-` |  |
| `score_threshold` | number, >= 0 <= 1 |  | `-` |  |
| `top_k` | int, >= 1 <= result_capacity |  | `-` |  |
| `max_items` | int, >= 1 <= result_capacity |  | `-` |  |
| 结果条数 | 由 `top_k` 限制 | | | |

### `keypoints` (postprocess, `models/keypoints`)

| 选项 | 类型与范围 | 必填 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `output_index` | int, >= 0 |  | `0` |  |
| `point_count` | int, >= 1 <= result_capacity | 是 | `-` |  |
| `values_per_keypoint` | int, >= 2 <= 3 |  | `2` |  |
| `coordinates` | string, one of `normalized`, `pixels` |  | `"normalized"` |  |
| `class_id` | int, >= 0 |  | `0` |  |
| `instance_id` | int, >= 0 |  | `0` |  |
| 结果条数 | 由 `point_count` 限制 | | | |

### `microwakeword` (postprocess, `models/microwakeword`)

| 选项 | 类型与范围 | 必填 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `output_index` | int, >= 0 |  | `0` |  |
| `transform` | string, one of `probability`, `sigmoid`, `softmax` |  | `"probability"` |  |
| `class_output_indices` | array, items int, unique |  | `-` |  |

### `yolo` (postprocess, `models/yolo`)

| 选项 | 类型与范围 | 必填 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `output_index` | int, >= 0 |  | `0` |  |
| `layout` | string, one of `auto`, `attributes_first`, `candidates_first` |  | `"auto"` |  |
| `has_objectness` | bool |  | `false` |  |
| `score_threshold` | number, >= 0 <= 1 |  | `0.4` |  |
| `max_candidates` | int, >= 0 |  | `10000` |  |
| `max_results` | int, >= 1 <= result_capacity |  | `-` |  |

### `yolo_pose` (postprocess, `models/yolo_pose`)

| 选项 | 类型与范围 | 必填 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `output_index` | int, >= 0 |  | `0` |  |
| `keypoint_count` | int, >= 1 <= 256 |  | `17` |  |
| `score_threshold` | number, >= 0 <= 1 |  | `0.5` |  |
| `iou_threshold` | number, >= 0 <= 1 |  | `0.45` |  |
| `max_candidates` | int, >= 1 |  | `100` |  |
| `max_results` | int, >= 1 <= result_capacity |  | `-` |  |
| 约束 | le: `max_results`, `max_candidates` | | | 前者不大于后者 |

## 标准流水线（recipe）

`python3 tools/mipk_tool.py new <recipe> --model <file> --labels <file|a,b> --output <dir> [--set name=value]`

### `classification`

Single-label image classification (ImageNet-style normalisation)

| 参数 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `model` | file | 必填 | model file, copied next to the manifest |
| `backend` | string | `"onnxruntime"` | backend module name |
| `labels` | labels | 必填 | class names, a file with one per line or a comma list |
| `input_format` | string | `"rgb"` | what the application feeds |
| `preprocess_format` | string | `"float"` | float for ONNX-style models, nv12 for .om with AIPP |
| `size` | int | `224` | model input size (square) |
| `input_width` | int | `"${size}"` | frame width the application feeds |
| `input_height` | int | `"${size}"` | frame height the application feeds |
| `mean` | json | `[123.675, 116.28, 103.53]` |  |
| `scale` | json | `[0.01712475, 0.017507, 0.01742919]` |  |
| `transform` | string | `"softmax"` |  |
| `top_k` | int | `5` |  |
| `score_threshold` | number | `0` |  |

流水线：`image_preprocess` → `model` → `classification`

### `keypoints`

Dense keypoint regression on a single crop (PFLD-style)

| 参数 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `model` | file | 必填 | model file, copied next to the manifest |
| `backend` | string | `"onnxruntime"` | backend module name |
| `labels` | labels | `["face"]` |  |
| `input_format` | string | `"rgb"` | what the application feeds |
| `preprocess_format` | string | `"float"` | float for ONNX-style models, nv12 for .om with AIPP |
| `size` | int | `112` | model input size (square) |
| `input_width` | int | `"${size}"` | frame width the application feeds |
| `input_height` | int | `"${size}"` | frame height the application feeds |
| `point_count` | int | `98` |  |
| `values_per_keypoint` | int | `2` |  |
| `coordinates` | string | `"normalized"` |  |
| `output_order` | string | `"bgr"` |  |

流水线：`image_preprocess` → `model` → `keypoints`

### `microwakeword`

MicroWakeWord: 16 kHz PCM, TF MicroFrontend features, streaming classifier

| 参数 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `model` | file | 必填 |  |
| `backend` | string | `"onnxruntime"` |  |
| `labels` | labels | 必填 | wake word names, one per output class |
| `result_capacity` | int | `1` | wake events reported per frame |
| `class_output_indices` | json | `[0]` | model output index for each wake word, in label order |
| `feature_channels` | int | `40` |  |
| `feature_frames` | int | `204` |  |
| `inference_stride` | int | `3` |  |
| `score_threshold` | number | `0.8` |  |
| `smoothing_window` | int | `5` | frames averaged before the threshold |
| `confirmation_hits` | int | `2` |  |
| `confirmation_gaps` | int | `2` |  |
| `cooldown_ms` | int | `2000` |  |

流水线：`tfmicrofrontend` → `model` → `microwakeword` → `wake_word_trigger`

### `yolo`

Ultralytics YOLO detection (v5/v8/v11 head without built-in NMS)

| 参数 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `model` | file | 必填 | model file, copied next to the manifest |
| `backend` | string | `"onnxruntime"` | backend module name |
| `labels` | labels | 必填 | class names, a file with one per line or a comma list |
| `input_format` | string | `"rgb"` | what the application feeds |
| `preprocess_format` | string | `"float"` | float for ONNX-style models, nv12 for .om with AIPP |
| `size` | int | `640` | model input size (square) |
| `model_width` | int | `"${size}"` | model input width |
| `model_height` | int | `"${size}"` | model input height |
| `input_width` | int | `"${model_width}"` | frame width the application feeds |
| `input_height` | int | `"${model_height}"` | frame height the application feeds |
| `score_threshold` | number | `0.25` |  |
| `iou_threshold` | number | `0.45` |  |
| `max_results` | int | `100` |  |
| `has_objectness` | bool | `false` | true for YOLOv5-style heads |
| `layout` | string | `"auto"` |  |
| `max_candidates` | int | 由 `yolo_anchors(model_width, model_height)` 推导 | anchors the head emits; derived from size |
| `track` | bool | `false` | append a ByteTrack stage |

流水线：`image_preprocess` → `model` → `yolo` → `nms` → `bytetrack` (可选)

### `yolo_pose`

Ultralytics YOLO pose head (boxes plus keypoints)

| 参数 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `model` | file | 必填 | model file, copied next to the manifest |
| `backend` | string | `"onnxruntime"` | backend module name |
| `labels` | labels | `["person"]` |  |
| `input_format` | string | `"rgb"` | what the application feeds |
| `preprocess_format` | string | `"float"` | float for ONNX-style models, nv12 for .om with AIPP |
| `size` | int | `640` | model input size (square) |
| `model_width` | int | `"${size}"` | model input width |
| `model_height` | int | `"${size}"` | model input height |
| `input_width` | int | `"${model_width}"` | frame width the application feeds |
| `input_height` | int | `"${model_height}"` | frame height the application feeds |
| `keypoint_count` | int | `17` |  |
| `score_threshold` | number | `0.5` |  |
| `iou_threshold` | number | `0.45` |  |
| `max_results` | int | `10` | objects per frame |
| `max_candidates` | int | 由 `yolo_anchors(model_width, model_height)` 推导 |  |

流水线：`image_preprocess` → `model` → `yolo_pose`

