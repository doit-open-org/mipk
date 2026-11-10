# manifest 与包格式

一个 mipk 包有两种形态，运行时和 `mipk_tool.py` 都接受：

- **recipe 目录**：`manifest.json` 加上模型文件、标签文件放在同一目录，便于开发时反复改；
- **`.mipk` 文件**：`mipk_tool.py create` 把 manifest 和模型打进一个带 CRC 的容器，用于部署。

各节点的选项、约束和标准流水线参数由 `module.json` 生成到 [modules.md](modules.md)，本文只讲结构。

## 顶层结构

```json
{
  "format_version": 2,
  "schema": "mipk.unified_pipeline.v2",
  "name": "yolo11n",
  "version": "1.0.0",
  "requires": {
    "runtime_min_version": "2.0.0",
    "capabilities": [
      { "kind": "postprocess", "name": "yolo", "abi_version": 2, "min_implementation_version": "2.0.0" }
    ]
  },
  "task": { "type": "vision_pipeline", "result_types": ["detection"], "result_capacity": 100 },
  "models": [ { "id": "detector", "backend": "onnxruntime", "file": "yolo11n.onnx" } ],
  "input": { "format": "rgb", "width": 640, "height": 640 },
  "label_sets": { "labels": { "file": "coco80.txt" } },
  "pipeline": [ ... ],
  "output": "detections"
}
```

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `format_version` | 是 | 固定为 `2` |
| `schema` | 是 | 固定为 `mipk.unified_pipeline.v2` |
| `name`, `version` | 否 | 默认 `pipeline` / `1.0.0`，通过 `mipk_pipeline_get_info()` 暴露给应用 |
| `requires.runtime_min_version` | 否 | 默认为打包工具自身的版本；运行时低于此版本拒绝打开 |
| `requires.capabilities` | 否 | 包依赖的模块清单。`create` 默认按流水线实际用到的模块自动填写（`--no-pin` 关闭）；条目必须都被用到 |
| `task.type` | 否 | `vision_pipeline` 或 `audio_pipeline`，缺省时由 `input.format` 推断 |
| `task.result_types` | 否 | 允许输出的结果类型子集，缺省为全部六种 |
| `task.result_capacity` | 否 | 每帧结果条数上限，默认 100；节点 `max_results` 之类的选项不能超过它 |
| `task.attachment_capacity`, `attachment_bytes` | 否 | 目前必须为 0 |
| `models[]` | 是 | 1 到 64 个；`id` 唯一，`backend` 是已编译的后端模块名，`file` 是相对 manifest 的路径，可选 `kind` |
| `input` | 是 | 见下 |
| `label_sets` | 是 | 见下 |
| `pipeline[]` | 是 | 1 到 32 个节点 |
| `output` | 是 | 作为流水线输出的节点 `output` 名，或 `$last` 表示最后一个节点 |

源文件统一按 `format_version`、`schema`、`name`、`version`、`requires`、`task`、
`models`、`input`、`label_sets`、`pipeline`、`output` 排列。`new` / `render`
保留名称和版本，省略默认结果容量、零附件配置、可由输入推断的任务类型和空依赖列表；
非默认容量、结果类型约束和显式运行时版本要求保留。打包时仍补齐运行时默认值。
标准配方和示例显式填写 `task.result_types`，说明最终输出允许包含哪些结果类型。

### input

图像输入（`vision_pipeline`）：

```json
{ "format": "nv12", "width": 1920, "height": 1080 }
```

`format` 为 `rgb`、`bgr`、`nv12`、`nv21` 之一，宽高是**应用喂入的帧尺寸**，不必等于模型尺寸；
`image_preprocess` 负责缩放到模型输入。

音频输入（`audio_pipeline`）：

```json
{ "format": "pcm_s16le", "sample_rate": 16000, "channels": 1, "frame_samples": 160, "reset_gap_ms": 2000 }
```

每次 `run` 送入 `frame_samples` 个采样；两帧 `pts_us` 间隔超过 `reset_gap_ms` 时流式状态被重置。

### label_sets

每个标签集是一个数组，元素为字符串或对象：

```json
"label_sets": {
  "labels": ["person", "bicycle", "car"],
  "faces": [
    { "name": "face", "source_id": 0, "enabled": true, "score_threshold": 0.6 },
    { "name": "mask", "enabled": false }
  ]
}
```

- `source_id`：模型输出里的类别序号，缺省为数组下标；
- `enabled: false`：该类别的结果被丢弃；
- `score_threshold`：按类别覆盖解码器的分数阈值。

在 recipe 目录里可以写 `{ "file": "coco80.txt" }`，每行一个名字，`#` 开头的行忽略；
打包时会内联成数组。

后处理节点用 `label_set` 指定自己的标签集，结果的 `label_set_id` 与 `class_id` 经
`mipk_pipeline_label_name()` 翻译成名字。

## pipeline 节点

三种节点共用 `id`、`type`、`inputs`、`output`、`execution` 字段；其余字段是该节点的选项。
`id` 可选，默认使用该节点的 `output` 名称。标准配方和示例省略节点 `id`，
用户需要独立于输出名称的标识时再显式填写。

```json
{
  "type": "operator",
  "operator": "nms",
  "iou_threshold": 0.45,
  "class_agnostic": false,
  "max_results": 100,
  "inputs": { "results": "candidates" },
  "output": "detections"
}
```

| `type` | 名称字段 | 说明 |
| --- | --- | --- |
| `operator` | `operator` | 通用算子：前处理、NMS、ROI 裁剪、跟踪、音频特征等 |
| `model` | `model` | 引用 `models[].id`，由该模型的后端执行；输入端口名是张量序号 `"0"`、`"1"`…；可选 `metadata_from` |
| `postprocess` | `postprocess` | 把模型输出张量解码成结果；可带 `label_set` |

`id` 缺省等于 `output`，运行时选项热更新用它定位节点。

### 端口连线

`inputs` 把节点的每个输入端口映射到一个数据源：

- `"$input"`：应用送入的原始帧或音频；
- 某个上游节点的 `output` 名；
- `{ "source": "model_output", "tensor_index": 1 }`：从多输出模型里选一个张量。

端口名和它能接受的值类型（帧、张量、结果、图像视图、集合）由模块声明，打包工具按类型检查每条连线。
结果类型也会沿着图传播：`nms` 声明 `inherit:results`，即输出类型与输入端口 `results` 相同；
`accepts` 限制该端口只接受检测结果。

### 逐项执行（per_item）

产生「集合」的节点（如 `roi_extract` 输出多个 ROI 视图）之后，下游节点用 `execution` 对每一项各跑一次：

```json
{
  "type": "model",
  "model": "landmark",
  "execution": { "mode": "per_item", "max_items": 4 },
  "inputs": { "0": "landmark_inputs" },
  "output": "landmark_tensors"
}
```

规则：

- 集合输入必须配 `per_item`，`per_item` 也必须有集合输入；
- 下游的 `max_items` 不能小于上游集合的上限；
- 有状态算子（如 `bytetrack`）不能 `per_item`。

`tests/data/multi_model_manifest.json` 是一个完整的「检测 → ROI → 关键点 → 对齐 → 特征 → 分类」级联示例。

## recipe.json

标准流水线由 `models/<族>/recipe.json` 定义：参数表加一个带 `${param}` 占位符的 manifest 模板，
`$if` 字段让某个节点按布尔参数可选（例如 `yolo` 的 `track` 参数追加 `bytetrack`）。
参数或模板里的值可以由其它参数推导：`"derive": {"expr": "(size // 16) ** 2", "args": ["size"]}`，
表达式只能引用 `args` 里的参数；仓库内的配方也可用 `"fn"` 指向工具内置的推导函数。

用户侧的 `recipe.json` 只记录选择：

```json
{ "recipe": "yolo", "model": "yolo11n.onnx", "labels": "coco80.txt", "track": true }
```

`mipk_tool.py new` 生成它并渲染 `manifest.json`；改过 `recipe.json` 后用 `render` 重新渲染。
需要自定义图时直接改 `manifest.json` 并删掉 `recipe.json`，否则下一次 `render` 会覆盖。

## 运行时选项覆盖

支持热更新的选项由模块在 `runtime_options` 里声明，可通过 `mipk_pipeline_runtime_option_at()` 枚举。
覆盖文件格式：

```json
{
  "schema": "mipk.runtime_overrides.v1",
  "nodes": {
    "detections": { "iou_threshold": 0.5, "class_agnostic": true },
    "candidates": { "score_threshold": 0.3 }
  }
}
```

`mipk_pipeline_load_overrides()` 读文件，`mipk_pipeline_apply_overrides_json()` 用字符串，
`mipk_pipeline_update_node_options()` 更新单个节点。三者都是事务性的：所有节点都校验通过才提交，
任一失败则整体不变。更新从不替换模型资产。

## .mipk 容器（v3）

所有整数小端：

```
header   32 B   magic "MIPK3\0\0\0", version u16, header_size u16, flags u16, reserved u16,
                manifest_size u32, entry_count u32, package_size u64
entries  32 B × entry_count
                type u32 (1 = model), flags u32, offset u64, size u64, crc32 u32, index u32
manifest manifest_size 字节的 UTF-8 JSON（已归一化：标签内联、默认值填齐）
payload  各模型按 entries 顺序排列
```

打开时校验文件大小、manifest 大小、每个条目的 CRC32，以及 `MIPK_PROFILE` 定义的资源上限
（manifest 最大尺寸、单模型最大字节数等，见 `core/include/mipk_config.h`）。`mipk_tool.py inspect`
可以打印容器内容而不需要运行时。
