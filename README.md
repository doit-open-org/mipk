# MIPK（Model Inference Package）

模块化的嵌入式 AI 推理流水线框架。用一份清单描述「前处理 → 模型 → 解码 → 后处理」的图，
连同模型打包成一个 `.mipk` 文件（Model Inference Package），由统一的 C 运行时在不同平台、
不同推理后端上执行。

应用只依赖公共头文件 `mipk.h` 和一个静态库 `mipk::mipk`；算子、模型解码器、推理后端、平台层
都是可插拔的模块，按需编译进库。

## 架构

```
应用 ──► mipk 运行时（core）──► 流水线图
                                  ├── operator     通用算子：图像前处理、ROI、对齐、NMS、跟踪、音频特征…
                                  ├── model        由 backend 执行：onnxruntime / svp_acl / …
                                  └── postprocess  模型族解码器：yolo、yolo_pose、classification、…
        platform 层：内存分配、cache 维护、NPU 仲裁、硬件图像前处理钩子
```

- **包（package）**：`manifest.json` 加模型与标签文件，开发期以目录形式运行，部署时打成带 CRC 的 `.mipk` 容器。
- **模块（module）**：每个目录一个模块，`module.json` 声明它的能力、端口和选项 schema，
  同时驱动构建注册、打包工具校验和文档生成，运行时与工具永不脱节。
- **后端（backend）**：同一个 manifest 只改 `models[].backend` 就能在 PC 上用 onnxruntime 验证，再切到 NPU 部署。
- **平台（platform）**：抽象出缓冲区分配、cache 同步和设备仲裁，运行时本身不依赖任何媒体框架。

## 支持范围

| 维度     | 已支持                                                                                                        |
| -------- | ------------------------------------------------------------------------------------------------------------- |
| 平台层   | POSIX（Linux / macOS / Windows 主机）、HiSilicon MPP（Hi3516CV610 等，MMZ + cache + NPU 仲裁）                |
| 推理后端 | onnxruntime（`.onnx`）、HiSilicon SVP ACL（`.om`）、null（图测试）                                        |
| 输入     | RGB / BGR / NV12 / NV21 帧，16 kHz PCM 音频                                                                   |
| 模型族   | YOLO 检测、YOLO-Pose、分类、关键点回归、MicroWakeWord 唤醒词；FastestDet、MediaPipe 手部在 `mipk_extra_module` |
| 算子     | image_preprocess、roi_extract、image_align、nms、result_filter、bytetrack、tfmicrofrontend、wake_word_trigger |
| 结果类型 | detection、classification、keypoint、attribute、track、text                                                   |

新增平台、后端或模型族都是加一个模块目录加一个 CMake 选项，不改核心。
产品专用的模块可以放在仓库外，通过 `MIPK_EXTRA_MODULE_DIRS` 编入，和内置模块走同一条路径；
平级的 `mipk_extra_module` 项目收纳这类模块。

## 快速开始

```sh
scripts/fetch_onnxruntime.sh
cmake -S . -B build -DMIPK_BACKEND_ONNXRUNTIME=ON -DMIPK_MODEL_YOLO=ON
cmake --build build

examples/host/models/yolo11n/export.sh        # 导出 yolo11n.onnx，需要 ultralytics
build/examples/host/mipk_run examples/host/models/yolo11n photo.jpg --out out.png
```

从模型到包：

```sh
python3 tools/mipk_tool.py new yolo --model yolov8n.onnx --labels coco80.txt --output pkg/yolov8n
python3 tools/mipk_tool.py create --manifest pkg/yolov8n/manifest.json \
    --model yolov8n.om --backend svp_acl --output yolov8n.mipk
```

`create` 可以用 `--target-capabilities` 对照目标固件构建时生成的 `mipk_capabilities.json`，
在打包阶段就发现缺模块。

## 使用

```c
mipk_pipeline *pipeline = NULL;
mipk_pipeline_open("yolov8n.mipk", &options, &pipeline);   /* options 里传平台上下文和诊断缓冲 */

mipk_output *output = NULL;
mipk_pipeline_output_create(pipeline, &output);
mipk_input input = mipk_input_nv12(MIPK_FORMAT_NV12, luma, 0, chroma, 0, width, height, pts_us);
mipk_pipeline_run_output(pipeline, &input, output);

for (size_t i = 0; i < mipk_output_result_count(output); ++i) {
    const mipk_result *r = mipk_output_result_at(output, i);
    /* r->type, r->class_id, r->score, r->value.bbox / keypoint / track ... */
}
```

运行时选项（如 NMS 阈值）可通过 `mipk_pipeline_update_node_options()` 事务性热更新，不重载模型。
完整示例见 [examples/host/mipk_run.c](examples/host/mipk_run.c)。

## 文档

- [构建与平台](docs/build.md)：CMake 选项、各平台的构建方式、平台层接口、编写模块、测试
- [manifest 与包格式](docs/manifest.md)：清单结构、端口连线、per_item、运行时覆盖、容器布局
- [模块参考](docs/modules.md)：每个算子、解码器的选项与约束，标准流水线参数（由 `tools/gen_module_docs.py` 生成）

## 许可证

Apache License 2.0，见 [LICENSE](LICENSE)。捆绑的第三方代码见 [NOTICE](NOTICE)。
