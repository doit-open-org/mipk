# 构建与平台

## 目录结构

| 目录 | 内容 |
| --- | --- |
| `include/` | 公共头文件；应用只包含 `mipk.h`，模块作者包含 `mipk_module.h` |
| `core/` | 运行时：容器解析、manifest 解析、图构建、执行器、注册表、诊断 |
| `ops/` | 通用算子，每个子目录一个模块 |
| `models/` | 模型族的解码器与标准流水线（`recipe.json`） |
| `backends/` | 推理后端：`onnxruntime`、`svp_acl`、`null` |
| `platforms/` | 平台层：`posix`、`hisilicon` |
| `tools/` | `mipk_tool.py` 打包工具、`gen_module_docs.py` 文档生成器 |
| `examples/host/` | `mipk_run` 示例程序与 `yolo11n` 示例包 |
| `tests/` | 主机测试套件，只在本地保留，不入库；存在时自动编入 |
| `scripts/` | onnxruntime 下载、交叉编译、模块组合矩阵构建 |
| `cmake/` | `MipkModule.cmake` 模块机制、SDK 查找、工具链文件 |
| `third_party/` | cJSON、kissfft、TF MicroFrontend、stb；onnxruntime 预编译包由脚本下载 |

## CMake 选项

需要 CMake 3.19+、C11/C++17 编译器、Python 3。所有模块都是选项，未启用的不进入库；
流水线引用了未编译进库的模块时，`mipk_pipeline_open()` 失败并在诊断信息里指出模块名。

| 选项 | 默认 | 说明 |
| --- | --- | --- |
| `MIPK_OPS_IMAGE` | ON | `image_preprocess`、`roi_extract`、`image_align` |
| `MIPK_OPS_DETECTION` | ON | `nms`、`result_filter` |
| `MIPK_OPS_TRACKING` | ON | `bytetrack` |
| `MIPK_OPS_AUDIO` | OFF | `tfmicrofrontend`、`wake_word_trigger`（引入 TF MicroFrontend） |
| `MIPK_MODEL_YOLO` / `_YOLO_POSE` / `_CLASSIFICATION` / `_KEYPOINTS` | OFF | 各模型族解码器 |
| `MIPK_MODEL_MICROWAKEWORD` | OFF | 唤醒词解码器，需要 `MIPK_OPS_AUDIO` |
| `MIPK_EXTRA_MODULE_DIRS` | 空 | 仓库外模块目录列表（分号分隔），每个目录含 `module.json` 与 `CMakeLists.txt`，见[编写模块](#编写模块) |
| `MIPK_BACKEND_NULL` | ON | 空后端，仅供图测试 |
| `MIPK_BACKEND_ONNXRUNTIME` | OFF | onnxruntime 后端；`MIPK_ONNXRUNTIME_ROOT` 或环境变量 `ONNXRUNTIME_ROOT` 指定位置 |
| `MIPK_BACKEND_SVP_ACL` | OFF | HiSilicon SVP ACL 后端；`MIPK_HI_SDK_ROOT` 或环境变量 `HI_SDK_ROOT` 指定 SDK |
| `MIPK_PLATFORM_POSIX` | ON | POSIX 平台层 |
| `MIPK_PLATFORM_HISILICON` | OFF | HiSilicon MPP 平台层 |
| `MIPK_PROFILE` | host | 资源上限档位：`host` 或 `hi3516cv610` |
| `MIPK_BUILD_TESTS` / `MIPK_BUILD_EXAMPLES` | ON | 测试与示例 |
| `MIPK_WERROR` | ON | 警告视为错误 |
| `MIPK_USE_SECUREC` | OFF | 使用 securec 的 `memcpy_s` / `memset_s` |

每次构建都会在构建目录生成 `mipk_capabilities.json`，列出该库链接进去的全部模块与版本，
供 `mipk_tool.py validate / create --target-capabilities` 检查包能否在目标上打开。

### 资源档位

`MIPK_PROFILE=host` 使用 `core/include/mipk_config.h` 里的宽松默认值；`hi3516cv610` 是 32 MiB 设备档位
（manifest 64 KiB、单模型 12 MiB、输出 512 KiB、scratch 1 MiB、MMZ 16 MiB）。新增档位在顶层
`CMakeLists.txt` 里以 `MIPK_MAX_*` 宏覆盖即可。

## 各平台构建

### 主机（Linux / macOS / Windows）

```sh
scripts/fetch_onnxruntime.sh          # Windows: scripts\fetch_onnxruntime.ps1
cmake -S . -B build -G Ninja -DMIPK_BACKEND_ONNXRUNTIME=ON \
    -DMIPK_MODEL_YOLO=ON -DMIPK_MODEL_YOLO_POSE=ON \
    -DMIPK_MODEL_CLASSIFICATION=ON -DMIPK_MODEL_KEYPOINTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

Windows 下 MSVC 通过 `third_party/win32` 的兼容头把 core 用到的 POSIX 调用映射到 CRT，
onnxruntime 的 DLL 在构建后自动复制到可执行文件旁。

### HiSilicon SoC（交叉编译）

以 Hi3516CV610 为例，需要 openEuler arm32 musl 工具链（在 PATH 上或由 `CV610_TOOLCHAIN_PATH` 指定）
和解压后的 SoC SDK：

```sh
export HI_SDK_ROOT=/path/to/sdk      # 含 soc/include/svp_acl.h
scripts/build_hi3516cv610.sh
```

脚本等价于使用 `cmake/toolchains/hi3516cv610.cmake` 工具链文件并打开 `MIPK_BACKEND_SVP_ACL`
与 `MIPK_PLATFORM_HISILICON`；工具链文件会把 `MIPK_PROFILE` 固定为 `hi3516cv610`。
SDK 从不 vendoring 进仓库，`cmake/HiSDK.cmake` 负责查找它的头文件和 `soc/libs/musl` 下的库。

移植到其它 SoC：加一个工具链文件、在 `platforms/` 下实现平台层、在 `backends/` 下实现后端，
各自一个 CMake 选项。

## 平台层接口

平台层通过 `mipk_platform_context`（`include/mipk_platform.h`）向运行时提供：

- `buffer_alloc` / `buffer_free`：按 `mipk_memory_type`（CPU、物理连续 cached / uncached）分配缓冲区；
- `sync_for_device` / `sync_for_cpu`：cache 维护；
- `shared_arbiter`：进程级锁，多个流水线共用一个 NPU 时串行化提交；
- `image_backend`：可选的硬件图像前处理（`mipk_image_backend_ops`），`image_preprocess` 算子先问
  `supports()`，不支持的请求回退到软件路径。

`posix` 平台用 malloc、无 cache 维护；`hisilicon` 平台从 MMZ 分配、经 MPI 做 cache 维护、自带 NPU 互斥锁，
要求 MPP 系统已初始化（`ss_mpi_sys_init`）。硬件前处理（VGS、IVE）依赖媒体框架的通道，
因此由应用实现并通过 `mipk_hisilicon_platform_options.image_backend` 注入，平台层本身不依赖媒体栈。

```c
#include "mipk_platform_hisilicon.h"

const mipk_hisilicon_platform_options options = {
    .struct_size = sizeof(options),
    .shared_arbiter = NULL,            /* NULL 使用平台自带的 NPU 互斥锁 */
    .mmz_zone = NULL,                  /* NULL 使用默认 zone */
    .image_backend = &my_vgs_backend,  /* 可选 */
};
mipk_hisilicon_platform *platform = NULL;
mipk_platform_hisilicon_create(&options, &platform);
/* mipk_platform_hisilicon_context(platform) 传给 mipk_pipeline_open_options.platform */
```

## 编写模块

一个模块是一个目录，包含 `module.json`、`CMakeLists.txt` 和源码：

```
ops/my_filter/
├── module.json          # 名称、注册函数、能力（端口、选项 schema、结果类型）
├── CMakeLists.txt       # mipk_add_module(SOURCES mipk_my_filter.c)
├── mipk_my_filter.h
└── mipk_my_filter.c     # mipk_my_filter_register() 向注册表登记 operator_ops
```

`module.json` 同时驱动三件事：构建系统生成注册代码（`cmake/MipkModule.cmake`）、
打包工具校验选项、`docs/modules.md` 生成，所以运行时和工具不会对模块能力产生分歧。
模块只包含 `mipk_module.h`，看不到 `core/` 内部。

内置模块的步骤：在顶层 `CMakeLists.txt` 加一个选项并 `add_subdirectory()`；如果模块有可热更新的选项，
在 ops 里声明 `runtime_options`；运行 `python3 tools/gen_module_docs.py` 更新文档。

### 仓库外模块

产品专用的模块不必进 mipk 仓库。目录结构与内置模块完全一样，放在工程里，构建时用
`MIPK_EXTRA_MODULE_DIRS` 指向它：

```cmake
set(MIPK_EXTRA_MODULE_DIRS "${CMAKE_CURRENT_LIST_DIR}/modules/my_filter")   # 多个用分号分隔
add_subdirectory(${MIPK_ROOT} mipk)
```

mipk 会像内置模块一样 `add_subdirectory()` 它、把注册函数写进生成的注册表、归档进 `libmipk.a`，
并计入 `mipk_capabilities.json`；应用不需要手动调用注册函数。

打包工具用 `--module-dir` 指向模块目录的**父目录**（它扫描 `<父目录>/*/module.json`），
同时也会读到该目录下的 `recipe.json`：

```sh
python3 tools/mipk_tool.py --module-dir my_app/modules new my_recipe --output pkg/x ...
python3 tools/mipk_tool.py --module-dir my_app/modules create --manifest pkg/x/manifest.json \
    --target-capabilities build/mipk_capabilities.json --output x.mipk
```

`gen_module_docs.py` 只记录仓库内的模块，仓库外模块的选项以它自己的 `module.json` 为准。

## 测试

```sh
ctest --test-dir build --output-on-failure   # 主机测试
scripts/build_matrix.sh -k                   # 多种模块组合的构建矩阵，捕捉模块间的链接依赖
python3 tools/gen_module_docs.py --check     # 生成文档是否最新
```

端到端测试 `test_onnx_pipeline` 需要导出的 `yolo11n.onnx`；`test_onnx_toy` 用 Python `onnx` 包
现场生成一个玩具模型，缺少该包时自动跳过。CI（`.github/workflows/ci.yml`）在 Ubuntu 上跑全部模块的构建、
测试和构建矩阵。
