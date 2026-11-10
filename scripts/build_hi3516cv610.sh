#!/bin/sh
# Cross-builds the runtime, the svp_acl backend, the hisilicon platform and
# mipk_run for Hi3516CV610.  Needs the openEuler arm32 musl toolchain (on
# PATH or in CV610_TOOLCHAIN_PATH) and the SoC SDK in HI_SDK_ROOT.
set -eu
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
BUILD="${MIPK_BUILD_DIR:-$ROOT/build-hi3516cv610}"
: "${HI_SDK_ROOT:?set HI_SDK_ROOT to the unpacked Hi3516CV610 SDK}"
GEN=""
command -v ninja >/dev/null 2>&1 && GEN="-G Ninja"
# shellcheck disable=SC2086
cmake -S "$ROOT" -B "$BUILD" $GEN \
    -DCMAKE_TOOLCHAIN_FILE="$ROOT/cmake/toolchains/hi3516cv610.cmake" \
    -DMIPK_HI_SDK_ROOT="$HI_SDK_ROOT" \
    -DMIPK_BACKEND_SVP_ACL=ON -DMIPK_PLATFORM_HISILICON=ON \
    -DMIPK_BACKEND_NULL=OFF -DMIPK_BACKEND_ONNXRUNTIME=OFF \
    -DMIPK_BUILD_TESTS=OFF \
    -DMIPK_MODEL_YOLO=ON -DMIPK_MODEL_YOLO_POSE=ON \
    -DMIPK_MODEL_CLASSIFICATION=ON -DMIPK_MODEL_KEYPOINTS=ON \
    -DMIPK_OPS_AUDIO=ON -DMIPK_MODEL_MICROWAKEWORD=ON \
    "$@"
cmake --build "$BUILD"
echo "built: $BUILD/examples/host/mipk_run"
