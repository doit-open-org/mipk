#!/bin/sh
# Builds a set of module combinations to catch link dependencies between
# modules that a single "everything on" build hides. Each row configures a
# fresh tree, builds it, and runs the tests that apply when a local tests/
# directory is present. Pass -k to keep going
# after a failure.
set -u
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
OUT="${MIPK_MATRIX_DIR:-$ROOT/build-matrix}"
KEEP_GOING=0
[ "${1:-}" = "-k" ] && KEEP_GOING=1
GEN=""
command -v ninja >/dev/null 2>&1 && GEN="-G Ninja"
ORT=""
[ -f "$ROOT/third_party/onnxruntime/linux-x64/include/onnxruntime_c_api.h" ] && ORT=1

ALL_OFF="-DMIPK_OPS_IMAGE=OFF -DMIPK_OPS_DETECTION=OFF -DMIPK_OPS_TRACKING=OFF -DMIPK_OPS_AUDIO=OFF -DMIPK_BACKEND_NULL=OFF -DMIPK_BUILD_TESTS=OFF"
rows="
core_only|$ALL_OFF
null_backend_only|$ALL_OFF -DMIPK_BACKEND_NULL=ON
yolo_without_ops|$ALL_OFF -DMIPK_MODEL_YOLO=ON
yolo_pose_without_ops|$ALL_OFF -DMIPK_MODEL_YOLO_POSE=ON
classification_keypoints|$ALL_OFF -DMIPK_MODEL_CLASSIFICATION=ON -DMIPK_MODEL_KEYPOINTS=ON
audio_only|$ALL_OFF -DMIPK_OPS_AUDIO=ON -DMIPK_MODEL_MICROWAKEWORD=ON
image_ops_only|$ALL_OFF -DMIPK_OPS_IMAGE=ON
tracking_only|$ALL_OFF -DMIPK_OPS_TRACKING=ON
defaults_with_tests|-DMIPK_MODEL_YOLO=ON -DMIPK_MODEL_YOLO_POSE=ON -DMIPK_MODEL_CLASSIFICATION=ON -DMIPK_MODEL_KEYPOINTS=ON
everything|-DMIPK_OPS_AUDIO=ON -DMIPK_MODEL_YOLO=ON -DMIPK_MODEL_YOLO_POSE=ON -DMIPK_MODEL_CLASSIFICATION=ON -DMIPK_MODEL_KEYPOINTS=ON -DMIPK_MODEL_MICROWAKEWORD=ON
"
if [ -n "$ORT" ]; then
    rows="$rows
onnx_yolo|$ALL_OFF -DMIPK_OPS_IMAGE=ON -DMIPK_OPS_DETECTION=ON -DMIPK_MODEL_YOLO=ON -DMIPK_BACKEND_ONNXRUNTIME=ON -DMIPK_BUILD_TESTS=ON
onnx_everything|-DMIPK_OPS_AUDIO=ON -DMIPK_MODEL_YOLO=ON -DMIPK_MODEL_YOLO_POSE=ON -DMIPK_MODEL_CLASSIFICATION=ON -DMIPK_MODEL_KEYPOINTS=ON -DMIPK_MODEL_MICROWAKEWORD=ON -DMIPK_BACKEND_ONNXRUNTIME=ON
"
fi

mkdir -p "$OUT"
rows_file="$OUT/rows"
printf '%s\n' "$rows" >"$rows_file"
failed=""
while IFS='|' read -r name flags; do
    [ -z "$name" ] && continue
    dir="$OUT/$name"
    printf '==> %s\n' "$name"
    # shellcheck disable=SC2086
    if cmake -S "$ROOT" -B "$dir" $GEN $flags >"$dir.log" 2>&1 &&
       cmake --build "$dir" >>"$dir.log" 2>&1 &&
       { case "$flags" in *MIPK_BUILD_TESTS=OFF*) true ;;
           *) ctest --test-dir "$dir" --output-on-failure >>"$dir.log" 2>&1 ;; esac; }; then
        printf '    ok\n'
    else
        printf '    FAILED (see %s)\n' "$dir.log"
        [ "$KEEP_GOING" = 1 ] || exit 1
        failed="$failed $name"
    fi
done <"$rows_file"
if [ -n "$failed" ]; then
    printf 'failed:%s\n' "$failed"
    exit 1
fi
printf 'build matrix passed\n'
