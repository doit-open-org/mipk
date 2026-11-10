#!/bin/sh
# Downloads a prebuilt onnxruntime into third_party/onnxruntime/<platform>.
set -eu
VERSION="${ONNXRUNTIME_VERSION:-1.23.2}"
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
case "$(uname -s)-$(uname -m)" in
    Linux-x86_64) NAME="onnxruntime-linux-x64-$VERSION"; TARGET=linux-x64 ;;
    Linux-aarch64) NAME="onnxruntime-linux-aarch64-$VERSION"; TARGET=linux-aarch64 ;;
    Darwin-arm64) NAME="onnxruntime-osx-arm64-$VERSION"; TARGET=osx-arm64 ;;
    *) echo "unsupported host $(uname -s)-$(uname -m)" >&2; exit 1 ;;
esac
DEST="$ROOT/third_party/onnxruntime/$TARGET"
if [ -f "$DEST/include/onnxruntime_c_api.h" ]; then
    echo "onnxruntime already present: $DEST"
    exit 0
fi
mkdir -p "$ROOT/third_party/onnxruntime"
URL="https://github.com/microsoft/onnxruntime/releases/download/v$VERSION/$NAME.tgz"
echo "downloading $URL"
curl -L --fail -o "/tmp/$NAME.tgz" "$URL"
tar xzf "/tmp/$NAME.tgz" -C "$ROOT/third_party/onnxruntime"
mv "$ROOT/third_party/onnxruntime/$NAME" "$DEST"
rm -f "/tmp/$NAME.tgz"
echo "installed: $DEST"
