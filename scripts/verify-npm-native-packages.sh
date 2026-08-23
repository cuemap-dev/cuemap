#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="${DIST_DIR:-${ROOT_DIR}/dist/npm-native}"
VERIFY_DIR="${DIST_DIR}/verify"

rm -rf "${VERIFY_DIR}"
mkdir -p "${VERIFY_DIR}"

(
  cd "${DIST_DIR}/tarballs"
  shasum -a 256 -c SHA256SUMS
)

for platform in darwin-arm64 darwin-x64 linux-x64; do
  tarball="$(find "${DIST_DIR}/tarballs" -maxdepth 1 -name "cuemap-dev-engine-${platform}-*.tgz" -print -quit)"
  if [[ -z "${tarball}" ]]; then
    echo "Missing tarball for ${platform}" >&2
    exit 1
  fi

  package_dir="${VERIFY_DIR}/${platform}"
  mkdir -p "${package_dir}"
  tar -xzf "${tarball}" -C "${package_dir}"

  test -x "${package_dir}/package/bin/cuemap"
  test -x "${package_dir}/package/bin/cuemap-native"
  test -s "${package_dir}/package/assets/en_tokenizer.bin"
done

if [[ "$(uname -s)" == "Darwin" ]]; then
  case "$(uname -m)" in
    arm64)
      node "${ROOT_DIR}/scripts/verify-npm-native-runtime.cjs" \
        "${VERIFY_DIR}/darwin-arm64/package/bin/cuemap" darwin-arm64
      if arch -x86_64 /usr/bin/true >/dev/null 2>&1; then
        node "${ROOT_DIR}/scripts/verify-npm-native-runtime.cjs" \
          "${VERIFY_DIR}/darwin-x64/package/bin/cuemap" darwin-x64
      fi
      ;;
    x86_64)
      node "${ROOT_DIR}/scripts/verify-npm-native-runtime.cjs" \
        "${VERIFY_DIR}/darwin-x64/package/bin/cuemap" darwin-x64
      ;;
    *)
      echo "Unsupported macOS architecture: $(uname -m)" >&2
      exit 1
      ;;
  esac
fi

docker run --rm --platform linux/amd64 \
  -v "${VERIFY_DIR}/linux-x64/package:/package:ro" \
  -v "${ROOT_DIR}/scripts/verify-npm-native-runtime.cjs:/verify-runtime.cjs:ro" \
  node:20-trixie-slim \
  node /verify-runtime.cjs /package/bin/cuemap linux-x64

echo "Native package structure, checksums, tokenizer, ingestion, and recall checks passed"
