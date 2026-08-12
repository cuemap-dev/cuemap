#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="${DIST_DIR:-${ROOT_DIR}/dist/npm-native}"
VERSION="${VERSION:-$(awk -F '"' '/^version = "/ { print $2; exit }' "${ROOT_DIR}/Cargo.toml")}"
TOKENIZER_URL="${TOKENIZER_URL:-https://cuemap.dev/assets/en_tokenizer.bin.gz}"
TOKENIZER_SHA256="${TOKENIZER_SHA256:-f54fd31ec463f8646d0239bb531a64e0210ed1ae02bf5e3b42aeeb9bff8305ba}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This release builder requires macOS to produce both Darwin binaries" >&2
  exit 1
fi

if [[ -z "${VERSION}" ]]; then
  echo "Could not read the package version from Cargo.toml" >&2
  exit 1
fi

for command in cargo curl docker gzip node npm rustup shasum; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "Required command not found: ${command}" >&2
    exit 1
  fi
done

if ! rustup target list --installed | grep -qx "x86_64-apple-darwin"; then
  rustup target add x86_64-apple-darwin
fi

rm -rf "${DIST_DIR}"
mkdir -p "${DIST_DIR}/binaries" "${DIST_DIR}/packages" "${DIST_DIR}/tarballs" "${DIST_DIR}/tokenizer"

echo "Downloading checksum-pinned tokenizer"
curl -fsSL --retry 3 "${TOKENIZER_URL}" -o "${DIST_DIR}/tokenizer/en_tokenizer.bin.gz"
actual_tokenizer_sha="$(shasum -a 256 "${DIST_DIR}/tokenizer/en_tokenizer.bin.gz" | awk '{print $1}')"
if [[ "${actual_tokenizer_sha}" != "${TOKENIZER_SHA256}" ]]; then
  echo "Tokenizer checksum mismatch: expected ${TOKENIZER_SHA256}, got ${actual_tokenizer_sha}" >&2
  exit 1
fi
gzip -dc "${DIST_DIR}/tokenizer/en_tokenizer.bin.gz" > "${DIST_DIR}/tokenizer/en_tokenizer.bin"

echo "Building Darwin ARM64 binary"
cargo build --manifest-path "${ROOT_DIR}/Cargo.toml" --locked --release --target aarch64-apple-darwin
cp "${ROOT_DIR}/target/aarch64-apple-darwin/release/cuemap" "${DIST_DIR}/binaries/cuemap-darwin-arm64"

echo "Building Darwin x64 binary"
cargo build --manifest-path "${ROOT_DIR}/Cargo.toml" --locked --release --target x86_64-apple-darwin
cp "${ROOT_DIR}/target/x86_64-apple-darwin/release/cuemap" "${DIST_DIR}/binaries/cuemap-darwin-x64"

echo "Building Linux x64 binary on Debian Bookworm"
linux_output="${DIST_DIR}/linux-x64-output"
docker buildx build \
  --platform linux/amd64 \
  --target native-binary \
  --output "type=local,dest=${linux_output}" \
  "${ROOT_DIR}"
cp "${linux_output}/cuemap" "${DIST_DIR}/binaries/cuemap-linux-x64"
rm -rf "${linux_output}"

write_package_json() {
  local output_path="$1"
  local package_name="$2"
  local os_name="$3"
  local cpu_name="$4"

  node -e '
    const fs = require("node:fs");
    const [output, name, version, os, cpu] = process.argv.slice(1);
    const manifest = {
      name,
      version,
      description: `Pre-compiled CueMap Engine for ${os} ${cpu}`,
      engines: { node: ">=18" },
      os: [os],
      cpu: [cpu],
      bin: { cuemap: "bin/cuemap" },
      files: ["bin", "assets", "README.md", "LICENSE"],
      repository: { type: "git", url: "https://github.com/cuemap-dev/cuemap.git" },
      author: "Kaan Demirel",
      license: "BSL-1.1",
      publishConfig: { access: "public" },
    };
    if (os === "linux") manifest.libc = ["glibc"];
    fs.writeFileSync(output, `${JSON.stringify(manifest, null, 2)}\n`);
  ' "${output_path}" "${package_name}" "${VERSION}" "${os_name}" "${cpu_name}"
}

stage_package() {
  local platform="$1"
  local os_name="$2"
  local cpu_name="$3"
  local binary_path="${DIST_DIR}/binaries/cuemap-${platform}"
  local package_name="@cuemap-dev/engine-${platform}"
  local package_dir="${DIST_DIR}/packages/engine-${platform}"

  mkdir -p "${package_dir}/bin" "${package_dir}/assets"
  cp "${ROOT_DIR}/scripts/npm-native-wrapper.cjs" "${package_dir}/bin/cuemap"
  cp "${binary_path}" "${package_dir}/bin/cuemap-native"
  cp "${DIST_DIR}/tokenizer/en_tokenizer.bin" "${package_dir}/assets/en_tokenizer.bin"
  cp "${ROOT_DIR}/scripts/npm-native-README.md" "${package_dir}/README.md"
  cp "${ROOT_DIR}/LICENSE" "${package_dir}/LICENSE"
  chmod 0755 "${package_dir}/bin/cuemap" "${package_dir}/bin/cuemap-native"
  write_package_json "${package_dir}/package.json" "${package_name}" "${os_name}" "${cpu_name}"

  npm pack "${package_dir}" --pack-destination "${DIST_DIR}/tarballs"
}

stage_package "darwin-arm64" "darwin" "arm64"
stage_package "darwin-x64" "darwin" "x64"
stage_package "linux-x64" "linux" "x64"

(
  cd "${DIST_DIR}/tarballs"
  shasum -a 256 ./*.tgz > SHA256SUMS
)

echo
echo "Native npm packages are ready in ${DIST_DIR}/tarballs"
