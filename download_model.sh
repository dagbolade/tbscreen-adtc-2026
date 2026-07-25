# Download GGUF weights for TBScreen
#
# Rules:
#   - Must be idempotent (safe to run multiple times).
#   - Must download without any credentials (public URL only).
#   - The output path must match `_runtime.model_path` in metadata.json.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$HERE/model"
MODEL_FILE="$MODEL_DIR/gemma-4-E2B-it-Q4_K_M.gguf"
MODEL_URL="https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf"
# Pinned revision file SHA-256 (matches Hugging Face / Ollama blob).
EXPECTED_SHA256="9378bc471710229ef165709b62e34bfb62231420ddaf6d729e727305b5b8672d"

mkdir -p "$MODEL_DIR"

verify_sha() {
  local file="$1"
  local actual
  if command -v shasum >/dev/null 2>&1; then
    actual="$(shasum -a 256 "$file" | awk '{print $1}')"
  elif command -v sha256sum >/dev/null 2>&1; then
    actual="$(sha256sum "$file" | awk '{print $1}')"
  else
    echo "warning: no sha256 tool found — skipping checksum verify" >&2
    return 0
  fi
  if [[ "$actual" != "$EXPECTED_SHA256" ]]; then
    echo "error: SHA-256 mismatch for $file" >&2
    echo "  expected: $EXPECTED_SHA256" >&2
    echo "  actual:   $actual" >&2
    return 1
  fi
}

if [[ -f "$MODEL_FILE" ]] || [[ -L "$MODEL_FILE" ]]; then
  if verify_sha "$MODEL_FILE"; then
    echo "model already present and verified at $MODEL_FILE"
    exit 0
  fi
  echo "existing model failed checksum — re-downloading…"
  rm -f "$MODEL_FILE"
fi

echo "downloading $MODEL_URL → $MODEL_FILE (~2.9 GB)…"

if command -v curl >/dev/null 2>&1; then
  curl -L --fail --progress-bar -o "$MODEL_FILE.partial" "$MODEL_URL"
elif command -v wget >/dev/null 2>&1; then
  wget --show-progress -O "$MODEL_FILE.partial" "$MODEL_URL"
else
  echo "error: neither curl nor wget found" >&2
  exit 1
fi

verify_sha "$MODEL_FILE.partial"
mv "$MODEL_FILE.partial" "$MODEL_FILE"
echo "done: $MODEL_FILE (sha256=$EXPECTED_SHA256)"
