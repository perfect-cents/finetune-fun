#!/usr/bin/env bash
#
# Package a trained artifact into a single .tar.gz for easy download off the pod.
#
# Usage:
#   bash pack.sh            # pack the LoRA adapter (default)
#   bash pack.sh adapter    # pack outputs/lora_adapter
#   bash pack.sh gguf       # pack outputs/gguf (the .gguf + Modelfile for Ollama)
#   bash pack.sh all        # pack both
#
# Then download the printed .tar.gz — e.g. `runpodctl send <file>` (then
# `runpodctl receive <code>` on your Mac), or via the JupyterLab file browser.

set -euo pipefail

what="${1:-adapter}"

pack_one() {
  local dir="$1"
  if [[ ! -d "$dir" ]]; then
    echo "SKIP: '$dir' not found (did training / export finish?)." >&2
    return 1
  fi
  local base out
  base="$(basename "$dir")"
  out="${base}.tar.gz"
  # -C so the archive contains a clean top-level folder, not the full path.
  tar czf "$out" -C "$(dirname "$dir")" "$base"
  echo "  packed $dir -> $out ($(du -h "$out" | cut -f1))"
}

echo "Packaging..."
case "$what" in
  adapter) pack_one outputs/lora_adapter ;;
  gguf)    pack_one outputs/gguf ;;
  all)     pack_one outputs/lora_adapter || true; pack_one outputs/gguf || true ;;
  *) echo "Unknown target '$what' — use: adapter | gguf | all" >&2; exit 1 ;;
esac

echo
echo "Download it, then TERMINATE the pod. Options:"
echo "  runpodctl send <file>.tar.gz     # then 'runpodctl receive <code>' on your Mac"
echo "  or right-click the .tar.gz in JupyterLab -> Download"
