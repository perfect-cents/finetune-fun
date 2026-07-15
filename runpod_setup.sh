#!/usr/bin/env bash
#
# One-shot environment bootstrap for a fresh RunPod GPU pod.
#
# Usage (run from inside the cloned repo, on the pod):
#   bash runpod_setup.sh            # install deps + verify the GPU/stack
#   bash runpod_setup.sh --preview  # also render one training example as a final check
#
# Pick a RunPod PyTorch template (CUDA 12.1+), open a terminal, clone this repo,
# then run this script. See the "Running on RunPod" section of the README.

set -euo pipefail

echo "==> 1/4  GPU check (nvidia-smi)"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. Are you on a GPU pod? Pick a PyTorch/CUDA template." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo
echo "==> 2/4  Installing Python dependencies"
python -m pip install --upgrade pip
# Install everything except unsloth first (unsloth is version-sensitive).
grep -viE '^\s*(#|unsloth)' requirements.txt > /tmp/reqs_no_unsloth.txt
python -m pip install -r /tmp/reqs_no_unsloth.txt

echo
echo "==> 3/4  Installing Unsloth (matched to this box's CUDA/torch)"
# Recent Unsloth auto-detects the right build from a plain install. If this ever
# fails on your specific image, swap in the tagged git install documented in the
# README (e.g. unsloth[cu121-torch240]).
python -m pip install --upgrade unsloth

echo
echo "==> 4/4  Verifying the stack imports and sees the GPU"
python - <<'PY'
import torch
print(f"torch            {torch.__version__}")
print(f"cuda available   {torch.cuda.is_available()}")
assert torch.cuda.is_available(), "CUDA not visible to torch — wrong template or driver."
print(f"gpu              {torch.cuda.get_device_name(0)}")
import unsloth            # noqa: F401
print("unsloth          OK")
import bitsandbytes       # noqa: F401
print("bitsandbytes     OK")
import trl, peft, datasets
print(f"trl {trl.__version__}  peft {peft.__version__}  datasets {datasets.__version__}")
print("\nEnvironment ready.")
PY

if [[ "${1:-}" == "--preview" ]]; then
  echo
  echo "==> Bonus: rendering one training example (downloads the base model once)"
  python train.py --dataset ProCreations/grug-think --preview
fi

echo
echo "Next steps:"
echo "  python train.py --subsample 50 --max-steps 5   # 1-min smoke test (Option B)"
echo "  python train.py --report-to wandb              # the real run"
echo "  python eval.py --adapter outputs/lora_adapter --compare-base unsloth/Qwen2.5-3B-Instruct"
