#!/usr/bin/env bash
#
# Launch TensorBoard to watch training loss/lr/grad-norm curves.
#
# Usage:
#   bash tensorboard.sh            # serve outputs/ on 0.0.0.0:6006
#   bash tensorboard.sh outputs 6007
#
# Training already logs here by default (train.py uses --report-to tensorboard,
# writing event files under outputs/runs/<timestamp>/). Run this in a SECOND
# terminal while (or after) training.
#
# On RunPod: expose HTTP port 6006 when you deploy the pod, then open the proxied
# URL from the pod's "Connect" menu. Binding to 0.0.0.0 (below) is what makes it
# reachable through that proxy.

set -euo pipefail

logdir="${1:-outputs}"
port="${2:-6006}"

if ! command -v tensorboard >/dev/null 2>&1; then
  echo "tensorboard not found — install it with: pip install tensorboard" >&2
  exit 1
fi

echo "Serving TensorBoard for '$logdir' on port $port (bind 0.0.0.0)..."
echo "RunPod: open the proxied URL for port $port from the pod's Connect menu."
exec tensorboard --logdir "$logdir" --host 0.0.0.0 --port "$port"
