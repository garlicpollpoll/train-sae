#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="${1:-configs/gemma3_12b_it_a100.yaml}"
SELECTED_PATH="${2:-outputs/selected_features.json}"

if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  PYTHON_BIN="python3"
fi

echo "[1/7] Generate dataset"
"$PYTHON_BIN" scripts/generate_experiment_dataset.py

echo "[2/7] Collect activations"
"$PYTHON_BIN" scripts/collect_activations.py --config "$CONFIG_PATH"

echo "[3/7] Train SAE"
"$PYTHON_BIN" scripts/train_sae.py --config "$CONFIG_PATH"

echo "[4/7] Score features"
"$PYTHON_BIN" scripts/score_features.py --config "$CONFIG_PATH"

echo "[5/7] Select features"
"$PYTHON_BIN" scripts/select_features.py --config "$CONFIG_PATH"

echo "[6/7] Build dense steering artifact"
"$PYTHON_BIN" scripts/build_artifact.py --config "$CONFIG_PATH" --selected "$SELECTED_PATH"

echo "[7/7] Build full SAE artifact"
"$PYTHON_BIN" scripts/build_full_artifact.py --config "$CONFIG_PATH" --selected "$SELECTED_PATH"

echo "Pipeline complete."
