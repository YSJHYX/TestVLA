#!/usr/bin/env bash
set -euo pipefail
# Fill these in your shell / job environment, not in this repository:
# export G05_DATASET_DIR=       # LeRobot v2.1 root containing meta/, data/, videos/
# export G05_OUTPUT_DIR=        # Writable training output root
# export WANDB_PROJECT=        # Your W&B project; defaults to G0.5_665
# export WANDB_ENTITY=         # Optional account/team; defaults to your authenticated account
# WANDB_API_KEY is injected by your environment. No wandb.login() is called here.
# Optional: G05_NORM_MODE=z-score, G05_VAL_PROPORTION=0.05, NPROC_PER_NODE=1
# G05_VAL_DATASET_DIR: explicit held-out validation LeRobot dir, separate from
# G05_DATASET_DIR. When set, periodic eval reads ALL of its episodes (proportion
# split bypassed) and additionally fires at every checkpointing_steps node, so
# each saved checkpoint pairs with wandb eval metrics (eval/action/*, per_emb/*).
# Forward Hydra overrides; for step-based training set model.max_epochs=null with model.max_steps.
for name in G05_DATASET_DIR G05_OUTPUT_DIR; do
  if [[ -z "${!name:-}" ]]; then
    echo "Set $name in your environment; see docs/RM65_SG100_665.md" >&2
    exit 2
  fi
  # Resolve caller-relative paths before changing to the repository directory.
  export "$name=$(realpath -m -- "${!name}")"
done
if [[ -n "${G05_VAL_DATASET_DIR:-}" ]]; then
  export G05_VAL_DATASET_DIR="$(realpath -m -- "$G05_VAL_DATASET_DIR")"
  [[ -d "$G05_VAL_DATASET_DIR" ]] || { echo "G05_VAL_DATASET_DIR is not a directory: $G05_VAL_DATASET_DIR" >&2; exit 2; }
fi
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
# The three shared/pretrained resources always use the official repo layout.
for path in checkpoints/g05-base/checkpoints/model_state_dict.pt checkpoints/action_tokenizer.pt; do
  [[ -f "$path" ]] || { echo "Missing $ROOT/$path; populate the official checkpoints/ bundle before training" >&2; exit 2; }
done
[[ -d checkpoints/qwen3_5_2b_base_processor ]] || { echo "Missing $ROOT/checkpoints/qwen3_5_2b_base_processor" >&2; exit 2; }
export PYTHONPATH="$ROOT/src:$ROOT:${PYTHONPATH:-}"
python scripts/validate_665_data.py --dataset "$G05_DATASET_DIR"
overrides=()
if [[ -n "${G05_VAL_PROPORTION:-}" ]]; then
  overrides+=("data.val_set_proportion=$G05_VAL_PROPORTION")
fi
exec bash scripts/run/finetune.sh "${NPROC_PER_NODE:-1}" rm65_sg100_665 "${overrides[@]}" "$@"
