#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET_REPO_ID="${DATASET_REPO_ID:-local/isaac_franka_front_wrist_state15_action4}"
DATASET_ROOT="${DATASET_ROOT:-$SCRIPT_DIR/outputs/lerobot_dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/outputs/smolvla_isaac_franka_front_wrist_state15_action4}"
POLICY_PATH="${POLICY_PATH:-lerobot/smolvla_base}"
LOCAL_POLICY_DIR="${LOCAL_POLICY_DIR:-$SCRIPT_DIR/outputs/pretrained/smolvla_isaac_franka_front_wrist_state15_action4_base}"
LOCAL_SOURCE_POLICY_DIR="${LOCAL_SOURCE_POLICY_DIR:-/home/mkls/xiao_run/lerobot_smolvla_mujoco_demo/outputs/pretrained/smolvla_panda_dualcam_state7_base}"
LOCAL_HF_CACHE_DIR="${LOCAL_HF_CACHE_DIR:-/home/mkls/xiao_run/lerobot_smolvla_mujoco_demo/.cache/huggingface}"
JOB_NAME="${JOB_NAME:-smolvla_isaac_franka_front_wrist_state15_action4}"
EPOCHS="${EPOCHS:-5}"
STEPS="${STEPS:-}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_FREQ="${LOG_FREQ:-20}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
EVAL_FREQ="${EVAL_FREQ:-0}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-lerobot}"
RESUME="${RESUME:-false}"
RESUME_CONFIG="${RESUME_CONFIG:-$OUTPUT_DIR/checkpoints/last/pretrained_model/train_config.json}"

export DATASET_REPO_ID DATASET_ROOT OUTPUT_DIR POLICY_PATH LOCAL_POLICY_DIR LOCAL_SOURCE_POLICY_DIR LOCAL_HF_CACHE_DIR JOB_NAME
export EPOCHS STEPS BATCH_SIZE NUM_WORKERS LOG_FREQ SAVE_FREQ EVAL_FREQ
export WANDB_ENABLE WANDB_PROJECT RESUME RESUME_CONFIG
export HF_HOME="${HF_HOME:-$SCRIPT_DIR/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-$HF_HUB_OFFLINE}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HUGGINGFACE_HUB_CACHE}"

DEFAULT_VLM_CACHE_REPO="$HUGGINGFACE_HUB_CACHE/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct"
FALLBACK_VLM_CACHE_REPO="$LOCAL_HF_CACHE_DIR/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct"
mkdir -p "$HUGGINGFACE_HUB_CACHE"
if [ ! -e "$DEFAULT_VLM_CACHE_REPO" ] && [ -d "$FALLBACK_VLM_CACHE_REPO" ]; then
  ln -s "$FALLBACK_VLM_CACHE_REPO" "$DEFAULT_VLM_CACHE_REPO"
fi

case "${RESUME,,}" in
  1|true|yes|y) RESUME_ENABLED=1 ;;
  *) RESUME_ENABLED=0 ;;
esac

if ! command -v lerobot-train >/dev/null 2>&1; then
  echo "lerobot-train not found. Activate your LeRobot environment first:"
  echo "  conda activate /home/mkls/xiao_run/.conda-lerobot-smolvla"
  exit 1
fi

if [ ! -f "$DATASET_ROOT/meta/info.json" ]; then
  echo "Dataset metadata not found:"
  echo "  $DATASET_ROOT/meta/info.json"
  echo
  echo "This training script expects a LeRobotDataset directory, not raw Isaac npz files."
  exit 1
fi

if [ -d "$OUTPUT_DIR" ] && [ "$RESUME_ENABLED" != "1" ]; then
  NEW_OUTPUT_DIR="${OUTPUT_DIR}_$(date +%Y%m%d_%H%M%S)"
  echo "Output directory already exists:"
  echo "  $OUTPUT_DIR"
  echo "Use a fresh output directory instead:"
  echo "  $NEW_OUTPUT_DIR"
  OUTPUT_DIR="$NEW_OUTPUT_DIR"
  export OUTPUT_DIR
fi

mkdir -p "$(dirname "$OUTPUT_DIR")"

if [ "$RESUME_ENABLED" = "1" ] && [ ! -f "$RESUME_CONFIG" ]; then
  AUTO_RESUME_CONFIG="$(find "$OUTPUT_DIR/checkpoints" -path "*/pretrained_model/train_config.json" 2>/dev/null | sort -V | tail -n 1 || true)"
  if [ -n "$AUTO_RESUME_CONFIG" ]; then
    RESUME_CONFIG="$AUTO_RESUME_CONFIG"
    export RESUME_CONFIG
  else
    echo "Resume config not found:"
    echo "  $RESUME_CONFIG"
    exit 1
  fi
fi

DATASET_SUMMARY="$(python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["DATASET_ROOT"])
info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
features = info["features"]
print(json.dumps(
    {
        "frames": int(info["total_frames"]),
        "episodes": int(info["total_episodes"]),
        "image_keys": sorted([k for k in features if k.startswith("observation.images.")]),
        "state_shape": features["observation.state"]["shape"],
        "action_shape": features["action"]["shape"],
    },
    ensure_ascii=False,
))
PY
)"
export DATASET_SUMMARY

DATASET_FRAMES="$(python - <<'PY'
import json
import os

summary = json.loads(os.environ["DATASET_SUMMARY"])
print(summary["frames"])
PY
)"
export DATASET_FRAMES

if [ -z "$STEPS" ]; then
  STEPS="$(python - <<'PY'
import math
import os

frames = int(os.environ["DATASET_FRAMES"])
epochs = float(os.environ["EPOCHS"])
batch_size = int(os.environ["BATCH_SIZE"])
print(math.ceil(frames * epochs / batch_size))
PY
)"
  STEPS_NOTE="computed from EPOCHS=$EPOCHS, frames=$DATASET_FRAMES, batch_size=$BATCH_SIZE"
else
  STEPS_NOTE="explicitly set by STEPS"
fi
export STEPS

echo "Dataset summary:"
python - <<'PY'
import json
import os

summary = json.loads(os.environ["DATASET_SUMMARY"])
print(f"  frames: {summary['frames']}")
print(f"  episodes: {summary['episodes']}")
print(f"  image_keys: {summary['image_keys']}")
print(f"  state_shape: {summary['state_shape']}")
print(f"  action_shape: {summary['action_shape']}")
PY

echo "Starting Isaac SmolVLA training:"
echo "  DATASET_REPO_ID=$DATASET_REPO_ID"
echo "  DATASET_ROOT=$DATASET_ROOT"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo "  POLICY_PATH=$POLICY_PATH"
echo "  RESUME=$RESUME"
echo "  RESUME_CONFIG=$RESUME_CONFIG"
echo "  EPOCHS=$EPOCHS"
echo "  DATASET_FRAMES=$DATASET_FRAMES"
echo "  STEPS=$STEPS"
echo "  STEPS_NOTE=$STEPS_NOTE"
echo "  BATCH_SIZE=$BATCH_SIZE"
echo "  NUM_WORKERS=$NUM_WORKERS"
echo

COMMON_ARGS=(
  --job_name="$JOB_NAME"
  --batch_size="$BATCH_SIZE"
  --num_workers="$NUM_WORKERS"
  --steps="$STEPS"
  --log_freq="$LOG_FREQ"
  --save_freq="$SAVE_FREQ"
  --eval_freq="$EVAL_FREQ"
  --wandb.enable="$WANDB_ENABLE"
  --wandb.project="$WANDB_PROJECT"
)

if [ "$RESUME_ENABLED" = "1" ]; then
  lerobot-train \
    --config_path="$RESUME_CONFIG" \
    --resume=true \
    "${COMMON_ARGS[@]}"
else
  TRAIN_POLICY_PATH="$POLICY_PATH"
  if [ "$POLICY_PATH" = "lerobot/smolvla_base" ] && [ -f "$LOCAL_POLICY_DIR/config.json" ]; then
    TRAIN_POLICY_PATH="$LOCAL_POLICY_DIR"
  elif [ "$POLICY_PATH" = "lerobot/smolvla_base" ]; then
    PREPARE_SOURCE="$POLICY_PATH"
    if [ -f "$LOCAL_SOURCE_POLICY_DIR/config.json" ]; then
      PREPARE_SOURCE="$LOCAL_SOURCE_POLICY_DIR"
    fi
    TRAIN_POLICY_PATH="$(
      python "$SCRIPT_DIR/prepare_smolvla_isaac_policy.py" \
        --source "$PREPARE_SOURCE" \
        --output "$LOCAL_POLICY_DIR" \
        --dataset-root "$DATASET_ROOT"
    )"
  fi

  lerobot-train \
    --dataset.repo_id="$DATASET_REPO_ID" \
    --dataset.root="$DATASET_ROOT" \
    --policy.path="$TRAIN_POLICY_PATH" \
    --policy.push_to_hub=false \
    --output_dir="$OUTPUT_DIR" \
    --resume=false \
    "${COMMON_ARGS[@]}"
fi
