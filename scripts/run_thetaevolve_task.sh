#!/bin/bash
set -euo pipefail

if [ $# -lt 4 ]; then
  echo "Usage: $0 TASK CONFIG_POSTFIX SMALL_MODEL_NAME SAVE_PATH"
  echo "Example: $0 ac1 it_XL dpsk_distill_qwen3_8b /data/thetaevolve"
  exit 1
fi

TASK="$1"
CONFIG_POSTFIX="$2"
SMALL_MODEL_NAME="$3"
SAVE_PATH="$4"

case "$TASK" in
  ac1|first|first_autocorr_inequality)
    EXAMPLE_TASK="first_autocorr_inequality"
    CONFIG_NAME="config_first_autocorr_inequality_${CONFIG_POSTFIX}.yaml"
    TASK_TAG="ac1"
    ;;
  ac2|second|second_autocorr_inequality)
    EXAMPLE_TASK="second_autocorr_inequality"
    CONFIG_NAME="config_second_autocorr_inequality_${CONFIG_POSTFIX}.yaml"
    TASK_TAG="ac2"
    ;;
  ac3|third|third_autocorr_inequality)
    EXAMPLE_TASK="third_autocorr_inequality"
    CONFIG_NAME="config_third_autocorr_inequality_${CONFIG_POSTFIX}.yaml"
    TASK_TAG="ac3"
    ;;
  erdos)
    EXAMPLE_TASK="erdos"
    CONFIG_NAME="config_erdos_${CONFIG_POSTFIX}.yaml"
    TASK_TAG="erdos"
    ;;
  circle26|circle_packing_n26|circle_packing_modular)
    EXAMPLE_TASK="circle_packing_modular"
    CONFIG_NAME="config_circle_packing_modular_it_XL.yaml"
    TASK_TAG="circle26"
    ;;
  circle32|circle_packing_n32)
    EXAMPLE_TASK="circle_packing_modular"
    CONFIG_NAME="config_circle_packing_modular_n32.yaml"
    TASK_TAG="circle32"
    ;;
  hadamard|hadamard_matrix)
    EXAMPLE_TASK="hadamard_matrix"
    CONFIG_NAME="config_hadamard_matrix_${CONFIG_POSTFIX}.yaml"
    TASK_TAG="hadamard"
    ;;
  *)
    echo "Unknown TASK: $TASK"
    echo "Supported tasks: ac1, ac2, ac3, erdos, circle26, circle32, hadamard"
    exit 1
    ;;
esac

case "$SMALL_MODEL_NAME" in
  dpsk_prorl_v2_1.5b)
    MODEL_FAMILY="nvidia"
    MODEL_NAME="Nemotron-Research-Reasoning-Qwen-1.5B"
    MODELS_FILE="deepseek-r1-distill-qwen-1.5B.sh"
    ;;
  dpsk_distill_qwen3_8b)
    MODEL_FAMILY="deepseek-ai"
    MODEL_NAME="DeepSeek-R1-0528-Qwen3-8B"
    MODELS_FILE="qwen3-8B.sh"
    ;;
  *)
    echo "Unknown SMALL_MODEL_NAME: $SMALL_MODEL_NAME"
    exit 1
    ;;
esac

mkdir -p "$SAVE_PATH"
mkdir -p "$SAVE_PATH/tmp" "$SAVE_PATH/hf" "$SAVE_PATH/wandb" "$SAVE_PATH/shm" "$SAVE_PATH/triton"

export TMPDIR=/tmp
export HF_HOME="$SAVE_PATH/hf"
export HUGGINGFACE_HUB_CACHE="$SAVE_PATH/hf/hub"
export TRANSFORMERS_CACHE="$SAVE_PATH/hf/hub"
export HF_DATASETS_CACHE="$SAVE_PATH/hf/datasets"
export SAVE_SHM_DIR="$SAVE_PATH/shm"
export TRITON_CACHE_DIR="$SAVE_PATH/triton"

export WANDB_CACHE_DIR="$SAVE_PATH/wandb"
export WANDB_DIR="$SAVE_PATH/wandb"
export WANDB_API_KEY="${WANDB_API_KEY:-aaa}"
export WANDB_ENTITY="${WANDB_ENTITY:-bbb}"
export WANDB_PROJECT="${WANDB_PROJECT:-ccc}"

INITIAL_PROGRAM="openevolve_adapted/examples/${EXAMPLE_TASK}/initial_programs/initial_program.py"
EVALUATOR_FILE="openevolve_adapted/examples/${EXAMPLE_TASK}/evaluators/evaluator_modular.py"
CONFIG_YAML="openevolve_adapted/examples/${EXAMPLE_TASK}/configs/${CONFIG_NAME}"
RUN_NAME="${SMALL_MODEL_NAME}_${TASK_TAG}_${CONFIG_POSTFIX}"

for REQUIRED_FILE in "$INITIAL_PROGRAM" "$EVALUATOR_FILE" "$CONFIG_YAML"; do
  if [ ! -f "$REQUIRED_FILE" ]; then
    echo "Missing required file: $REQUIRED_FILE"
    exit 1
  fi
done

if [ ! -d "$SAVE_SHM_DIR/$MODEL_NAME" ]; then
  hf download "$MODEL_FAMILY/$MODEL_NAME" --local-dir "./$MODEL_NAME"
  cp -r "$MODEL_NAME" "$SAVE_SHM_DIR/"
fi

source "scripts/models/${MODELS_FILE}"

if [ ! -d "$SAVE_SHM_DIR/${MODEL_NAME}_torch_dist" ]; then
  PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py "${MODEL_ARGS[@]}" --hf-checkpoint "$SAVE_SHM_DIR/$MODEL_NAME" --save "$SAVE_SHM_DIR/${MODEL_NAME}_torch_dist"
fi

bash "scripts_evolve/${MODEL_NAME}/general.sh" \
  "$WANDB_PROJECT" \
  "$RUN_NAME" \
  "$INITIAL_PROGRAM" \
  "$EVALUATOR_FILE" \
  "$CONFIG_YAML" \
  "$SAVE_PATH" \
  "True" \
  "1" \
  "original_reward" \
  "3407"
