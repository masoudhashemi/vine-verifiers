#!/usr/bin/env bash
export HF_TOKEN=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
export HF_HOME=/mnt/core_llm/masoud/model_cache
export HF_HUB_CACHE=/mnt/core_llm/masoud/model_cache
export HF_HUB_CACHE_DIR=/mnt/core_llm/masoud/model_cache
export TRANSFORMERS_CACHE=/mnt/core_llm/masoud/model_cache

# WandB Configuration (Set your API key here or via environment variables)
export WANDB_API_KEY="475b4ed6f1a5029a6215b025c05ac280820fe7ce"
export WANDB_PROJECT="vineppo_imprisoned"
export WANDB_RUN_ID=$(date +%Y%m%d_%H%M%S)_$(echo "$MODEL_NAME" | tr '/' '_')

# Default values
MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR="./output/vineppo"
NUM_EPOCHS=1
GRADIENT_ACCUMULATION_STEPS=1
ROLLOUT_BATCH_SIZE=4
NUM_PPO_UPDATES=100
LEARNING_RATE=5e-6
SCHEDULER_TYPE="cosine"
NUM_WARMUP_STEPS=10
GAMMA=0.99
CLIP_EPS=0.2
BETA=0.04
BUFFER_SIZE=1000
MAX_STEPS=8
MC_MAX_STEPS=30
MC_ROLLOUTS=20
SEED=42
VLLM_DEVICE="cuda:1"
VLLM_GPU_MEMORY_UTILIZATION=0.99
VLLM_DTYPE="float16"
BLOCK_SIZE=8000
Q_TABLE_PATH=""
USE_Q_TABLE_VALUE=false
USE_REF_MODEL=true
ADVANTAGE_TYPE="td"
SAVE_STEPS=10
REF_MODEL_UPDATE_STEPS=10

# Parse command line arguments
while [ $# -gt 0 ]; do
  case $1 in
    --model_name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --num_epochs)
      NUM_EPOCHS="$2"
      shift 2
      ;;
    --num_ppo_updates)
      NUM_PPO_UPDATES="$2"
      shift 2
      ;;
    --learning_rate)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --scheduler_type)
      SCHEDULER_TYPE="$2"
      shift 2
      ;;
    --num_warmup_steps)
      NUM_WARMUP_STEPS="$2"
      shift 2
      ;;
    --mc_rollouts)
      MC_ROLLOUTS="$2"
      shift 2
      ;;
    --gamma)
      GAMMA="$2"
      shift 2
      ;;
    --clip_eps)
      CLIP_EPS="$2"
      shift 2
      ;;
    --beta)
      BETA="$2"
      shift 2
      ;;
    --buffer_size)
      BUFFER_SIZE="$2"
      shift 2
      ;;
    --max_steps)
      MAX_STEPS="$2"
      shift 2
      ;;
    --mc_max_steps)
      MC_MAX_STEPS="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --vllm_device)
      VLLM_DEVICE="$2"
      shift 2
      ;;
    --vllm_gpu_memory_utilization)
      VLLM_GPU_MEMORY_UTILIZATION="$2"
      shift 2
      ;;
    --vllm_dtype)
      VLLM_DTYPE="$2"
      shift 2
      ;;
    --block_size)
      BLOCK_SIZE="$2"
      shift 2
      ;;
    --rollout_batch_size)
      ROLLOUT_BATCH_SIZE="$2"
      shift 2
      ;;
    --q_table_path)
      Q_TABLE_PATH="$2"
      shift 2
      ;;
    --use_q_table_value)
      USE_Q_TABLE_VALUE=true
      shift 1
      ;;
    --no_ref_model)
      USE_REF_MODEL=false
      shift 1
      ;;
    --advantage_type)
      ADVANTAGE_TYPE="$2"
      shift 2
      ;;
    --gradient_accumulation_steps)
      GRADIENT_ACCUMULATION_STEPS="$2"
      shift 2
      ;;
    --save_steps)
      SAVE_STEPS="$2"
      shift 2
      ;;
    --ref_model_update_steps)
      REF_MODEL_UPDATE_STEPS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Construct the command arguments
CMD_ARGS=(
  --model_name "$MODEL_NAME"
  --output_dir "$OUTPUT_DIR"
  --num_train_epochs "$NUM_EPOCHS"
  --num_ppo_updates "$NUM_PPO_UPDATES"
  --learning_rate "$LEARNING_RATE"
  --scheduler_type "$SCHEDULER_TYPE"
  --num_warmup_steps "$NUM_WARMUP_STEPS"
  --mc_rollouts "$MC_ROLLOUTS"
  --gamma "$GAMMA"
  --clip_eps "$CLIP_EPS"
  --beta "$BETA"
  --buffer_size "$BUFFER_SIZE"
  --max_steps "$MAX_STEPS"
  --mc_max_steps "$MC_MAX_STEPS"
  --seed "$SEED"
  --vllm_device "$VLLM_DEVICE"
  --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION"
  --vllm_dtype "$VLLM_DTYPE"
  --rollout_batch_size "$ROLLOUT_BATCH_SIZE"
  --advantage_type "$ADVANTAGE_TYPE"
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --save_steps "$SAVE_STEPS"
)

# Add ref_model_update_steps if set to a positive value
if [ "$REF_MODEL_UPDATE_STEPS" -gt 0 ]; then
  CMD_ARGS+=(--ref_model_update_steps "$REF_MODEL_UPDATE_STEPS")
fi

# Add q_table_path if it's set
if [ -n "$Q_TABLE_PATH" ]; then
  CMD_ARGS+=(--q_table_path "$Q_TABLE_PATH")
fi

# Add use_q_table_value flag if it's true
if [ "$USE_Q_TABLE_VALUE" = true ]; then
  CMD_ARGS+=(--use_q_table_value)
fi

# Add no_ref_model flag if it's false
if [ "$USE_REF_MODEL" = false ]; then
  CMD_ARGS+=(--no_ref_model)
fi

# PYTHONPATH=. accelerate launch --config_file ./configs/zero3.yaml verifiers/examples/train_vineppo.py \
PYTHONPATH=. python -m verifiers.examples.train_vineppo "${CMD_ARGS[@]}"
