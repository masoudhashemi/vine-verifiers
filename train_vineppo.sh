#!/usr/bin/env bash
export HF_TOKEN=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
export HF_HOME=/mnt/core_llm/masoud/model_cache
export HF_HUB_CACHE=/mnt/core_llm/masoud/model_cache
export HF_HUB_CACHE_DIR=/mnt/core_llm/masoud/model_cache
export TRANSFORMERS_CACHE=/mnt/core_llm/masoud/model_cache

# WandB Configuration (Set your API key here or via environment variables)
export WANDB_API_KEY="475b4ed6f1a5029a6215b025c05ac280820fe7ce"
export WANDB_PROJECT="vineppo_${ENV_TYPE}"
export WANDB_RUN_ID=$(date +%Y%m%d_%H%M%S)_$(echo "$MODEL_NAME" | tr '/' '_')_low${CLIP_EPS_LOW}_high${CLIP_EPS_HIGH}

# Default values
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
# MODEL_NAME="Qwen/Qwen3-0.6B"
OUTPUT_DIR="./output/vineppo_ema_${ENV_TYPE}"
NUM_EPOCHS=1
GRADIENT_ACCUMULATION_STEPS=6
ROLLOUT_BATCH_SIZE=5
NUM_PPO_UPDATES=1000
LEARNING_RATE=2e-6
SCHEDULER_TYPE="constant"
NUM_WARMUP_STEPS=10
GAMMA=0.99
CLIP_EPS_LOW=0.2
CLIP_EPS_HIGH=0.28
BETA=0.04
BUFFER_SIZE=1000
MAX_STEPS=8
MC_MAX_STEPS=8
MC_ROLLOUTS=20
MC_TOP_P=1.0
SEED=42
VLLM_DEVICE="cuda:1"
VLLM_GPU_MEMORY_UTILIZATION=0.99
VLLM_DTYPE="float16"
BLOCK_SIZE=8000
Q_TABLE_PATH=""
USE_Q_TABLE_VALUE=false
USE_REF_MODEL=true
ADVANTAGE_TYPE="gae"
SAVE_STEPS=100
REF_MODEL_UPDATE_STEPS=-1
LOSS_TYPE="grpo"
GENERATION_TEMPERATURE=1.0
HISTORY_WINDOW_SIZE=-1 # Default to -1 for full history
VALUE_VARIANCE_THRESHOLD=0.0 # Default minimum variance to keep rollout, <= 0 disables
ENTROPY_COEFF=0.0 # Default coefficient for entropy bonus
POLICY_LOSS_TYPE="ppo" # Default policy loss type, "reinforce" or "grpo"
EMA_DECAY=0 # Default EMA decay (0 = disabled)
ENV_TYPE="two_treasures_maze" # Default environment type, "imprisoned" or "two_treasures_maze"

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
    --mc_top_p)
      MC_TOP_P="$2"
      shift 2
      ;;
    --gamma)
      GAMMA="$2"
      shift 2
      ;;
    --clip_eps_low)
      CLIP_EPS_LOW="$2"
      shift 2
      ;;
    --clip_eps_high)
      CLIP_EPS_HIGH="$2"
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
    --loss_type)
      LOSS_TYPE="$2"
      shift 2
      ;;
    --generation_temperature)
      GENERATION_TEMPERATURE="$2"
      shift 2
      ;;
    --history_window_size)
      HISTORY_WINDOW_SIZE="$2"
      shift 2
      ;;
    --value_variance_threshold)
      VALUE_VARIANCE_THRESHOLD="$2"
      shift 2
      ;;
    --entropy_coeff)
      ENTROPY_COEFF="$2"
      shift 2
      ;;
    --policy_loss_type)
      POLICY_LOSS_TYPE="$2"
      shift 2
      ;;
    --ema_decay)
      EMA_DECAY="$2"
      shift 2
      ;;
    --env_type)
      ENV_TYPE="$2"
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
  --mc_top_p "$MC_TOP_P"
  --gamma "$GAMMA"
  --clip_eps_low "$CLIP_EPS_LOW"
  --clip_eps_high "$CLIP_EPS_HIGH"
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
  --generation_temperature "$GENERATION_TEMPERATURE"
)

# Add history_window_size if specified and >= 0
if [ "$HISTORY_WINDOW_SIZE" -ge 0 ]; then
  CMD_ARGS+=(--history_window_size "$HISTORY_WINDOW_SIZE")
fi

# Add loss_type if specified
if [ -n "$LOSS_TYPE" ]; then
  CMD_ARGS+=(--loss_type "$LOSS_TYPE")
fi

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

# Add value_variance_threshold
CMD_ARGS+=(--value_variance_threshold "$VALUE_VARIANCE_THRESHOLD")

# Add entropy_coeff
CMD_ARGS+=(--entropy_coeff "$ENTROPY_COEFF")

# Add policy_loss_type
CMD_ARGS+=(--policy_loss_type "$POLICY_LOSS_TYPE")

# Add ema_decay if specified and > 0
if (( $(echo "$EMA_DECAY > 0" | bc -l) )); then
  CMD_ARGS+=(--ema_decay "$EMA_DECAY")
fi

# Add env_type argument
CMD_ARGS+=(--env_type "$ENV_TYPE")

# PYTHONPATH=. accelerate launch --config_file ./configs/zero3.yaml verifiers/examples/train_vineppo.py \
PYTHONPATH=. python -m verifiers.examples.train_vineppo "${CMD_ARGS[@]}"
