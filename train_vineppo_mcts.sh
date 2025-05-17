#!/usr/bin/env bash
export HF_TOKEN=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
export HF_HOME=/mnt/core_llm/masoud/model_cache
export HF_HUB_CACHE=/mnt/core_llm/masoud/model_cache
export HF_HUB_CACHE_DIR=/mnt/core_llm/masoud/model_cache
export TRANSFORMERS_CACHE=/mnt/core_llm/masoud/model_cache

# WandB Configuration (Set your API key here or via environment variables)
export WANDB_API_KEY="475b4ed6f1a5029a6215b025c05ac280820fe7ce"
export WANDB_PROJECT="vineppo_mcts_imprisoned" # Updated project name
export WANDB_RUN_ID=$(date +%Y%m%d_%H%M%S)_$(echo "$MODEL_NAME" | tr '/' '_')_mcts

# Default values
MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR="./output/vineppo_mcts" # Updated output dir
NUM_EPOCHS=1
GRADIENT_ACCUMULATION_STEPS=4
ROLLOUT_BATCH_SIZE=4
NUM_PPO_UPDATES=150
LEARNING_RATE=5e-6
SCHEDULER_TYPE="cosine"
NUM_WARMUP_STEPS=50
GAMMA=0.99
# Use separate low/high clip eps
CLIP_EPS_LOW=0.2
CLIP_EPS_HIGH=0.28
# CLIP_EPS removed
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
Q_TABLE_PATH="" # For V-table initialization
# USE_Q_TABLE_VALUE removed
USE_REF_MODEL=true
ADVANTAGE_TYPE="td"
SAVE_STEPS=10
# New defaults
GENERATION_TEMPERATURE=0.7
REF_MODEL_UPDATE_STEPS=0 # Default disabled
HISTORY_WINDOW_SIZE=-1 # Default full history
ENTROPY_COEFF=0.01
LOSS_TYPE="default"
MAX_COMPLETION_LENGTH=100

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
    # Update clip_eps args
    --clip_eps_low)
      CLIP_EPS_LOW="$2"
      shift 2
      ;;
    --clip_eps_high)
      CLIP_EPS_HIGH="$2"
      shift 2
      ;;
    # Remove --clip_eps parsing
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
    # Removed --use_q_table_value flag processing
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
    # Add parsing for new args
    --generation_temperature)
      GENERATION_TEMPERATURE="$2"
      shift 2
      ;;
    --ref_model_update_steps)
      REF_MODEL_UPDATE_STEPS="$2"
      shift 2
      ;;
    --history_window_size)
      HISTORY_WINDOW_SIZE="$2"
      shift 2
      ;;
    --entropy_coeff)
      ENTROPY_COEFF="$2"
      shift 2
      ;;
    --loss_type)
      LOSS_TYPE="$2"
      shift 2
      ;;
    --max_completion_length)
      MAX_COMPLETION_LENGTH="$2"
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
  --num_train_epochs "$NUM_EPOCHS" # Used to calculate total steps in python script
  --num_ppo_updates "$NUM_PPO_UPDATES" # Used to calculate total steps in python script
  --learning_rate "$LEARNING_RATE"
  --scheduler_type "$SCHEDULER_TYPE"
  --num_warmup_steps "$NUM_WARMUP_STEPS"
  --mc_rollouts "$MC_ROLLOUTS"
  --gamma "$GAMMA"
  # Use low/high clip eps
  --clip_eps_low "$CLIP_EPS_LOW"
  --clip_eps_high "$CLIP_EPS_HIGH"
  # Remove --clip_eps
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
  # Add new args
  --generation_temperature "$GENERATION_TEMPERATURE"
  --ref_model_update_steps "$REF_MODEL_UPDATE_STEPS"
  --history_window_size "$HISTORY_WINDOW_SIZE"
  --entropy_coeff "$ENTROPY_COEFF"
  --loss_type "$LOSS_TYPE"
  --max_completion_length "$MAX_COMPLETION_LENGTH"
)

# Add q_table_path if it's set (for V-table init)
if [ -n "$Q_TABLE_PATH" ]; then
  CMD_ARGS+=(--q_table_path "$Q_TABLE_PATH")
fi

# Add no_ref_model flag if it's false
if [ "$USE_REF_MODEL" = false ]; then
  CMD_ARGS+=(--no_ref_model)
fi

# Use python -m for consistency, can switch back to accelerate launch if needed
# Note: Ensure the python environment has access to the verifiers package
# PYTHONPATH=. accelerate launch --config_file ./configs/zero3.yaml verifiers/examples/train_vineppo_mcts.py "${CMD_ARGS[@]}"
PYTHONPATH=. python -m verifiers.examples.train_vineppo_mcts "${CMD_ARGS[@]}" 