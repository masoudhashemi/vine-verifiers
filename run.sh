export HF_TOKEN=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
export HF_HOME=/mnt/core_llm/masoud/model_cache
export HF_HUB_CACHE=/mnt/core_llm/masoud/model_cache
export HF_HUB_CACHE_DIR=/mnt/core_llm/masoud/model_cache
export TRANSFORMERS_CACHE=/mnt/core_llm/masoud/model_cache

# Choose which trainer to run
TRAINER_TYPE=${1:-"grpo"}  # Default to GRPO if not specified

if [ "$TRAINER_TYPE" = "vineppo" ]; then
    # Run VinePPO trainer
    accelerate launch --config_file ./configs/zero3.yaml --num_processes 7 verifiers/examples/train_vineppo.py \
        --model_name "Qwen/Qwen2.5-3B-Instruct" \
        --n_rollouts 10 \
        --num_samples 1500
elif [ "$TRAINER_TYPE" = "grpo" ]; then
    # Run GRPO trainer
    accelerate launch --config_file ./configs/zero3.yaml --num_processes 7 verifiers/examples/imprisoned_trainer.py \
        --num_samples 1000 \
        --use_reward_filtering --reward_min_threshold 0.01 --reward_max_threshold 0.9 --max_reward 5.0
else
    echo "Unknown trainer type: $TRAINER_TYPE. Use 'vineppo' or 'grpo'."
    exit 1
fi

# Uncomment to run other examples
# accelerate launch --config_file ./configs/zero3.yaml --num_processes 7 verifiers/examples/gsm8k_calculator.py
# accelerate launch --config_file ./configs/zero3.yaml --num_processes 7 verifiers/examples/knights_knaves_simple.py \
#   --use_reward_filtering --reward_min_threshold 0.3 --reward_max_threshold 0.9 --max_reward 1.0
#   --use_sequential_sampler
