import os
import argparse
import torch
from transformers import TrainingArguments
from unittest.mock import patch
from vllm import LLM, SamplingParams
import pickle

from verifiers.utils.model_utils import get_model_and_tokenizer
# Import the MCTS trainer and buffer
from verifiers.trainers.vine_ppo_mcts import VinePPOMCTSTrainer
from verifiers.trainers.vine_buffer_mcts import VineBuffer
from verifiers.envs.imprisoned_gym_env import ImprisonedGymEnv
from torch.utils.data import Dataset
from rich.console import Console

console = Console()

class DummyDataset(Dataset):
    def __len__(self):
        return 1  # or any positive number

    def __getitem__(self, idx):
        return {}  # return an empty dict, as it will be ignored

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with VinePPO MCTS on the Imprisoned environment")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Name of the model to use")
    parser.add_argument("--output_dir", type=str, default="./output_mcts",
                        help="Directory to save the model")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs (used to calculate total steps)")
    parser.add_argument("--num_ppo_updates", type=int, default=100,
                        help="Number of PPO optimizer updates per epoch")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="Effective batch size per device (usually 1 for PPO)")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--scheduler_type", type=str, default="cosine", choices=["linear", "cosine", "constant"],
                        help="Learning rate scheduler type (linear, cosine, or constant)")
    parser.add_argument("--num_warmup_steps", type=int, default=0,
                        help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--mc_rollouts", type=int, default=5,
                        help="Number of Monte Carlo rollouts for value estimation/update")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--clip_eps_low", type=float, default=0.2,
                        help="PPO lower clip epsilon")
    parser.add_argument("--clip_eps_high", type=float, default=0.28,
                        help="PPO upper clip epsilon")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="Beta coefficient for KL divergence penalty")
    parser.add_argument("--buffer_size", type=int, default=1000,
                        help="Size of the replay buffer")
    parser.add_argument("--max_steps", type=int, default=20,
                        help="Maximum steps per episode for main rollouts")
    parser.add_argument("--mc_max_steps", type=int, default=25,
                        help="Maximum steps per episode for MC value estimation rollouts")
    parser.add_argument("--rollout_batch_size", type=int, default=2,
                        help="Number of episodes to collect before a PPO update")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--vllm_device", type=str, default="cuda:1",
                        help="Device to run vLLM on")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.99,
                        help="GPU memory utilization for vLLM")
    parser.add_argument("--vllm_dtype", type=str, default="float16",
                        help="Data type for vLLM (float16, bfloat16)")
    parser.add_argument("--block_size", type=int, default=4096,
                        help="Maximum sequence length")
    parser.add_argument("--q_table_path", type=str, default=None,
                        help="Path to the Q-table pickle file for V-table initialization (optional)")
    parser.add_argument("--no_ref_model", action="store_false", dest="use_ref_model",
                        help="Do not use a separate reference model for KL divergence calculation.")
    parser.set_defaults(use_ref_model=True)
    parser.add_argument("--advantage_type", type=str, default="td", choices=["gae", "td"],
                        help="Type of advantage estimation to use (gae or td, td often pairs well with MCTS)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of steps to accumulate gradients before updating model weights")
    parser.add_argument("--save_steps", type=int, default=10,
                        help="Save checkpoint every X updates steps (effective optimizer steps).")
    parser.add_argument("--generation_temperature", type=float, default=0.7,
                        help="Temperature for sampling during generation and logit scaling")
    parser.add_argument("--ref_model_update_steps", type=int, default=0,
                        help="Update reference model every N optimizer steps (0=disable)")
    parser.add_argument("--history_window_size", type=int, default=-1,
                        help="Number of past steps for history context (-1 for full history)")
    parser.add_argument("--entropy_coeff", type=float, default=0.01,
                        help="Coefficient for the entropy bonus in the PPO loss.")
    parser.add_argument("--loss_type", type=str, default="default", choices=["default", "grpo", "dr_grpo"],
                        help="Loss normalization type (default, grpo, dr_grpo)")
    parser.add_argument("--max_completion_length", type=int, default=100,
                        help="Max completion length used for dr_grpo loss type")
    return parser.parse_args()

def init_vllm(model_name, vllm_device, vllm_gpu_memory_utilization, vllm_dtype, block_size):
    """Initialize vLLM for faster inference."""
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None
    )

    if vllm_dtype == "float16":
        torch_dtype = torch.float16
    elif vllm_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16
        console.print(f"[yellow]Warning: Unknown dtype {vllm_dtype}, using float16[/yellow]")

    console.print(f"Loading {model_name} on vLLM with device {vllm_device}")
    with world_size_patch, profiling_patch:
        llm = LLM(
            model=model_name,
            device=vllm_device,
            gpu_memory_utilization=vllm_gpu_memory_utilization,
            max_model_len=block_size,
            dtype=torch_dtype,
        )
    return llm

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load Q-table if specified for initialization
    q_table = None
    if args.q_table_path:
        try:
            with open(args.q_table_path, 'rb') as f:
                q_table = pickle.load(f)
            console.print(f"[green]Successfully loaded Q-table for V-table initialization from {args.q_table_path}[/green]")
            if not isinstance(q_table, dict):
                console.print("[yellow]Warning: Loaded Q-table is not a dictionary. Initializing V-table empty.[/yellow]")
                q_table = None
        except FileNotFoundError:
            console.print(f"[red]Error: Q-table file not found at {args.q_table_path}. Initializing V-table empty.[/red]")
        except Exception as e:
            console.print(f"[red]Error loading Q-table: {e}. Initializing V-table empty.[/red]")
            q_table = None
    else:
        console.print("[blue]No Q-table path provided. Initializing V-table empty.[/blue]")


    # Load model and tokenizer (assuming policy model on cuda:0)
    model, tokenizer = get_model_and_tokenizer(
        args.model_name,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
            "use_cache": False,
            "device_map": "cuda:0", # Explicitly map policy model to cuda:0
        }
    )

    # Initialize vLLM (potentially on a different device)
    vllm_instance = init_vllm(
        args.model_name,
        args.vllm_device,
        args.vllm_gpu_memory_utilization,
        args.vllm_dtype,
        args.block_size
    )

    # Create environment factory
    def env_factory():
        return ImprisonedGymEnv(max_steps=args.max_steps)

    # Create buffer
    buffer = VineBuffer(
        buffer_size=args.buffer_size,
        gamma=args.gamma,
        gae_lambda=0.95, # gae_lambda still needed if advantage_type is gae
        device="cuda:0", # Buffer itself doesn't use device much, but can store preference
        advantage_type=args.advantage_type
    )

    dummy_dataset = DummyDataset()

    # Create training arguments
    # Note: Total steps calculated based on epochs * ppo_updates
    total_training_steps = args.num_ppo_updates * args.num_train_epochs
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=total_training_steps, # Control total training steps
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        save_strategy="steps",
        save_steps=args.save_steps, # Saving based on optimizer steps
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10, # Log every 10 optimizer steps
        report_to="wandb",
        seed=args.seed,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        remove_unused_columns=False, # Important for custom trainer
    )

    # Add custom attributes needed by the trainer/scheduler
    training_args.block_size = args.block_size
    training_args.scheduler_type = args.scheduler_type
    training_args.num_warmup_steps = args.num_warmup_steps
    # Trainer uses max_train_steps for scheduler setup
    training_args.max_train_steps = total_training_steps // args.gradient_accumulation_steps
    training_args.loss_type = args.loss_type
    training_args.max_completion_length = args.max_completion_length

    # Create the MCTS Trainer
    trainer = VinePPOMCTSTrainer(
        model=model,
        train_dataset=dummy_dataset, # Dummy dataset as data comes from buffer
        args=training_args,
        env_factory=env_factory,
        buffer=buffer,
        processing_class=tokenizer,
        mc_rollouts=args.mc_rollouts,
        mc_max_steps=args.mc_max_steps,
        mc_top_p=0.2, # mc_top_p for logging, not V-table update
        rollout_batch_size=args.rollout_batch_size,
        gamma=args.gamma,
        clip_eps_low=args.clip_eps_low,
        clip_eps_high=args.clip_eps_high,
        beta=args.beta,
        generation_temperature=args.generation_temperature,
        q_table=q_table,
        vllm_llm=vllm_instance,
        block_size=args.block_size,
        use_ref_model=args.use_ref_model,
        ref_model_update_steps=args.ref_model_update_steps,
        history_window_size=args.history_window_size,
        entropy_coeff=args.entropy_coeff,
        loss_type=args.loss_type,
    )

    # Train the model using the custom train method
    # Pass the total number of *optimizer* steps
    trainer.train(num_ppo_updates=training_args.max_train_steps)

    # Save the final model
    trainer.save_model(args.output_dir)
    console.print(f"[green]Training complete. Final model saved to {args.output_dir}[/green]")

if __name__ == "__main__":
    main() 