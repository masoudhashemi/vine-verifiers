import os
import argparse
import torch
from transformers import TrainingArguments
from unittest.mock import patch
from vllm import LLM, SamplingParams
import pickle

from verifiers.utils.model_utils import get_model_and_tokenizer
from verifiers.trainers.vine_ppo import VinePPOTrainer
from verifiers.trainers.vine_buffer import VineBuffer
from verifiers.envs.imprisoned_gym_env import ImprisonedGymEnv
from verifiers.envs.two_treasures_maze_gym_env import TwoTreasuresMazeGymEnv
from torch.utils.data import Dataset
from rich.console import Console

console = Console()

class DummyDataset(Dataset):
    def __len__(self):
        return 1  # or any positive number

    def __getitem__(self, idx):
        return {}  # return an empty dict, as it will be ignored


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with VinePPO on the Imprisoned environment")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", 
                        help="Name of the model to use")
    parser.add_argument("--output_dir", type=str, default="./output", 
                        help="Directory to save the model")
    parser.add_argument("--num_train_epochs", type=int, default=3, 
                        help="Number of training epochs")
    parser.add_argument("--num_ppo_updates", type=int, default=100, 
                        help="Number of PPO updates per epoch")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, 
                        help="Batch size per device during training")
    parser.add_argument("--learning_rate", type=float, default=1e-5, 
                        help="Learning rate")
    parser.add_argument("--scheduler_type", type=str, default="cosine", choices=["linear", "cosine", "constant"],
                        help="Learning rate scheduler type (linear, cosine, or constant)")
    parser.add_argument("--num_warmup_steps", type=int, default=0,
                        help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--mc_rollouts", type=int, default=5, 
                        help="Number of Monte Carlo rollouts for value estimation")
    parser.add_argument("--mc_top_p", type=float, default=0.5, 
                        help="Top-p sampling for MC value estimation rollouts")
    parser.add_argument("--gamma", type=float, default=0.99, 
                        help="Discount factor")
    parser.add_argument("--clip_eps_low", type=float, default=0.2, 
                        help="PPO lower clip epsilon")
    parser.add_argument("--clip_eps_high", type=float, default=0.2, 
                        help="PPO upper clip epsilon")
    parser.add_argument("--beta", type=float, default=0.1, 
                        help="Beta coefficient for KL divergence penalty")
    parser.add_argument("--buffer_size", type=int, default=1000, 
                        help="Size of the replay buffer")
    parser.add_argument("--max_steps", type=int, default=20, 
                        help="Maximum steps per episode for main rollouts")
    parser.add_argument("--mc_max_steps", type=int, default=25, 
                        help="Maximum steps per episode for MC value estimation")
    parser.add_argument("--rollout_batch_size", type=int, default=2, 
                        help="Batch size for rollouts")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed")
    
    # Local vLLM specific arguments
    parser.add_argument("--vllm_device", type=str, default="cuda:1", 
                        help="Device for local vLLM instance (if not using server). E.g., 'cuda:0'.")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.99, 
                        help="GPU memory utilization for local vLLM instance.")
    parser.add_argument("--vllm_dtype", type=str, default="float16", 
                        help="Data type for local vLLM (float16, bfloat16).")
    
    # VLLM Server arguments
    parser.add_argument("--use_vllm_server", action="store_true",
                        help="Use a remote VLLM server instead of a local vLLM instance.")
    parser.add_argument("--vllm_host", type=str, default="0.0.0.0",
                        help="Hostname for the VLLM server.")
    parser.add_argument("--vllm_server_port", type=int, default=8000,
                        help="Port for the VLLM server API.")
    parser.add_argument("--vllm_group_port", type=int, default=51216,
                        help="Group port for VLLM server communication.")
    parser.add_argument("--vllm_connection_timeout", type=float, default=60.0,
                        help="Connection timeout for VLLM server.")

    parser.add_argument("--block_size", type=int, default=8000, 
                        help="Maximum sequence length (used by local vLLM and potentially by tokenizer).")
    parser.add_argument("--q_table_path", type=str, default=None, 
                        help="Path to the Q-table pickle file (optional)")
    parser.add_argument("--use_q_table_value", action="store_true", 
                        help="Use Q-table for value estimation instead of MC rollouts")
    parser.add_argument("--no_ref_model", action="store_false", dest="use_ref_model",
                        help="Do not use a separate reference model for KL divergence calculation.")
    parser.set_defaults(use_ref_model=True) # Default to True if --no_ref_model is not specified
    parser.add_argument("--advantage_type", type=str, default="gae", choices=["gae", "td"],
                        help="Type of advantage estimation to use (gae or td)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of steps to accumulate gradients before updating model weights")
    parser.add_argument("--save_steps", type=int, default=10,
                        help="Save checkpoint every X updates steps (effective optimizer steps).")
    parser.add_argument("--ref_model_update_steps", type=int, default=10, 
                        help="Update reference model every N optimizer steps. Set to 0 or negative to disable.")
    parser.add_argument("--loss_type", type=str, default="default", choices=["default", "grpo", "dr_grpo"],
                        help="Loss normalization type (default, grpo, bnpo, dr_grpo)")
    parser.add_argument("--generation_temperature", type=float, default=0.7,
                        help="Temperature for sampling during generation and for logit scaling in loss calculation")
    parser.add_argument("--history_window_size", type=int, default=-1,
                        help="Number of past steps to include in history for generation (-1 for full history)")
    parser.add_argument("--value_variance_threshold", type=float, default=0.01,
                        help="Minimum variance of step values within a rollout to keep it. Set to 0.0 or less to disable filtering.")
    parser.add_argument("--entropy_coeff", type=float, default=0.01,
                        help="Coefficient for the entropy bonus in the PPO loss.")
    parser.add_argument("--policy_loss_type", type=str, default="ppo", choices=["ppo", "reinforce"],
                        help="Policy loss calculation type (ppo or reinforce)")
    parser.add_argument("--ema_decay", type=float, default=0.0,
                        help="Exponential Moving Average decay factor for MC value estimation model (0.0 to disable)")
    parser.add_argument("--env_type", type=str, default="imprisoned", choices=["imprisoned", "two_treasures_maze"],
                        help="Type of environment to use (imprisoned or two_treasures_maze)")
    return parser.parse_args()

def init_vllm(model_name, vllm_device, vllm_gpu_memory_utilization, vllm_dtype, block_size):
    """Initialize vLLM for faster inference."""
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None
    )
    
    # Convert string dtype to torch dtype
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
    
    # Set random seeds
    torch.manual_seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Q-table if specified
    q_table = None
    if args.use_q_table_value and args.q_table_path:
        try:
            with open(args.q_table_path, 'rb') as f:
                q_table = pickle.load(f)
            console.print(f"[green]Successfully loaded Q-table from {args.q_table_path}[/green]")
            if not isinstance(q_table, dict):
                console.print("[yellow]Warning: Loaded Q-table is not a dictionary. Disabling Q-table usage.[/yellow]")
                q_table = None
                args.use_q_table_value = False
        except FileNotFoundError:
            console.print(f"[red]Error: Q-table file not found at {args.q_table_path}. Disabling Q-table usage.[/red]")
            args.use_q_table_value = False
        except Exception as e:
            console.print(f"[red]Error loading Q-table: {e}. Disabling Q-table usage.[/red]")
            q_table = None
            args.use_q_table_value = False
    elif args.use_q_table_value and not args.q_table_path:
        console.print("[yellow]Warning: --use_q_table_value specified but --q_table_path not provided. MC rollouts will be used.[/yellow]")
        args.use_q_table_value = False

    # Load model and tokenizer
    model, tokenizer = get_model_and_tokenizer(
        args.model_name,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
            "use_cache": False,
            "device_map": "cuda:0",
        }
    )
    
    # Initialize local vLLM instance if not using server
    llm_instance_for_trainer = None
    if not args.use_vllm_server:
        console.print("INFO: `use_vllm_server` is False. Initializing local vLLM instance.")
        llm_instance_for_trainer = init_vllm(
            args.model_name,
            args.vllm_device, # This should be a local device like "cuda:0" or "cuda:1"
            args.vllm_gpu_memory_utilization,
            args.vllm_dtype,
            args.block_size
        )
    else:
        console.print("INFO: `use_vllm_server` is True. Will attempt to connect to VLLM server.")
        console.print("[bold yellow]Reminder: Ensure the VLLM server is running and accessible, and a NATS server is running for it to connect to.[/bold yellow]")
        console.print(f"[yellow]The trainer will attempt to connect to: {args.vllm_host}:{args.vllm_server_port} (group port: {args.vllm_group_port})[/yellow]")
        # No need to call init_vllm() if using server. VLLMClient handles connection.
    
    # Create environment factory
    def env_factory():
        if args.env_type == "imprisoned":
            return ImprisonedGymEnv(max_steps=args.max_steps)
        elif args.env_type == "two_treasures_maze":
            return TwoTreasuresMazeGymEnv(max_steps=args.max_steps, seed=args.seed)
        else:
            raise ValueError(f"Unsupported environment type: {args.env_type}")
    
    # Create buffer
    buffer = VineBuffer(
        buffer_size=args.buffer_size,
        gamma=args.gamma,
        device=model.device,
        advantage_type=args.advantage_type
    )
    
    dummy_dataset = DummyDataset()

    # Create training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        report_to="wandb",
        seed=args.seed,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    
    # Instead, add custom attributes after initialization
    training_args.block_size = args.block_size
    training_args.scheduler_type = args.scheduler_type
    training_args.num_warmup_steps = args.num_warmup_steps
    training_args.max_train_steps = args.num_ppo_updates * args.num_train_epochs
    training_args.loss_type = args.loss_type
    training_args.policy_loss_type = args.policy_loss_type
    
    # Create trainer with the initialized vLLM model
    trainer = VinePPOTrainer(
        model=model,
        train_dataset=dummy_dataset,
        args=training_args,
        env_factory=env_factory,
        buffer=buffer,
        processing_class=tokenizer,
        mc_rollouts=args.mc_rollouts,
        mc_top_p=args.mc_top_p,
        gamma=args.gamma,
        clip_eps_low=args.clip_eps_low,
        clip_eps_high=args.clip_eps_high,
        beta=args.beta,
        rollout_batch_size=args.rollout_batch_size,
        # Pass the potentially None llm_instance_for_trainer
        llm_instance=llm_instance_for_trainer,
        # Pass VLLM Server related args
        use_vllm_server=args.use_vllm_server,
        vllm_host=args.vllm_host,
        vllm_server_port=args.vllm_server_port,
        vllm_group_port=args.vllm_group_port,
        vllm_connection_timeout=args.vllm_connection_timeout,
        mc_max_steps=args.mc_max_steps,
        q_table=q_table,
        use_q_table_value=args.use_q_table_value,
        use_ref_model=args.use_ref_model,
        ref_model_update_steps=args.ref_model_update_steps,
        generation_temperature=args.generation_temperature,
        history_window_size=args.history_window_size,
        entropy_coeff=args.entropy_coeff,
        ema_decay=args.ema_decay,
        value_variance_threshold=args.value_variance_threshold
    )
    
    # Train the model using the custom train method
    trainer.train(num_ppo_updates=args.num_ppo_updates)
    
    # Save the final model
    trainer.save_model(args.output_dir)
    
if __name__ == "__main__":
    main()
