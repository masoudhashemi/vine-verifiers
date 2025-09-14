from contextlib import contextmanager
import copy
import os
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import Trainer, StoppingCriteria, StoppingCriteriaList, AutoModelForCausalLM
from typing import List, Tuple, Any, Optional, Union, Dict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.markup import escape
from torch.utils.data import DataLoader, Dataset

from unittest.mock import patch
from vllm import LLM, SamplingParams

from torch.optim import AdamW
from accelerate import Accelerator
from accelerate.utils import is_deepspeed_available
from verifiers.utils.training_utils import setup_scheduler
from verifiers.envs.two_treasures_maze_gym_env import TwoTreasuresMazeGymEnv

import re
from collections import defaultdict
import random

if is_deepspeed_available():
    import deepspeed
    DeepSpeedEngine = deepspeed.DeepSpeedEngine
else:
    DeepSpeedEngine = None

console = Console()

# @contextmanager
# def unwrap_model_for_generation(
#     model,
#     accelerator: "Accelerator",
#     is_peft_model: bool = False,
#     gather_deepspeed3_params: bool = True,
# ):
#     """Context manager to unwrap a model for generation."""
#     unwrapped_model = accelerator.unwrap_model(model)
#     if is_peft_model:
#         unwrapped_model.pretrained_model.disable_adapter()
#     if accelerator.state.deepspeed_plugin is not None and accelerator.state.deepspeed_plugin.zero_stage == 3:
#         if not gather_deepspeed3_params:
#             yield accelerator.unwrap_model(model)
#         else:
#             with deepspeed.zero.GatheredParameters(model.parameters()):
#                 yield accelerator.unwrap_model(model)
#     else:
#         yield unwrapped_model

class StopOnTag(StoppingCriteria):
    def __init__(self, tokenizer, stop_token: str):
        self.tokenizer = tokenizer
        self.stop_token_ids = self.tokenizer.encode(stop_token, add_special_tokens=False)
        self.found_stop_token = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        generated_ids = input_ids[0]
        if len(generated_ids) >= len(self.stop_token_ids):
            stop_token_tensor = torch.tensor(self.stop_token_ids, device=generated_ids.device)
            if torch.all(generated_ids[-len(self.stop_token_ids):] == stop_token_tensor):
                self.found_stop_token = True
                return True
        return False

class DummyDataset(Dataset):
    def __len__(self):
        return 1  # minimal non-zero length

    def __getitem__(self, idx):
        return {
            "input_ids": torch.ones((1, 1), dtype=torch.long),
            "attention_mask": torch.ones((1, 1), dtype=torch.long),
        }

class VinePPOTrainer(Trainer):
    """
    Custom Trainer implementing VinePPO for text-based environments using a chat template.
    It integrates:
    - Reduced GPU memory usage (by offloading the reference model)
    - Accelerate for device management
    - vLLM for batched Monte Carlo rollouts
    """
    def __init__(
        self,
        env_factory,
        buffer,
        processing_class,  # assumed to be a tokenizer that supports apply_chat_template
        mc_rollouts: int = 5,
        mc_max_steps: int = 25,
        mc_top_p: float = 0.5,
        rollout_batch_size: int = 2,
        gamma: float = 0.99,
        clip_eps_low: float = 0.2,
        clip_eps_high: float = 0.28,
        beta: float = 0.1,
        generation_temperature: float = 0.7,
        q_table: Optional[Dict[Tuple[str, str], float]] = None,
        use_q_table_value: bool = False,
        # vLLM configuration parameters:
        llm=None,
        block_size: int = 2048,
        use_ref_model: bool = True,
        ref_model_update_steps: int = 10,
        history_window_size: Optional[int] = None,
        value_variance_threshold: float = 0.01,
        entropy_coeff: float = 0.01,
        ema_decay: float = 0.0, # Set default to 0 to disable EMA unless specified
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.env_factory = env_factory
        self.buffer = buffer
        self.processing_class = processing_class
        self.mc_rollouts = mc_rollouts
        self.mc_max_steps = mc_max_steps
        self.mc_top_p = mc_top_p
        self.rollout_batch_size = rollout_batch_size
        self.gamma = gamma
        self.clip_eps_low = clip_eps_low
        self.clip_eps_high = clip_eps_high
        self.beta = beta
        self.generation_temperature = generation_temperature
        self.q_table = q_table
        self.use_q_table_value = use_q_table_value
        self.llm = llm
        self.use_ref_model = use_ref_model
        self.ref_model_update_steps = ref_model_update_steps
        self.history_window_size = history_window_size
        self.v_table = None
        self.average_v_table_value = 0.0
        self.entropy_coeff = entropy_coeff
        self.value_variance_threshold = value_variance_threshold

        # Add the new parameter for value variance threshold
        if self.value_variance_threshold <= 0:
            print(f"INFO: Rollout value variance filtering is DISABLED (threshold={self.value_variance_threshold})")
        else:
            print(f"INFO: Rollout value variance filtering is ENABLED (threshold={self.value_variance_threshold})")

        # --- Starting State Diversity ---
        self.possible_start_state_ids = []
        self.current_start_state_index = -1 # Initialize to -1 to start with index 0
        self.is_two_treasures_maze = False # Initialize flag
        if self.env_factory:
            try:
                temp_env_instance = self.env_factory()
                # Check if the instantiated env is TwoTreasuresMazeGymEnv
                if isinstance(temp_env_instance, TwoTreasuresMazeGymEnv) or \
                   (hasattr(temp_env_instance, 'env') and isinstance(temp_env_instance.env, TwoTreasuresMazeGymEnv)):
                    self.is_two_treasures_maze = True
                    console.print("[blue]INFO: Detected TwoTreasuresMazeGymEnv. Will use <thinking>action</thinking> format.[/blue]")
                else:
                    console.print("[blue]INFO: Standard environment detected. Will use <thinking>plan</thinking><action>act</action> format.[/blue]")

                if hasattr(temp_env_instance, 'env') and hasattr(temp_env_instance.env, 'starting_states'):
                    self.possible_start_state_ids = list(temp_env_instance.env.starting_states)
                    if self.possible_start_state_ids:
                        console.print(f"[blue]INFO: Found {len(self.possible_start_state_ids)} possible starting states: {self.possible_start_state_ids}[/blue]")
                    else:
                        console.print("[yellow]WARNING: No starting states found in the environment configuration.[/yellow]")
                elif hasattr(temp_env_instance, 'starting_states'): # Direct attribute for other env types
                    self.possible_start_state_ids = list(temp_env_instance.starting_states)
                    if self.possible_start_state_ids:
                        console.print(f"[blue]INFO: Found {len(self.possible_start_state_ids)} possible starting states (direct): {self.possible_start_state_ids}[/blue]")
                    else:
                        console.print("[yellow]WARNING: No starting states found directly in the environment.[/yellow]")
                else:
                    console.print("[yellow]WARNING: Could not retrieve starting states from the environment. Environment structure might be unexpected.[/yellow]")
            except Exception as e:
                console.print(f"[red]ERROR: Failed to get starting states from env_factory: {e}[/red]")
        else:
            console.print("[yellow]WARNING: env_factory is None, cannot determine starting states for diversity.[/yellow]")
        # --- End Starting State Diversity ---

        self.accelerator = Accelerator(device_placement=False)

        if not hasattr(self.args, 'include_num_input_tokens_seen'):
            self.args.include_num_input_tokens_seen = False
        if not hasattr(self.args, 'block_size'):
            self.args.block_size = block_size

        # --- EMA Model Initialization ---
        self.ema_decay = ema_decay
        self.use_ema_for_mc_value = self.ema_decay > 0
        self.ema_state_dict = None

        if self.use_ema_for_mc_value:
             print(f"INFO: EMA enabled for MC value estimation with decay {self.ema_decay}.")
             if self.llm is None:
                 console.print("[bold red]Error: EMA for MC value requires vLLM (`llm` instance) to be configured.[/bold red]")
                 console.print("[yellow]Warning: Disabling EMA for MC value due to missing vLLM.[/yellow]")
                 self.use_ema_for_mc_value = False
                 self.ema_decay = 0.0
             else:
                # Create EMA state dict on CPU *before* preparing the main model
                try:
                    model_to_copy = self.model # The model passed to super().__init__
                    # self.ema_model = copy.deepcopy(model_to_copy)
                    # self.ema_model.eval()
                    # for param in self.ema_model.parameters():
                    #     param.requires_grad = False
                    self.ema_state_dict = {k: v.cpu().detach().clone() for k, v in model_to_copy.state_dict().items()}
                    print("INFO: Created EMA state dict on CPU.")
                except Exception as e:
                    console.print(f"[bold red]Error creating EMA state dict: {e}[/bold red]")
                    console.print("[yellow]Warning: Disabling EMA for MC value due to creation error.[/yellow]")
                    # self.ema_model = None
                    self.ema_state_dict = None
                    self.use_ema_for_mc_value = False
                    self.ema_decay = 0.0
        else:
            print("INFO: EMA for MC value estimation is disabled.")
        # --- End EMA Model Initialization ---

        # Store the initial state dict *before* preparing the main model
        initial_state_dict = None
        model_name_or_path = None # Store model name/path
        model_dtype = None # Store model dtype
        if self.use_ref_model:
            # It's crucial that the model passed to super().__init__ is the one we snapshot
            model_name_or_path = self.model.config._name_or_path
            model_dtype = self.model.dtype
            initial_state_dict = copy.deepcopy(self.model.state_dict())
            print("INFO: Saved initial model state dict for reference model.")
            # Detach the main model from the graph briefly to ensure the copy is clean,
            # although deepcopy should handle this.
            # self.model = self.model.cpu() # Avoid moving if possible, deepcopy should be enough

        # Prepare the main model and optimizer using Accelerator
        # Ensure the optimizer is created *before* preparing it if it doesn't exist
        if not hasattr(self, 'optimizer') or self.optimizer is None:
            lr = self.args.learning_rate if hasattr(self.args, 'learning_rate') and self.args.learning_rate is not None else 1e-6
            # Filter parameters that require gradients for the optimizer
            optimizer_grouped_parameters = [
                {"params": [p for n, p in self.model.named_parameters() if p.requires_grad]}
            ]
            self.optimizer = AdamW(optimizer_grouped_parameters, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
            console.print(f"[yellow]Warning: No optimizer found or provided, created default AdamW optimizer with lr={lr}[/yellow]")
        else:
            # Ensure the existing optimizer only targets parameters requiring gradients
            self.optimizer.param_groups = [
                {"params": [p for n, p in self.model.named_parameters() if p.requires_grad], **group}
                for group in self.optimizer.param_groups
            ]

        # Prepare model and optimizer
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        print(f"INFO: Prepared main model device: {self.accelerator.device}") # Use accelerator device

        # Set up learning rate scheduler
        if hasattr(self.args, 'scheduler_type') and hasattr(self.args, 'num_warmup_steps') and hasattr(self.args, 'max_train_steps'):
            self.lr_scheduler = setup_scheduler(
                self.optimizer,
                self.args.num_warmup_steps,
                self.args.max_train_steps,
                self.accelerator,
                scheduler_type=self.args.scheduler_type
            )
            print(f"INFO: Initialized {self.args.scheduler_type} learning rate scheduler with {self.args.num_warmup_steps} warmup steps")
        else:
            self.lr_scheduler = None
            print("INFO: No learning rate scheduler configured")

        # Create and prepare the reference model *after* the main model is prepared
        self.ref_model = None
        if self.use_ref_model and initial_state_dict is not None and model_name_or_path is not None:
            try:
                print(f"INFO: Loading reference model from {model_name_or_path} with dtype {model_dtype}")
                # Load a fresh instance using the stored name/path and dtype
                self.ref_model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    torch_dtype=model_dtype # Use the stored dtype
                )
                self.ref_model.load_state_dict(initial_state_dict)
                self.ref_model.eval()
                for param in self.ref_model.parameters():
                    param.requires_grad = False

                self.ref_model.to(self.accelerator.device)
                # self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                print(f"INFO: Prepared reference model device: {self.accelerator.device}") # Should match main model
            except Exception as e:
                console.print(f"[bold red]Error creating or preparing reference model: {e}[/bold red]")
                console.print("[yellow]Warning: Proceeding without reference model due to error.[/yellow]")
                self.ref_model = None
                self.use_ref_model = False # Disable ref model usage if creation failed
        elif self.use_ref_model:
            console.print("[yellow]Warning: Could not create reference model (missing initial state or name). Disabling ref model usage.[/yellow]")
            self.use_ref_model = False

        if not self.use_ref_model:
            self.ref_model = None # Ensure ref_model is None if not used
            print("INFO: Reference model not used or creation failed.")

        # Create V-table from Q-table if applicable
        if self.use_q_table_value and self.q_table is not None:
            self._create_state_value_table_from_q_table()

        # Statistics tracking.
        self.episode_rewards = []
        self.episode_lengths = []
        self.buffer.device = self._get_device()

    def _get_device(self):
        return self.accelerator.device
        # return torch.device("cuda:0")
        # return self.model.device

    def _extract_action_tag(self, text: str) -> Optional[str]:
        """Extracts content within relevant tags based on environment type."""
        if self.is_two_treasures_maze:
            # For TwoTreasuresMaze, action is within <thinking>...</thinking>
            match = re.search(r"<thinking>(.*?)</thinking>", text, re.DOTALL | re.IGNORECASE)
        else:
            # For other envs, action is within <action>...</action>
            match = re.search(r"<action>(.*?)</action>", text, re.DOTALL | re.IGNORECASE)
        
        if match:
            return match.group(1).strip()
        return None # Return None if the relevant tag is not found or content is empty

    def get_train_dataloader(self) -> DataLoader:
        dataset = DummyDataset()
        dataloader = DataLoader(dataset, batch_size=1)
        return self.accelerator.prepare(dataloader)

    def _update_vllm_model(self):
        """Synchronize vLLM's model weights with the appropriate model (main or EMA state dict)."""
        if self.accelerator.is_main_process and self.llm is not None:
            state_dict_to_load = None
            model_source = ""

            # Check if using EMA and the state dict exists
            if self.use_ema_for_mc_value and self.ema_state_dict is not None:
                # Load EMA state dict (which is on CPU) for MC value estimation
                state_dict_to_load = self.ema_state_dict
                model_source = "EMA state dict (CPU)"
            else:
                # Load live policy model weights
                # Ensure we get the state dict correctly from potentially wrapped model
                # Use accelerator.get_state_dict for consistency
                state_dict_to_load = self.accelerator.get_state_dict(self.model)
                model_source = "Main policy model"

            if state_dict_to_load:
                try:
                    # Pass the state dict items (name, tensor tuples) to vLLM's load_weights
                    # vLLM should handle moving tensors to its internal devices.
                    llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                    llm_model.load_weights(list(state_dict_to_load.items())) # Pass list of (name, tensor) tuples
                    # print(f"DEBUG: Updated vLLM weights with {model_source}.") # Optional debug
                except Exception as e:
                    console.print(f"[bold red]Error updating vLLM weights with {model_source}: {e}[/bold red]")
            else:
                console.print("[yellow]Warning: Could not determine state dict to load into vLLM.[/yellow]")

    def _mc_value(self, env: Any, chat_history: List[Dict[str, str]], main_episode_current_step: int, main_episode_max_steps: int) -> float:
        """
        Compute the Monte Carlo value estimate using vLLM for batched generation.
        The length of MC rollouts is determined by the remaining steps in the main episode.
        """
        # If we are at or beyond the main episode's max steps, no further MC rollout is meaningful
        if main_episode_current_step >= main_episode_max_steps:
            console.print(f"[blue]MC Value: At/beyond main episode horizon ({main_episode_current_step}/{main_episode_max_steps}). Returning 0.[/blue]")
            return 0.0

        # Calculate the allowed number of steps for these MC rollouts
        # This will be > 0 because the condition above handles cases where it would be <= 0.
        rollout_specific_max_steps = main_episode_max_steps - main_episode_current_step
        
        console.print(f"[blue]MC Value: Setting MC rollout length to {rollout_specific_max_steps} (Main episode: {main_episode_current_step}/{main_episode_max_steps}, Configured mc_max_steps: {self.mc_max_steps})[/blue]")

        rollouts = []
        for _ in range(self.mc_rollouts):
            env_copy = env.clone() if hasattr(env, "clone") and callable(env.clone) else copy.deepcopy(env)
            # Set the MC-specific max steps for this environment copy
            env_copy.max_steps = rollout_specific_max_steps # Use the dynamically calculated max steps
            rollouts.append({
                "env": env_copy,
                "history": copy.deepcopy(chat_history),
                "done": False,
                "truncated": False,
                "reward": 0.0,
                "step_count": 0,
                "trajectory": []
            })

        while any(not (r["done"] or r["truncated"] or r["step_count"] >= r["env"].max_steps) for r in rollouts):
            active_indices = [i for i, r in enumerate(rollouts) if not (r["done"] or r["truncated"] or r["step_count"] >= r["env"].max_steps)]
            if not active_indices:
                break

            messages_to_step = [rollouts[i]["history"] for i in active_indices]
            
            # Check if any message history exceeds the block size
            messages_to_process = []
            indices_to_process = []
            for idx, message_history in zip(active_indices, messages_to_step):
                # Estimate token count by applying chat template and checking length
                tokenized = self.processing_class.apply_chat_template(message_history, tokenize=True, return_tensors="pt")
                if tokenized.size(1) >= self.args.block_size:
                    # Mark as truncated if exceeding block size
                    rollouts[idx]["truncated"] = True
                    console.print(f"[bold red]Rollout {idx} truncated: Message history exceeds block size ({tokenized.size(1)} tokens)[/bold red]")
                else:
                    messages_to_process.append(message_history)
                    indices_to_process.append(idx)
            
            if not indices_to_process:
                break
                
            # Truncate histories before passing to vLLM
            truncated_messages_to_process = [self._truncate_history(hist) for hist in messages_to_process]
            
            # --- Apply template and add '<thinking>' string --- 
            think_token_str = "<thinking>"
            prompts_for_vllm = []
            for hist in truncated_messages_to_process:
                # Apply template to the original history
                base_prompt_str = self.processing_class.apply_chat_template(hist, tokenize=False)
                # Append '<thinking>' string
                prompt_with_thinking = base_prompt_str + think_token_str
                prompts_for_vllm.append(prompt_with_thinking)
            # --- End Apply template and add '<thinking>' string ---

            sampling_params = SamplingParams(
                n=1,
                temperature=self.generation_temperature,
                top_p=0.95,
                max_tokens=200,
                include_stop_str_in_output=True,
                stop=["</action>", "</Action>"],
            )

            # Use llm.generate with the prompt strings
            llm_responses = self.llm.generate(prompts_for_vllm, sampling_params=sampling_params, use_tqdm=False)

            # Extract generated texts (structure of llm_responses should be the same)
            generated_texts = [response.outputs[0].text for response in llm_responses]

            for j, idx in enumerate(indices_to_process):
                rollout = rollouts[idx]
                # Prepend '<thinking>' to the raw generated text
                raw_action = generated_texts[j] if j < len(generated_texts) else ""
                
                # Check if the generated text *already* starts with <thinking>...
                if raw_action.startswith(think_token_str):
                    action = raw_action
                else:
                    action = think_token_str + raw_action # Add it if missing

                next_state, env_reward, done, truncated, _ = rollout["env"].step(action)

                # Calculate PPO-specific formatting reward
                ppo_formatting_reward, _ = self._calculate_ppo_reward_and_validity(action, rollout["env"])
                
                # Combine rewards
                total_step_reward = env_reward + ppo_formatting_reward
                
                # Track this step in the trajectory
                rollout["trajectory"].append({
                    "step": rollout["step_count"] + 1,
                    "action": action,
                    "state": next_state,
                    "reward": total_step_reward # Use combined reward
                })
                
                rollout["history"].append({"role": "assistant", "content": action})
                rollout["history"].append({"role": "user", "content": next_state})
                rollout["reward"] += (self.gamma ** rollout["step_count"]) * total_step_reward # Use combined reward
                rollout["step_count"] += 1
                rollout["done"] = done
                rollout["truncated"] = truncated

        # Print details of each rollout
        console.print("[bold blue]MC Rollout Results:[/bold blue]")
        for i, rollout in enumerate(rollouts):
            status = "Done" if rollout["done"] else "Truncated" if rollout["truncated"] else "Max Steps"
            console.print(f"[bold blue]Rollout {i+1}:[/bold blue] Return: {rollout['reward']:.2f}, Steps: {rollout['step_count']}, Status: {status}")
            
            # Create a table for this rollout's trajectory
            table = Table(title=f"Rollout {i+1} Trajectory", box=box.ROUNDED)
            table.add_column("Step", justify="right", style="cyan")
            table.add_column("Action", style="yellow", no_wrap=False)
            table.add_column("State", style="green", no_wrap=False)
            table.add_column("Reward", justify="right", style="magenta")
            
            for step_data in rollout["trajectory"]:
                table.add_row(
                    str(step_data["step"]),
                    escape(step_data["action"]),  # Escape action
                    escape(step_data["state"]),   # Escape state
                    f"{step_data['reward']:.2f}"
                )
            
            console.print(table)

        returns = [r["reward"] for r in rollouts]
        # use the last rollout's return as the return
        # returns = [r["trajectory"][-1]["reward"] for r in rollouts]
        if not returns:
            avg_return = 0.0
        elif self.mc_top_p == 1.0:
            avg_return = sum(returns) / len(returns)
        else:
            # Determine the number of rollouts to average based on mc_top_p
            num_rollouts_to_average = max(1, int(len(returns) * self.mc_top_p))
            
            # Filter and sort non-zero returns
            non_zero_returns = sorted([r for r in returns if r != 0.0], reverse=True)
            
            top_selected_returns = []
            
            # Take from non_zero_returns first
            num_from_non_zero = min(len(non_zero_returns), num_rollouts_to_average)
            top_selected_returns.extend(non_zero_returns[:num_from_non_zero])
            
            # If not enough, add zeros to meet the num_rollouts_to_average count
            num_zeros_needed = num_rollouts_to_average - len(top_selected_returns)
            if num_zeros_needed > 0:
                top_selected_returns.extend([0.0] * num_zeros_needed)
            
            # Calculate average; len(top_selected_returns) will be num_rollouts_to_average
            # (unless returns was empty, handled by the first 'if' clause)
            if not top_selected_returns: # Should ideally not be hit if returns was not empty
                avg_return = 0.0
            else:
                avg_return = sum(top_selected_returns) / len(top_selected_returns)

        console.print(f"Average MC Return (Top {self.mc_top_p*100:.1f}%): {avg_return:.2f}", style="bold blue")
        return avg_return

    def _normalize_per_episode(self, tensor_to_normalize: torch.Tensor, dones: List[bool]) -> torch.Tensor:
        """Normalizes a tensor per episode based on done flags."""
        normalized_list = []
        start_idx = 0
        device = tensor_to_normalize.device

        if len(tensor_to_normalize) <= 1:
            # Cannot normalize if length is 0 or 1
            return torch.zeros_like(tensor_to_normalize)

        for i in range(len(dones)):
            if dones[i]:
                episode_tensor = tensor_to_normalize[start_idx : i + 1]
                if len(episode_tensor) > 1:
                    mean = episode_tensor.mean()
                    std = episode_tensor.std()
                    # Normalize, avoid division by zero
                    norm_tensor = (episode_tensor - mean) / (std + 1e-8)
                elif len(episode_tensor) == 1:
                    # Tensor for single-step episode is 0 after normalization
                    norm_tensor = torch.zeros_like(episode_tensor)
                else: # Should not happen if buffer logic is correct
                    print(f"Error: Episode tensor length is {len(episode_tensor)} for episode {i}. Returning empty tensor.")
                    norm_tensor = torch.tensor([], device=device)

                normalized_list.append(norm_tensor)
                start_idx = i + 1
        
        # Check for remaining items if the last episode didn't end with done=True
        if start_idx < len(tensor_to_normalize):
            print(f"Warning: Processing remaining items [{start_idx}:] after last 'done' flag during normalization.")
            episode_tensor = tensor_to_normalize[start_idx:]
            if len(episode_tensor) > 1:
                mean = episode_tensor.mean()
                std = episode_tensor.std()
                norm_tensor = (episode_tensor - mean) / (std + 1e-8)
            elif len(episode_tensor) == 1:
                norm_tensor = torch.zeros_like(episode_tensor)
            else:
                norm_tensor = torch.tensor([], device=device)
            normalized_list.append(norm_tensor)

        if normalized_list:
            normalized_tensor = torch.cat(normalized_list)
            return normalized_tensor
        else:
            return torch.tensor([], device=device)

    def _truncate_history(self, chat_history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Truncates chat history to include system prompt and last N messages."""
        if self.history_window_size is None or self.history_window_size <= 0: # Treat 0 as invalid/full history too
            return chat_history # Use full history if window size is not set or invalid

        if not chat_history:
            return []

        system_prompt_message = None
        # Check if the first message is a system prompt
        if chat_history[0].get("role") == "system":
            system_prompt_message = chat_history[0]
            # The actual conversation history (excluding system prompt)
            history_to_consider = chat_history[1:]
        else:
            # No explicit system prompt found at the start
            history_to_consider = chat_history

        num_messages_to_keep = self.history_window_size # N messages

        if len(history_to_consider) <= num_messages_to_keep:
            # History is already short enough or exactly N messages
            return chat_history # Return original (including system prompt if it was there)

        # Keep the last N messages (dictionaries) from the non-system part
        truncated_messages = history_to_consider[-num_messages_to_keep:]

        # Combine system prompt (if any) and the truncated messages
        final_history = []
        if system_prompt_message:
            final_history.append(system_prompt_message)
        final_history.extend(truncated_messages)

        return final_history

    def _compute_action_stats(self, chat_history: List[Dict[str, str]], action: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes per-token log probabilities for policy and ref models, action mask, and logits."""
        device = self._get_device()
        
        prompt = self.processing_class.apply_chat_template(chat_history, tokenize=False)
        chat_with_action = copy.deepcopy(chat_history)
        chat_with_action.append({"role": "assistant", "content": action})
        full_prompt = self.processing_class.apply_chat_template(chat_with_action, tokenize=False)

        # Tokenize and move to device in one step
        prompt_tokens = {k: v.to(device) for k, v in self.processing_class(prompt, return_tensors="pt").items()}
        full_tokens = {k: v.to(device) for k, v in self.processing_class(full_prompt, return_tensors="pt").items()}

        prompt_len = prompt_tokens["input_ids"].size(1)
        full_len = full_tokens["input_ids"].size(1)
        action_len = full_len - prompt_len

        outputs = self.model(input_ids=full_tokens["input_ids"], attention_mask=full_tokens["attention_mask"])
        # Correctly slice logits: prediction for token i+1 is at logit index i
        # We need logits corresponding to the action tokens
        logits = outputs.logits[:, prompt_len - 1 : full_len - 1, :]
        # Apply temperature scaling
        logits = logits / self.generation_temperature
        log_probs = torch.log_softmax(logits, dim=-1)

        # Action tokens are from prompt_len up to full_len
        next_tokens = full_tokens["input_ids"][:, prompt_len:full_len]
        
        # Ensure dimensions match for gather operation
        if next_tokens.size(1) != log_probs.size(1):
            # This should ideally not happen with the corrected slicing. Raise error if it does.
            raise ValueError(f"Mismatch between log_probs dim 1 ({log_probs.size(1)}) and next_tokens dim 1 ({next_tokens.size(1)}).")
            # print(f"WARNING: Mismatch between log_probs dim 1 ({log_probs.size(1)}) and next_tokens dim 1 ({next_tokens.size(1)}). Adjusting next_tokens.")
            # min_len = min(next_tokens.size(1), log_probs.size(1))
            # next_tokens = next_tokens[:, :min_len]
            # log_probs = log_probs[:, :min_len, :] # Also adjust log_probs to be safe

        # Get the log probability of the actual action tokens
        per_token_log_probs = log_probs.gather(2, next_tokens.unsqueeze(-1)).squeeze(-1) # Shape: (batch_size=1, action_len)

        # Compute reference log probability using the reference model
        if self.ref_model is not None:
            with torch.no_grad():
                ref_outputs = self.ref_model(input_ids=full_tokens["input_ids"], attention_mask=full_tokens["attention_mask"])
                ref_logits = ref_outputs.logits[:, prompt_len - 1 : full_len - 1, :]
                # Apply temperature scaling to ref logits
                ref_logits = ref_logits / self.generation_temperature
                if ref_logits.size(1) != next_tokens.size(1):
                    # Raise error if mismatch
                    raise ValueError(f"Mismatch between ref_logits dim 1 ({ref_logits.size(1)}) and next_tokens dim 1 ({next_tokens.size(1)}).")
                    # print(f"WARNING: Mismatch between ref_logits dim 1 ({ref_logits.size(1)}) and next_tokens dim 1 ({next_tokens.size(1)}). Adjusting ref_logits.")
                    # ref_logits = ref_logits[:, :next_tokens.size(1), :]

                ref_log_probs_dist = torch.log_softmax(ref_logits, dim=-1)
                ref_per_token_log_probs = ref_log_probs_dist.gather(2, next_tokens.unsqueeze(-1)).squeeze(-1) # Shape: (batch_size=1, action_len)
        else:
            # If no ref model, use the policy model's log probs detached as reference
            # This means KL will be zero, effectively disabling the KL penalty
            ref_per_token_log_probs = per_token_log_probs.detach()

        # Create the action mask
        action_mask = torch.ones_like(per_token_log_probs, device=device) # Shape: (batch_size=1, action_len)

        # Return per-token log probs, ref log probs, the mask, and policy logits
        # Squeeze the batch dimension (since batch size is 1 here)
        return per_token_log_probs.squeeze(0), ref_per_token_log_probs.squeeze(0), action_mask.squeeze(0), logits.squeeze(0)

    def _create_state_value_table_from_q_table(self):
        """
        Computes a state-value table V(s) by averaging Q(s, a) for each state
        present in the self.q_table.
        """
        if not self.q_table:
            console.print("[yellow]Warning: Q-table is empty or None. Cannot create V-table.[/yellow]")
            self.v_table = {}
            self.average_v_table_value = 0.0 # Set default average
            return

        state_q_values = defaultdict(list)
        for (state_id, _), q_value in self.q_table.items():
            state_q_values[state_id].append(q_value)

        self.v_table = {}
        total_v_value = 0
        num_states = 0
        for state_id, q_values in state_q_values.items():
            if q_values:
                self.v_table[state_id] = sum(q_values) / len(q_values)
                total_v_value += self.v_table[state_id]
                num_states += 1
            else:
                # Should not happen with defaultdict logic, but handle defensively
                self.v_table[state_id] = 0.0 # Or some other default

        # Calculate the average V-value across all states in the table
        self.average_v_table_value = total_v_value / num_states if num_states > 0 else 0.0

        console.print(f"[blue]INFO: Created V-table with {len(self.v_table)} states from Q-table. Average V(s) = {self.average_v_table_value:.3f}[/blue]")

    def compute_loss(
        self, model: nn.Module, inputs: dict, return_outputs: bool = False, num_items_in_batch: Optional[int] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        # Collect rollout episodes.

        # --- Select diverse starting state (cycling) ---
        chosen_start_state_id = None
        if self.possible_start_state_ids:
            if len(self.possible_start_state_ids) == 1:
                chosen_start_state_id = self.possible_start_state_ids[0]
                self.current_start_state_index = 0 # Explicitly set for single state case
            else: # More than one possible start state, cycle through them
                self.current_start_state_index = (self.current_start_state_index + 1) % len(self.possible_start_state_ids)
                chosen_start_state_id = self.possible_start_state_ids[self.current_start_state_index]

            if chosen_start_state_id is not None:
                 console.print(f"[blue]INFO: For this batch, starting episode rollouts from state (cycling index {self.current_start_state_index}): '{chosen_start_state_id}'[/blue]")
            else:
                 # This case should be less likely now with cycling unless possible_start_state_ids is empty
                 console.print(f"[yellow]INFO: No specific starting state selected for this batch. Using environment's default reset.[/yellow]")
        else:
            console.print("[yellow]INFO: No specific starting states configured/found for diversity. Using environment's default reset behavior.[/yellow]")
        # --- End select diverse starting state ---

        env = self.env_factory() # Initialize env once per batch
        # Pass the chosen_start_state_id to reset.
        initial_obs, _ = env.reset(start_state_id=chosen_start_state_id) # Pass chosen_start_state_id
        initial_state_dict = env.get_state() # Capture the exact state

        for _ in range(self.rollout_batch_size):
            # Pass env and the specific initial state dict to each rollout
            self._rollout_episode(env, initial_state_dict)

        self.buffer.compute_returns_and_advantages()
        # Ensure the buffer returns per-token log probabilities
        # We assume batch contains: states, actions, advantages, old_log_probs (per token), action_masks
        batch = self.buffer.get()

        if len(batch.states) == 0:
            # Create a zero loss tensor that requires gradients
            # Sum over a parameter that requires grad, then multiply by zero
            # Find a parameter that requires grad
            grad_param = None
            for param in model.parameters():
                if param.requires_grad:
                    grad_param = param
                    break
            
            if grad_param is not None:
                computed_loss = (grad_param * 0.0).sum() # Creates a zero tensor with grad_fn
                print("WARNING: Buffer was empty after rollouts. Using zero loss with grad_fn.")
            else:
                # Fallback if no parameter requires grad (shouldn't happen in training)
                # Ensure requires_grad=True if we create a tensor directly
                computed_loss = torch.tensor(0.0, device=self._get_device(), requires_grad=True) 
                print("WARNING: Buffer was empty and no parameter requires grad. Using requires_grad=True.")

            metrics = {"loss": 0.0, "policy_loss": 0.0, "kl_loss": 0.0, "avg_reward": 0.0, "entropy": 0.0}
            # Clear buffer and stats even if empty
            self.buffer.clear()
            self.episode_rewards = []
            self.episode_lengths = []

        else:
            device = self._get_device()
            # --- Select Advantage Source (Q-table or GAE/TD) ---
            advantages_source_tensor = None
            if False: # self.use_q_table_value and self.q_table is not None:
                advantages_source_tensor = torch.tensor(batch.values, device=device, dtype=torch.float32)
                advantages_to_use = advantages_source_tensor - advantages_source_tensor.mean()
                # advantages_to_use = self._normalize_per_episode(advantages_source_tensor, batch.dones)
                console.print(f"[blue]INFO: Using Q-table values ({len(advantages_source_tensor)} steps) as advantage source.[/blue]")
            else:
                # Use GAE/TD advantages
                advantages_source_tensor = torch.tensor(batch.advantages, device=device, dtype=torch.float32)
                # batch_values = torch.tensor(batch.values, device=device, dtype=torch.float32)
                # advantages_source_tensor = torch.where(batch_values < 0, torch.zeros_like(advantages_source_tensor), advantages_source_tensor)
                # advantages_to_use = self._normalize_per_episode(advantages_source_tensor, batch.dones)
                if self.buffer.advantage_type == "gae":
                    # advantages_to_use = advantages_source_tensor - advantages_source_tensor.mean()
                    advantages_to_use = self._normalize_per_episode(advantages_source_tensor, batch.dones)
                elif self.buffer.advantage_type == "td":
                    # advantages_to_use = (advantages_source_tensor - advantages_source_tensor.mean()) # / (advantages_source_tensor.std() + 1e-8)
                    # advantages_to_use = advantages_source_tensor - advantages_source_tensor.mean()
                    advantages_to_use = self._normalize_per_episode(advantages_source_tensor, batch.dones)
                console.print(f"[blue]INFO: Using GAE advantages ({len(advantages_source_tensor)} steps) as advantage source.[/blue]")
            # --- End Advantage Selection ---


            # --- Start Per-Token Loss Calculation ---
            all_per_token_losses = []
            all_per_token_kls = []
            all_masks = []
            all_advantages_expanded = []
            all_policy_per_token_losses = []
            all_coef_1s = []  # For clipping ratio metrics
            all_coef_2s = []  # For clipping ratio metrics
            all_per_token_entropies = [] # List to store entropy

            current_advantage_idx = 0
            for i in range(len(batch.states)):
                chat_history = batch.states[i]
                action = batch.actions[i]
                # Ensure old log probs are tensors
                old_per_token_logp = torch.tensor(batch.old_log_probs[i], device=device)
                action_mask = torch.tensor(batch.action_masks[i], device=device)
                action_len = action_mask.sum().int().item()

                # Compute current per-token log probs, reference log probs, and logits
                per_token_logp, ref_per_token_logp, _, per_token_logits = self._compute_action_stats(chat_history, action)

                # Align shapes if necessary (should match if buffer/stats are correct)
                if per_token_logp.shape != old_per_token_logp.shape:
                    raise ValueError(f"Shape mismatch: per_token_logp {per_token_logp.shape} vs old_per_token_logp {old_per_token_logp.shape}")
                if per_token_logp.shape != ref_per_token_logp.shape:
                    raise ValueError(f"Shape mismatch: per_token_logp {per_token_logp.shape} vs ref_per_token_logp {ref_per_token_logp.shape}")
                if per_token_logp.shape != action_mask.shape:
                    raise ValueError(f"Shape mismatch: per_token_logp {per_token_logp.shape} vs action_mask {action_mask.shape}")


                # Get the advantage corresponding to this step (should be a single value per step)
                # Make sure the advantage tensor slicing aligns with the steps
                if current_advantage_idx >= len(advantages_to_use):
                    raise IndexError("Advantage index out of bounds. Check buffer logic and advantage calculation.")
                advantage = advantages_to_use[current_advantage_idx]
                current_advantage_idx += 1 # Move to the next step's advantage

                # Calculate per-token KL divergence
                per_token_kl = torch.zeros_like(per_token_logp)
                if self.beta != 0.0 and self.ref_model is not None:
                    # Ensure no NaNs or Infs from logp differences before exp
                    logp_diff = ref_per_token_logp - per_token_logp
                    exp_term = torch.exp(logp_diff)
                    per_token_kl = exp_term - logp_diff - 1
                
                # --- Calculate per-token entropy --- 
                probs = torch.softmax(per_token_logits, dim=-1)
                log_probs_full_dist = torch.log_softmax(per_token_logits, dim=-1)
                per_token_entropy = -(probs * log_probs_full_dist).sum(dim=-1)
                all_per_token_entropies.append(per_token_entropy) # Store entropy
                # --- End entropy calculation ---

                # Calculate PPO ratio and surrogate losses
                # Ensure old log probs are detached if they came directly from buffer without recomputation
                log_ratio = per_token_logp - old_per_token_logp.detach()
                ratio = torch.exp(log_ratio)
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_eps_low, 1 + self.clip_eps_high)

                # Store coefficients for clipping metrics
                all_coef_1s.append(ratio)
                all_coef_2s.append(clipped_ratio)

                # Expand advantage to match the shape of per-token tensors
                # Advantage is per-step, apply it to all tokens in that step's action
                expanded_advantage = advantage.expand_as(per_token_logp)

                # Determine policy loss type
                policy_loss_type = getattr(self.args, 'policy_loss_type', 'reinforce') # Default to 'reinforce'

                if policy_loss_type == 'reinforce':
                    # REINFORCE loss: -log_prob * Advantage
                    policy_per_token_loss = -per_token_logp * expanded_advantage
                else: # Default to PPO (even if simplified)
                    per_token_loss1 = ratio * expanded_advantage
                    per_token_loss2 = clipped_ratio * expanded_advantage
                    policy_per_token_loss = -torch.min(per_token_loss1, per_token_loss2)


                # Combine Policy loss, KL penalty, and entropy bonus
                combined_per_token_loss = policy_per_token_loss # Start with the chosen policy loss
                if self.beta != 0.0:
                     combined_per_token_loss += self.beta * per_token_kl
                if self.entropy_coeff != 0.0:
                     combined_per_token_loss -= self.entropy_coeff * per_token_entropy # Subtract entropy bonus

                all_per_token_losses.append(combined_per_token_loss)
                all_policy_per_token_losses.append(policy_per_token_loss) # Append chosen policy loss part
                all_per_token_kls.append(per_token_kl)
                all_masks.append(action_mask)
                all_advantages_expanded.append(expanded_advantage) # Store for potential metric calculation

            # Concatenate all tensors from the batch
            if all_per_token_losses:
                loss_type = getattr(self.args, 'loss_type', 'default')

                if loss_type in ['grpo', 'dr_grpo']:
                    # --- Use Padding for Per-Sequence Calculation (GRPO / DR-GRPO) ---
                    # Pad the lists of tensors (losses and masks)
                    padded_losses = pad_sequence(all_per_token_losses, batch_first=True, padding_value=0.0)
                    padded_policy_losses = pad_sequence(all_policy_per_token_losses, batch_first=True, padding_value=0.0)
                    padded_kls = pad_sequence(all_per_token_kls, batch_first=True, padding_value=0.0)
                    padded_masks = pad_sequence(all_masks, batch_first=True, padding_value=0.0)
                    padded_advantages = pad_sequence(all_advantages_expanded, batch_first=True, padding_value=0.0)
                    padded_coef_1s = pad_sequence(all_coef_1s, batch_first=True, padding_value=1.0) # Pad ratios with 1
                    padded_coef_2s = pad_sequence(all_coef_2s, batch_first=True, padding_value=1.0) # Pad ratios with 1
                    padded_entropies = pad_sequence(all_per_token_entropies, batch_first=True, padding_value=0.0)

                    # Calculate number of valid tokens per sequence
                    sequence_token_counts = padded_masks.sum(dim=1).clamp(min=1.0)

                    if loss_type == 'grpo':
                         # Calculate combined loss per sequence sum, then average per sequence, then mean over batch
                        sequence_loss_sums = (padded_losses * padded_masks).sum(dim=1)
                        sequence_averages = sequence_loss_sums / sequence_token_counts
                        computed_loss = sequence_averages.mean()
                    elif loss_type == 'dr_grpo':
                        # Sum over all tokens, divide by total number of sequences * max_completion_length
                        total_loss_sum = (padded_losses * padded_masks).sum()
                        num_sequences = padded_losses.size(0)
                        max_completion_length = getattr(self.args, 'max_completion_length', 100) # Use padded max length? or fixed? Let's stick to arg for now.
                        computed_loss = total_loss_sum / (num_sequences * max_completion_length)


                    # --- Calculate Metrics Per Sequence ---
                    # Policy Loss
                    sequence_policy_loss_sums = (padded_policy_losses * padded_masks).sum(dim=1)
                    mean_policy_loss = (sequence_policy_loss_sums / sequence_token_counts).mean()

                    # KL Loss
                    sequence_kl_sums = (padded_kls * padded_masks).sum(dim=1)
                    mean_kl = (sequence_kl_sums / sequence_token_counts).mean()

                    # Entropy
                    sequence_entropy_sums = (padded_entropies * padded_masks).sum(dim=1)
                    mean_entropy = (sequence_entropy_sums / sequence_token_counts).mean()

                    # Advantage
                    sequence_advantage_sums = (padded_advantages * padded_masks).sum(dim=1)
                    avg_advantage = (sequence_advantage_sums / sequence_token_counts).mean()

                    # Clipping Ratios (Calculated per-token, then averaged per sequence, then averaged over batch)
                    is_low_clipped = (padded_coef_1s < 1 - self.clip_eps_low) & (padded_advantages < 0)
                    is_high_clipped = (padded_coef_1s > 1 + self.clip_eps_high) & (padded_advantages > 0)
                    is_region_clipped = is_low_clipped | is_high_clipped

                    sequence_low_clip_sums = (is_low_clipped.float() * padded_masks).sum(dim=1)
                    sequence_high_clip_sums = (is_high_clipped.float() * padded_masks).sum(dim=1)
                    sequence_region_clip_sums = (is_region_clipped.float() * padded_masks).sum(dim=1)

                    low_clip = (sequence_low_clip_sums / sequence_token_counts).mean()
                    high_clip = (sequence_high_clip_sums / sequence_token_counts).mean()
                    clip_ratio = (sequence_region_clip_sums / sequence_token_counts).mean()

                else: # Default PPO (per-token averaging across batch)
                    # --- Use Concatenation for Default Per-Token Calculation ---
                    cat_per_token_losses = torch.cat(all_per_token_losses)
                    cat_per_token_kls = torch.cat(all_per_token_kls)
                    cat_masks = torch.cat(all_masks)
                    cat_advantages = torch.cat(all_advantages_expanded) # Concatenated expanded advantages
                    cat_coef_1s = torch.cat(all_coef_1s) # Concatenated coef_1 for clipping metrics
                    cat_coef_2s = torch.cat(all_coef_2s) # Concatenated coef_2 for clipping metrics
                    cat_entropies = torch.cat(all_per_token_entropies) # Concatenated entropies
                    cat_policy_per_token_losses = torch.cat(all_policy_per_token_losses) # Concatenate chosen policy losses

                    total_mask_sum = cat_masks.sum().clamp(min=1e-8)

                    # Default loss: average over all valid tokens in the batch
                    computed_loss = (cat_per_token_losses * cat_masks).sum() / total_mask_sum

                    # Compute and log metrics (averaged per token across batch)
                    mean_kl = (cat_per_token_kls * cat_masks).sum() / total_mask_sum
                    mean_policy_loss = (cat_policy_per_token_losses * cat_masks).sum() / total_mask_sum
                    avg_advantage = (cat_advantages * cat_masks).sum() / total_mask_sum
                    mean_entropy = (cat_entropies * cat_masks).sum() / total_mask_sum

                    # Compute clipping metrics (averaged per token across batch)
                    is_low_clipped = (cat_coef_1s < 1 - self.clip_eps_low) & (cat_advantages < 0)
                    is_high_clipped = (cat_coef_1s > 1 + self.clip_eps_high) & (cat_advantages > 0)
                    is_region_clipped = is_low_clipped | is_high_clipped

                    low_clip = (is_low_clipped.float() * cat_masks).sum() / total_mask_sum
                    high_clip = (is_high_clipped.float() * cat_masks).sum() / total_mask_sum
                    clip_ratio = (is_region_clipped.float() * cat_masks).sum() / total_mask_sum


                # --- Common Metrics and Logging (independent of loss type) ---
                # Calculate average value estimate from the buffer
                batch_values = torch.tensor(batch.values, device=device)
                avg_value = batch_values.mean().item() if len(batch_values) > 0 else 0.0

                metrics = {
                    "loss": computed_loss.item(),
                    "policy_loss": mean_policy_loss.item(), # Log the chosen policy part of the loss
                    "kl_loss": mean_kl.item(), # Log average per-token KL (or per-seq avg)
                    "entropy": mean_entropy.item(), # Log average entropy (or per-seq avg)
                    "avg_reward": sum(self.episode_rewards) / len(self.episode_rewards) if self.episode_rewards else 0.0,
                    "avg_episode_length": sum(self.episode_lengths) / len(self.episode_lengths) if self.episode_lengths else 0.0,
                    "avg_advantage": avg_advantage.item(), # Log avg advantage (or per-seq avg)
                    "avg_value": avg_value,
                    "clip_ratio/low": low_clip.item(),
                    "clip_ratio/high": high_clip.item(),
                    "clip_ratio/region": clip_ratio.item(),
                }
            else:
                # Handle case where batch processing resulted in no valid data
                computed_loss = torch.tensor(0.0, device=device, requires_grad=True) # Ensure grad if empty
                metrics = {
                    "loss": 0.0, 
                    "policy_loss": 0.0, 
                    "kl_loss": 0.0, 
                    "entropy": 0.0, 
                    "avg_reward": 0.0,
                    "avg_episode_length": 0.0,
                    "avg_advantage": 0.0,
                    "avg_value": 0.0,
                    "clip_ratio/low": 0.0,
                    "clip_ratio/high": 0.0,
                    "clip_ratio/region": 0.0,
                }

            self.log(metrics)
            self.buffer.clear()
            self.episode_rewards = []
            self.episode_lengths = []

        if return_outputs:
            return computed_loss, metrics
        else:
            return computed_loss

    def _rollout_episode(self, env: Any, initial_state_dict: Dict[str, Any]) -> float: # Accept env and initial state dict
        """
        Generate an episode using the chat template, starting from a specific state.
        """
        # env = self.env_factory() # Remove internal creation
        # init_text, _ = env.reset() # Remove internal reset
        env.set_state(initial_state_dict) # Set the environment to the specific starting state
        init_text = env.current_state_desc # Get the description from the now-set state

        # Extract system prompt if present in the initial text
        system_prompt = ""
        if hasattr(env, 'system_prompt') and env.system_prompt:
            system_prompt = env.system_prompt

        # Create chat history with system prompt if available
        chat_history = []
        if system_prompt:
            chat_history.append({"role": "system", "content": system_prompt})

        # Add the initial user message (environment state)
        chat_history.append({"role": "user", "content": init_text})

        console.print(f"[bold green]Starting new episode[/bold green]")
        console.print(f"[cyan]Initial state:[/cyan] {init_text}")

        done = False
        truncated = False
        episode_reward = 0
        step_count = 0

        # Store episode data temporarily
        episode_data = []

        max_action_retries = 5
        while not done and not truncated and step_count < env.max_steps:
            action = ""
            has_action_tag = False
            for attempt in range(max_action_retries):
                action, has_action_tag = self._generate_action(chat_history)
                escaped_action = escape(action)
                console.print(f"[yellow](Attempt {attempt + 1}/{max_action_retries}) Generated Action:[/yellow] {escaped_action}")
                console.print(f"[{'green' if has_action_tag else 'red'}]Action Tag Found: {has_action_tag}[/{'green' if has_action_tag else 'red'}]")
                
                # is valid action?
                is_valid_action = self._extract_action_tag(action) in env.get_available_actions()  
                console.print(f"[{'green' if is_valid_action else 'red'}]Is Valid Action: {is_valid_action}[/{'green' if is_valid_action else 'red'}]")

                if is_valid_action and has_action_tag:
                    break

            # --- Determine Value Estimate V(s) ---
            estimated_value = 0.0
            state_id = env.current_state_id      
            ppo_reward, is_action_valid = self._calculate_ppo_reward_and_validity(action, env)
            
            if self.use_q_table_value and self.v_table is not None:
                # Use the pre-computed V(s) from the V-table
                # Use the average V-value as fallback if state_id not in table
                estimated_value = self.v_table.get(state_id, self.average_v_table_value)
                console.print(f"[blue]Using V-table value for state {state_id}: {estimated_value:.2f} (PPO Reward: {ppo_reward:.2f}, Valid: {is_action_valid})[/blue]")
            else:
                # Use Monte Carlo simulation to estimate V(s)
                # The raw_estimated_value from _mc_value is used, and then propagated in buffer.
                console.print(f"[blue]Using MC estimation for state {state_id} (PPO Reward: {ppo_reward:.2f}, Valid: {is_action_valid})[/blue]")
                if not is_action_valid:
                    estimated_value = 0.0
                else:
                    estimated_value = self._mc_value(
                        env, 
                        chat_history, 
                        main_episode_current_step=step_count, 
                        main_episode_max_steps=env.max_steps
                    )
            # --- End Value Estimate ---

            # Calculate per-token log probs and mask
            per_token_logp, _, action_mask, _ = self._compute_action_stats(chat_history, action)

            current_chat_history = copy.deepcopy(chat_history) # History *before* adding assistant action
            chat_history.append({"role": "assistant", "content": action})

            next_state_text, reward, done, truncated, info = env.step(action)

            reward += ppo_reward

            if not is_action_valid:
                console.print(f"[bold red]Applied penalty for missing action tag. Reward: {reward}[/bold red]")

            chat_history.append({"role": "user", "content": next_state_text})

            # Store data for this step
            episode_data.append({
                "step_num": step_count + 1,
                "state_history": current_chat_history,
                "action": action,
                "action_token_ids": self._get_action_token_ids(current_chat_history, action),
                "reward": reward,
                "value": estimated_value,
                "log_prob": per_token_logp.detach().cpu().float().numpy(),
                "action_mask": action_mask.detach().cpu().float().numpy(),
                "is_action_valid": is_action_valid,
                "next_state_text": next_state_text,
                "done": done,
                "truncated": truncated,
            })

            episode_reward += reward
            step_count += 1

        # --- Calculate value variance for the episode --- 
        episode_values = [s['value'] for s in episode_data]
        value_variance = 0.0
        if len(episode_values) > 1:
            value_variance = torch.tensor(episode_values).var().item()
        
        # --- Conditionally add collected data to buffer --- 
        # Keep if variance meets threshold OR filtering is disabled (threshold <= 0) OR it's a very short episode
        keep_episode = (value_variance >= self.value_variance_threshold or self.value_variance_threshold <= 0 or len(episode_values) <= 1)

        if keep_episode:
            if self.value_variance_threshold > 0:
                 console.print(f"[green]Keeping episode: Value variance ({value_variance:.4f}) >= threshold ({self.value_variance_threshold}) or episode length <= 1.[/green]")
            else:
                 console.print(f"[green]Keeping episode (Filtering disabled). Variance: {value_variance:.4f}[/green]")
            start_buffer_idx = self.buffer.size
            for i, step_data in enumerate(episode_data):
                is_last_step = (i == len(episode_data) - 1)
                # Mark buffer 'done' if step ended naturally, was truncated, OR it's the last step of the episode
                buffer_done = step_data["done"] or step_data["truncated"] or is_last_step
                self.buffer.add(
                    state=step_data["state_history"],
                    action=step_data["action"],
                    action_token_ids=step_data["action_token_ids"],
                    reward=step_data["reward"],
                    value=step_data["value"],
                    log_prob=step_data["log_prob"],
                    action_mask=step_data["action_mask"],
                    done=buffer_done,
                    is_action_valid=step_data["is_action_valid"]
                )
            # --- End buffer add ---

            # --- Compute GAE and extract episode-specific advantages/returns ---
            # GAE is computed based on per-step values and rewards, remains unchanged
            self.buffer.compute_returns_and_advantages()
            episode_advantages = self.buffer.advantages[start_buffer_idx : start_buffer_idx + len(episode_data)]
            episode_returns = self.buffer.returns[start_buffer_idx : start_buffer_idx + len(episode_data)]
            # --- End GAE ---

            # --- Print the table ---
            table = Table(title="Episode Trajectory (Kept)", box=box.ROUNDED)
            table.add_column("Step", justify="right", style="cyan")
            table.add_column("User (Env)", style="green", no_wrap=False)
            table.add_column("Assistant (Action)", style="yellow", no_wrap=False)
            table.add_column("Reward", justify="right", style="magenta")
            table.add_column("Value (Q/MC/V)", justify="right", style="blue") # Updated label
            table.add_column("Return (GAE/TD)", justify="right", style="purple") # Updated label
            table.add_column("Advantage (GAE/TD)", justify="right", style="orange_red1") # Updated label
            table.add_column("Action Tag", justify="center", style="red")

            for i, step_data in enumerate(episode_data):
                advantage = episode_advantages[i] if i < len(episode_advantages) else float('nan') # Handle potential edge cases
                ret = episode_returns[i] if i < len(episode_returns) else float('nan')
                table.add_row(
                    str(step_data["step_num"]),
                    step_data["next_state_text"],
                    step_data["action"],
                    f"{step_data['reward']:.2f}",
                    f"{step_data['value']:.2f}",
                    f"{ret:.2f}",
                    f"{advantage:.2f}",
                    "✓" if step_data["is_action_valid"] else "✗"
                )

            console.print(table)
            # --- End table print ---
            console.print(Panel(f"Episode finished with reward: {episode_reward:.2f} (Kept)", style="bold green"))
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(step_count)
        else:
            # Episode is discarded due to low variance (only happens if filtering is enabled)
            console.print(f"[yellow]Discarding episode: Value variance ({value_variance:.4f}) < threshold ({self.value_variance_threshold}).[/yellow]")
            # Optional: Still log basic stats for discarded episodes if needed
            # self.log({"discarded_episode_reward": episode_reward, "discarded_episode_length": step_count})

        return episode_reward

    def _calculate_ppo_reward_and_validity(self, action_text: str, env: Any) -> Tuple[float, bool]:
        """
        Calculates PPO-specific reward and action validity based on environment type.
        For TwoTreasuresMaze: expects <thinking>action</thinking>
            - Correct format & valid action: +0.3
            - Incorrect format or invalid action: -0.3
        For Others: expects <thinking>plan</thinking><action>action</action>
            - Thinking bonus: +0.15
            - Valid action in tag bonus: +0.15
            - Invalid action/no tag penalty: -0.3 (overrides bonuses)
        """
        ppo_reward = 0.0
        is_action_valid_and_formatted = False # General flag for successful parsing and valid action

        if self.is_two_treasures_maze:
            # Logic for TwoTreasuresMaze: <thinking>action_content</thinking>
            extracted_content = self._extract_action_tag(action_text) # Will get from <thinking>
            if extracted_content is not None: # Check if <thinking>...</thinking> was found and content extracted
                # Further check if the original action_text truly matches the <thinking>content</thinking> structure
                # This is implicitly handled by _generate_action returning has_tag_and_correct_format True
                # and _extract_action_tag returning non-None from a well-formed tag.
                # The `has_tag_and_correct_format` from _generate_action is the primary format check.
                # Here, we primarily care about the validity of the *extracted_content*.
                available_actions = env.get_available_actions()
                if extracted_content.lower() in available_actions:
                    is_action_valid_and_formatted = True # Action content is valid
                    ppo_reward = 0.1
                else:
                    # Content inside <thinking> is invalid
                    is_action_valid_and_formatted = False
                    ppo_reward = 0
            else:
                # <thinking> tags malformed or missing
                is_action_valid_and_formatted = False
                ppo_reward = 0
        else:
            # Original PPO Logic for other environments
            thinking_present = "<thinking>" in action_text.lower() # Check for <thinking> anywhere for the plan
            extracted_action_content = self._extract_action_tag(action_text) # Will get from <action>
            action_tag_exists_and_content_valid = False

            if thinking_present:
                ppo_reward += 0.15 # Thinking bonus

            if extracted_action_content is not None: # <action>...</action> found with content
                available_actions = env.get_available_actions()
                if extracted_action_content.lower() in available_actions:
                    action_tag_exists_and_content_valid = True
                    ppo_reward += 0.15 # Valid action in tag bonus
                # If content in <action> is not valid, no bonus, penalty might apply below.
            
            # Determine overall validity for this mode
            is_action_valid_and_formatted = thinking_present and action_tag_exists_and_content_valid
            
            if not is_action_valid_and_formatted:
                # Apply penalty if not fully valid (e.g., missing thinking, missing action tag, or invalid action content)
                # This penalty overrides previous bonuses.
                ppo_reward = -0.1 
        
        return ppo_reward, is_action_valid_and_formatted

    def train(self, num_ppo_updates: int = 100):
        """
        Custom training loop to perform multiple PPO updates with gradient accumulation.
        `num_ppo_updates` here refers to the number of *optimizer* steps.
        """
        self.model.train()
        gradient_accumulation_steps = getattr(self.args, 'gradient_accumulation_steps', 1)
        print(f"INFO: Using gradient accumulation with {gradient_accumulation_steps} steps.")

        # num_ppo_updates now represents the number of OPTIMIZER steps
        total_optimizer_steps = num_ppo_updates
        total_accumulation_steps = total_optimizer_steps * gradient_accumulation_steps

        print(f"INFO: Total optimizer steps: {total_optimizer_steps}")
        print(f"INFO: Total accumulation steps: {total_accumulation_steps}")

        for step in range(total_accumulation_steps):
            # compute_loss already handles rollouts and buffer management
            loss = self.compute_loss(self.model, {}, return_outputs=False)

            # Scale loss by accumulation steps
            # Avoid division by zero if gradient_accumulation_steps is somehow 0 or less
            loss = loss / max(gradient_accumulation_steps, 1)

            self.accelerator.backward(loss)

            # Log loss for the current accumulation step
            # The logged value represents the loss for this micro-batch, scaled for accumulation
            self.log({"train/accumulation_step_loss": loss.item()})
            # print(f"INFO: Accumulation step {step + 1}/{total_accumulation_steps}, Scaled Loss: {loss.item():.4f}") # Can be verbose

            # Perform optimizer step, zero grad, and scheduler step only after accumulating grads
            if (step + 1) % gradient_accumulation_steps == 0:
                effective_step = (step + 1) // gradient_accumulation_steps

                # --- Calculate and log gradient norm ---
                grad_norm = 0.0
                if self.accelerator.sync_gradients:
                    # Clip gradients before checking the norm
                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm if hasattr(self.args, 'max_grad_norm') else 1.0)

                # Log the norm returned by clip_grad_norm_
                if grad_norm is not None:
                    self.log({"train/grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm})


                self.optimizer.step()
                self.optimizer.zero_grad()
                self._update_ema_model()

                # Step the learning rate scheduler if it exists
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                    # Ensure we get a single LR value, handle potential list/dict cases
                    lr_list = self.lr_scheduler.get_last_lr()
                    current_lr = lr_list[0] if lr_list else 0.0
                    self.log({"train/learning_rate": current_lr})
                    # print(f"INFO: Stepped LR scheduler. Current LR: {current_lr:.1e}") # Can be verbose

                # --- Update Reference Model --- 
                if (self.use_ref_model and 
                        self.ref_model is not None and 
                        self.ref_model_update_steps > 0 and 
                        effective_step % self.ref_model_update_steps == 0):
                    # Update ref_model with the current model's state dict
                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    self.ref_model.load_state_dict(unwrapped_model.state_dict())
                    # Keep the ref_model in eval mode and grads off
                    self.ref_model.eval()
                    for param in self.ref_model.parameters():
                        param.requires_grad = False
                    console.print(f"[blue]Updated reference model weights at effective step {effective_step}[/blue]")
                # --- End Reference Model Update ---

                self.state.global_step += 1

                # Update vLLM (will now load EMA weights if configured for MC value)
                self._update_vllm_model()

                print(f"INFO: Optimizer Step {effective_step}/{total_optimizer_steps} completed. Grad Norm: {f'{grad_norm:.4f}' if grad_norm is not None else 'N/A'}") # Also print to console

                # save full model using accelerator
                # Adjust save frequency based on effective steps
                save_frequency = getattr(self.args, 'save_steps', 10)
                if self.accelerator.is_main_process and effective_step % save_frequency == 0:
                    save_path = os.path.join(self.args.output_dir, f"Step_{effective_step}")
                    self.accelerator.save_state(save_path)
                    self.save_model(save_path)
                    console.print(f"[green]Saved accelerator state at effective step {effective_step} to {save_path}[/green]")

        console.print("[bold green]Training complete.[/bold green]")

    def _generate_action(self, chat_history: List[Dict[str, str]]) -> Tuple[str, bool]:
        """Generate an action and return whether the action tag was found"""
        truncated_history = self._truncate_history(chat_history)
        base_prompt_str = self.processing_class.apply_chat_template(truncated_history, tokenize=False)

        think_token_str = "<thinking>" # Common start for both formats from LLM POV
        prompt_for_llm = base_prompt_str + think_token_str

        stop_tokens_for_criteria: List[str]
        if self.is_two_treasures_maze:
            stop_tokens_for_criteria = ["</thinking>", "</Thinking>"]
        else:
            stop_tokens_for_criteria = ["</action>", "</Action>"]

        inputs = self.processing_class(prompt_for_llm, return_tensors="pt")
        device = self._get_device()
        inputs = {k: v.long().to(device) for k, v in inputs.items()}
        
        stopping_criteria_objects = [StopOnTag(self.processing_class, stop_token=st) for st in stop_tokens_for_criteria]
        stopping_criteria_list = StoppingCriteriaList(stopping_criteria_objects)

        print(f"INFO: Inputs device: {inputs['input_ids'].device}")
        print(f"INFO: Model device: {self.model.device}")

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=100,
                pad_token_id=self.processing_class.eos_token_id,
                do_sample=True,
                temperature=self.generation_temperature,
                top_p=0.95,
                stopping_criteria=stopping_criteria_list
            )

        prompt_length = inputs["input_ids"].shape[1]
        # model_output_text is what the model generated *after* the initial prompt_for_llm (which ended with "<thinking>")
        model_output_text = self.processing_class.decode(
            outputs[0][prompt_length:],
            skip_special_tokens=True
        )

        cleaned_model_output = re.sub(r'(?i)\b(assistant|system|user)\b\s*', '', model_output_text).strip()

        final_action_text: str
        has_tag_and_correct_format: bool

        if self.is_two_treasures_maze:
            # Expects: <thinking>action_content</thinking>
            # LLM was prompted with "<thinking>", it should generate "action_content</thinking>"
            # Ensure the closing tag is present if stopping criteria found it
            closed_cleaned_output = cleaned_model_output
            if not any(closed_cleaned_output.strip().endswith(st) for st in stop_tokens_for_criteria) and \
               any(sc.found_stop_token for sc in stopping_criteria_objects):
                # Append the first stop token that was found
                for i, sc in enumerate(stopping_criteria_objects):
                    if sc.found_stop_token:
                        closed_cleaned_output += stop_tokens_for_criteria[i]
                        break
            
            final_action_text = think_token_str + closed_cleaned_output # Reconstruct: <thinking>action_content</thinking>
            match = re.search(r"^<thinking>(.*)</thinking>$", final_action_text, re.DOTALL | re.IGNORECASE)
            has_tag_and_correct_format = bool(match)
        else:
            # Expects: <thinking>plan</thinking><action>actual_action</action>
            # LLM was prompted with "<thinking>", it should generate "plan</thinking><action>actual_action</action>"
            final_action_text = think_token_str + cleaned_model_output # This is the full <thinking>...</action> part
            # Check for <action>...</action> within the final_action_text
            match = re.search(r"<action>(.*?)</action>", final_action_text, re.DOTALL | re.IGNORECASE)
            has_tag_and_correct_format = bool(match) # For this mode, "correct format" means the action tag is present
        
        return final_action_text, has_tag_and_correct_format

    def _get_action_token_ids(self, chat_history: List[Dict[str, str]], action: str) -> List[int]:
        before_tokens = self.processing_class.apply_chat_template(chat_history, tokenize=True, return_tensors="pt").to(self._get_device())
        
        chat_with_action = copy.deepcopy(chat_history)
        chat_with_action.append({"role": "assistant", "content": action})
        after_tokens = self.processing_class.apply_chat_template(chat_with_action, tokenize=True, return_tensors="pt").to(self._get_device())

        before_len = before_tokens.size(1)
        action_token_ids = after_tokens[0, before_len:].cpu().tolist()

        return action_token_ids

    def _update_ema_model(self):
        """Updates the EMA state dict weights, performing the calculation on CPU."""
        # Check if EMA is enabled and the state dict exists
        if not self.use_ema_for_mc_value or self.ema_state_dict is None or self.ema_decay <= 0:
            return

        with torch.no_grad():
            # Get the main model's state dict (potentially on GPU)
            model_state_dict = self.accelerator.get_state_dict(self.model)
            # The EMA state dict (self.ema_state_dict) is already on CPU

            new_ema_state_dict = {} # Create a new dict to store updated values
            for name, param in model_state_dict.items():
                if name in self.ema_state_dict:
                    ema_param_cpu = self.ema_state_dict[name] # Already on CPU

                    # --- Perform update on CPU ---
                    # 1. Move the main model's parameter to CPU
                    param_cpu = param.cpu()

                    # 2. Perform EMA update (lerp) on the CPU
                    # Ensure dtypes match for lerp, using ema_param's dtype as reference
                    updated_param_cpu = ema_param_cpu.lerp(param_cpu.to(ema_param_cpu.dtype), 1 - self.ema_decay)

                    # 3. Store the result (already on CPU)
                    new_ema_state_dict[name] = updated_param_cpu
                    # --- End update ---

                else:
                    # If a parameter exists in the main model but not EMA dict, copy it to CPU.
                    print(f"Warning: Parameter {name} found in main model but not in EMA state dict. Copying.")
                    new_ema_state_dict[name] = param.cpu().clone().detach()

            # Replace the old EMA state dict with the new one
            self.ema_state_dict = new_ema_state_dict
        # print("DEBUG: Updated EMA state dict (on CPU).") # Optional debug message
