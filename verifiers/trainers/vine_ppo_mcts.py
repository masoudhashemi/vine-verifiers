from contextlib import contextmanager
import copy
import os
import numpy as np
import torch
import logging # Added
from torch import nn
from transformers import Trainer, StoppingCriteria, StoppingCriteriaList, AutoModelForCausalLM
from typing import List, Tuple, Any, Optional, Union, Dict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.markup import escape
from torch.utils.data import DataLoader, Dataset

# Assuming vllm might be used externally, keep import commented if not direct dependency
# from vllm import LLM, SamplingParams
from vllm import SamplingParams # Keep if SamplingParams is used directly

from torch.optim import AdamW
from accelerate import Accelerator
from accelerate.utils import is_deepspeed_available
from verifiers.utils.training_utils import setup_scheduler
# Import the new buffer class
from .vine_buffer_mcts import VineBuffer, VineBufferSample


import re
from collections import defaultdict

if is_deepspeed_available():
    import deepspeed
    DeepSpeedEngine = deepspeed.DeepSpeedEngine
else:
    DeepSpeedEngine = None

console = Console()
logger = logging.getLogger(__name__) # Added


class StopOnActionTag(StoppingCriteria):
    def __init__(self, tokenizer, stop_token="</action>"):
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

class VinePPOMCTSTrainer(Trainer): # Renamed class
    """
    Custom Trainer implementing VinePPO with MCTS-like value updates.
    It integrates:
    - Persistent V-table updated via MC rollouts.
    - All generated steps (main + MC) used for PPO update.
    - Accelerate for device management.
    - vLLM for batched Monte Carlo rollouts.
    """
    def __init__(
        self,
        env_factory,
        buffer: VineBuffer, # Use the correct buffer type hint
        processing_class,
        mc_rollouts: int = 5,
        mc_max_steps: int = 25,
        mc_top_p: float = 0.2, # Top-p for selecting returns (doesn't affect V-table update)
        rollout_batch_size: int = 2,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        clip_eps_low: Optional[float] = None, # New
        clip_eps_high: Optional[float] = None, # New
        beta: float = 0.1,
        generation_temperature: float = 0.7, 
        q_table: Optional[Dict[Tuple[str, str], float]] = None, 
        # vllm_llm=None, # Removed
        block_size: int = 2048,
        use_ref_model: bool = True,
        ref_model_update_steps: int = 0, 
        history_window_size: Optional[int] = None, 
        entropy_coeff: float = 0.01, 
        loss_type: str = "default", 
        # Added VLLM Client parameters
        llm_instance=None, # Added llm_instance back
        use_vllm_server: bool = False,
        vllm_host: str = "0.0.0.0",
        vllm_server_port: int = 8000,
        vllm_group_port: int = 51216,
        vllm_connection_timeout: float = 60.0,
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
        self.clip_eps_low = clip_eps_low if clip_eps_low is not None else clip_eps
        self.clip_eps_high = clip_eps_high if clip_eps_high is not None else clip_eps
        self.beta = beta
        self.generation_temperature = generation_temperature
        self.q_table = q_table
        self.use_ref_model = use_ref_model
        self.ref_model_update_steps = ref_model_update_steps
        self.history_window_size = history_window_size
        self.entropy_coeff = entropy_coeff
        self.loss_type = loss_type

        # Store VLLM client parameters
        self.use_vllm_server = use_vllm_server
        self.vllm_host = vllm_host
        self.vllm_server_port = vllm_server_port
        self.vllm_group_port = vllm_group_port
        self.vllm_connection_timeout = vllm_connection_timeout
        
        # Initialize V-table and visit counts for MCTS-like updates
        self.v_table = defaultdict(float) 
        self.visit_counts = defaultdict(int) 

        self.accelerator = Accelerator(device_placement=False)

        # Initialize self.llm and self.vllm_client based on use_vllm_server and llm_instance
        self.vllm_client = None
        self.llm = None # This is for the local vLLM instance (renamed from self.vllm_llm)

        if self.use_vllm_server:
            from verifiers.utils.vllm_client import VLLMClient # Ensure import is here
            logger.info(f"[{self.__class__.__name__}] Initializing VLLMClient for server at {self.vllm_host}:{self.vllm_server_port}")
            from verifiers.utils.vllm_client import VLLMClient # Ensure import is here
            logger.info(f"[{self.__class__.__name__}] Initializing VLLMClient for server at {self.vllm_host}:{self.vllm_server_port}")
            
            model_ref_name = "trainer_model_ref_mcts" # Placeholder for VLLMClient
            # self.model is not fully initialized by super().__init__ yet, so cannot access self.model.config here.
            # Using a fixed placeholder is safer.
            
            self.vllm_client = VLLMClient(
                model=model_ref_name, 
                host=self.vllm_host,
                server_port=self.vllm_server_port,
                group_port=self.vllm_group_port,
                connection_timeout=self.vllm_connection_timeout
            )
            
            # Guard the init_communicator call
            if self.accelerator is None or \
               self.accelerator.state.distributed_type == 'NO' or \
               self.accelerator.is_main_process:
                try:
                    logger.info(f"[{self.__class__.__name__}] Main process (or single process) attempting to initialize VLLMClient communicator.")
                    self.vllm_client.init_communicator()
                    logger.info(f"[{self.__class__.__name__}] VLLMClient communicator initialized by main process (or single process).")
                except Exception as e:
                    logger.error(f"[{self.__class__.__name__}] Error initializing VLLMClient communicator for main process: {e}", exc_info=True)
                    # Depending on desired behavior, might re-raise or mark client as unusable for comms
            else:
                logger.info(f"[{self.__class__.__name__}] Non-main process (local rank {self.accelerator.process_index}) will not initialize VLLMClient communicator for weight sync. Client can still be used for API calls like generate.")

            if llm_instance is not None: # This warning remains the same
                logger.warning(f"[{self.__class__.__name__}] 'llm_instance' was provided but 'use_vllm_server' is True. The local 'llm_instance' will be ignored.")
        else: # Not using VLLM server, try to use local llm_instance
            if llm_instance is not None:
                self.llm = llm_instance
                logger.info(f"[{self.__class__.__name__}] Using provided local vLLM instance (self.llm).")
            else:
                logger.warning(f"[{self.__class__.__name__}] 'use_vllm_server' is False and no 'llm_instance' provided. Monte Carlo rollouts will not function.")


        if not hasattr(self.args, 'include_num_input_tokens_seen'):
            self.args.include_num_input_tokens_seen = False
        if not hasattr(self.args, 'block_size'):
            self.args.block_size = block_size
        # Add loss_type to args if not present
        if not hasattr(self.args, 'loss_type'):
            self.args.loss_type = self.loss_type
        # Add max_completion_length to args if not present (needed for dr_grpo)
        # Default to a reasonable value if not provided through TrainingArguments
        if not hasattr(self.args, 'max_completion_length'):
            self.args.max_completion_length = 100 # Default value for dr_grpo loss type


        # Store the initial state dict *before* preparing the main model
        initial_state_dict = None
        model_name_or_path = None # Store model name/path
        model_dtype = None # Store model dtype
        if self.use_ref_model:
            model_name_or_path = self.model.config._name_or_path
            model_dtype = self.model.dtype
            initial_state_dict = copy.deepcopy(self.model.state_dict())
            print("INFO: Saved initial model state dict for reference model.")

        # Prepare the main model and optimizer using Accelerator
        if not hasattr(self, 'optimizer') or self.optimizer is None:
            lr = self.args.learning_rate if hasattr(self.args, 'learning_rate') and self.args.learning_rate is not None else 1e-6
            optimizer_grouped_parameters = [
                {"params": [p for n, p in self.model.named_parameters() if p.requires_grad]}
            ]
            # Add weight decay like in VinePPOTrainer
            self.optimizer = AdamW(optimizer_grouped_parameters, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
            console.print(f"[yellow]Warning: No optimizer found or provided, created default AdamW optimizer with lr={lr}[/yellow]")
        else:
            self.optimizer.param_groups = [
                {"params": [p for n, p in self.model.named_parameters() if p.requires_grad], **group}
                for group in self.optimizer.param_groups
            ]

        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        print(f"INFO: Prepared main model device: {self.accelerator.device}")

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
                self.ref_model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    torch_dtype=model_dtype
                )
                self.ref_model.load_state_dict(initial_state_dict)
                self.ref_model.eval()
                for param in self.ref_model.parameters():
                    param.requires_grad = False

                self.ref_model.to(self.accelerator.device)
                # self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True) # Optional prepare
                print(f"INFO: Prepared reference model device: {self.accelerator.device}")
            except Exception as e:
                console.print(f"[bold red]Error creating or preparing reference model: {e}[/bold red]")
                console.print("[yellow]Warning: Proceeding without reference model due to error.[/yellow]")
                self.ref_model = None
                self.use_ref_model = False
        elif self.use_ref_model:
            console.print("[yellow]Warning: Could not create reference model (missing initial state or name). Disabling ref model usage.[/yellow]")
            self.use_ref_model = False

        if not self.use_ref_model:
            self.ref_model = None
            print("INFO: Reference model not used or creation failed.")

        # Initialize V-table from Q-table if provided
        if self.q_table is not None:
            self._initialize_v_table_from_q_table()
        else:
            print("[blue]INFO: No Q-table provided, V-table initialized empty.[/blue]")

        # Statistics tracking.
        self.episode_rewards = []
        self.episode_lengths = []
        self.buffer.device = self._get_device()

    def _get_device(self):
        return self.accelerator.device

    def _extract_action_tag(self, text: str) -> Optional[str]:
        """Extracts content within <action>...</action> tags."""
        match = re.search(r"<action>(.*?)</action>", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def get_train_dataloader(self) -> DataLoader:
        dataset = DummyDataset()
        dataloader = DataLoader(dataset, batch_size=1)
        return self.accelerator.prepare(dataloader)

    def _update_vllm_model(self):
        """Synchronize vLLM's model weights with the current training model."""
        if not self.accelerator.is_main_process:
            return

        if self.use_vllm_server and self.vllm_client:
            try:
                # self.model is the main policy model, already prepared by accelerator
                self.vllm_client.update_model_params(self.model)
                logger.info(f"[{self.__class__.__name__}] Successfully updated VLLM server weights with Main policy model.")
            except Exception as e:
                logger.error(f"[{self.__class__.__name__}] Error updating VLLM server weights: {e}", exc_info=True)
        elif not self.use_vllm_server and self.llm and self.accelerator.is_main_process: # Changed self.vllm_llm to self.llm
            # Existing logic for local vLLM instance
            try:
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                state_dict = unwrapped_model.state_dict()
                llm_model_instance = self.llm.llm_engine.model_executor.driver_worker.model_runner.model # Changed self.vllm_llm to self.llm
                llm_model_instance.load_weights(list(state_dict.items())) 
                logger.info(f"[{self.__class__.__name__}] Updated local vLLM (self.llm) weights with Main policy model.")
            except Exception as e:
                logger.error(f"[{self.__class__.__name__}] Error updating local vLLM (self.llm) weights: {e}", exc_info=True)
        else:
            if self.accelerator.is_main_process: # Log only on main process if no usable vLLM backend
                 logger.debug("MCTS: No vLLM backend (server or local) configured for model update in _update_vllm_model.")


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

    def _mc_value(self, env: Any, chat_history: List[Dict[str, str]], has_action_tag: bool = True) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Perform MC rollouts, update V(s) for all visited states in MC rollouts,
        and return the updated V(s) and the trajectory data from rollouts.
        Uses history truncation and generation temperature.
        If has_action_tag is False, return a negative value based on remaining steps.
        """
        # Determine vLLM backend
        active_vllm_backend_available = (self.use_vllm_server and self.vllm_client) or \
                                        (not self.use_vllm_server and self.llm) # Changed self.vllm_llm to self.llm

        if not active_vllm_backend_available:
            logger.error(f"[{self.__class__.__name__}] _mc_value called but neither VLLMClient nor local vLLM (self.llm) is available.")
            return 0.0, []
        
        try:
            root_state_id = env.current_state_id
        except AttributeError:
            console.print("[bold red]Error: Environment does not have 'current_state_id'. Cannot perform MCTS-like updates. Returning 0 value.[/bold red]")
            return 0.0, []

        # If the action doesn't have proper tags, return a negative value and empty data
        if not has_action_tag:
            NEGATIVE_REWARD = -0.1
            remaining_steps = env.max_steps - env.step_count
            negative_value = NEGATIVE_REWARD * (self.gamma ** remaining_steps)
            console.print(f"[bold red]Missing action tag: Assigning negative value {negative_value:.2f}[/bold red]")
            # Also update V-table with this negative value? Or just return it?
            # Let's just return it and not update V-table in this case.
            return negative_value, []

        rollouts = []
        all_rollout_steps_data = [] # List to store data from all steps across all rollouts
        for _ in range(self.mc_rollouts):
            env_copy = env.clone() if hasattr(env, "clone") and callable(env.clone) else copy.deepcopy(env)
            env_copy.max_steps = self.mc_max_steps
            rollouts.append({
                "env": env_copy,
                "history": copy.deepcopy(chat_history), # Start with the initial history
                "done": False,
                "truncated": False,
                "rewards": [], # Store sequence of rewards for backprop
                "step_count": 0,
                "visited_state_ids": [] # Track visited states for MCTS update
            })

        while any(not (r["done"] or r["truncated"] or r["step_count"] >= r["env"].max_steps) for r in rollouts):
            active_indices = [i for i, r in enumerate(rollouts) if not (r["done"] or r["truncated"] or r["step_count"] >= r["env"].max_steps)]
            if not active_indices:
                break

            # Truncate history for each active rollout before passing to vLLM
            histories_to_process = [self._truncate_history(rollouts[i]["history"]) for i in active_indices]

            # Check token count and prepare prompts after truncation
            messages_to_process_vllm = []
            indices_to_process_vllm = []
            for idx, truncated_history in zip(active_indices, histories_to_process):
                try:
                    # Apply template to truncated history to check length
                    tokenized = self.processing_class.apply_chat_template(truncated_history, tokenize=True, return_tensors="pt")
                    if tokenized.size(1) >= self.args.block_size:
                        rollouts[idx]["truncated"] = True
                        console.print(f"[bold red]Rollout {idx} truncated: Truncated history exceeds block size ({tokenized.size(1)} tokens)[/bold red]")
                    else:
                        # Prepare prompt for vLLM: apply template + <thinking>
                        base_prompt_str = self.processing_class.apply_chat_template(truncated_history, tokenize=False)
                        think_token_str = "<thinking>"
                        prompt_with_thinking = base_prompt_str + think_token_str
                        messages_to_process_vllm.append(prompt_with_thinking)
                        indices_to_process_vllm.append(idx)
                except Exception as e:
                    console.print(f"[bold red]Error tokenizing truncated history for rollout {idx}: {e}. Marking as truncated.[/bold red]")
                    rollouts[idx]["truncated"] = True

            if not indices_to_process_vllm:
                break

            sampling_params = SamplingParams(
                n=1,
                temperature=self.generation_temperature, # Use configured temperature
                top_p=0.9, # Add top_p like in VinePPOTrainer
                max_tokens=200,
                include_stop_str_in_output=True,
                stop=["</action>", "</Action>"],
            )

            # Use llm.generate with the pre-formatted prompt strings
            generated_texts = []
            if self.use_vllm_server and self.vllm_client:
                logger.debug(f"MCTS: Using VLLMClient for MC generation. Prompts: {len(messages_to_process_vllm)}")
                client_sampling_params = {
                    "n": sampling_params.n,
                    "temperature": sampling_params.temperature,
                    "top_p": sampling_params.top_p,
                    "top_k": sampling_params.top_k if hasattr(sampling_params, 'top_k') else -1, # vLLM uses top_k
                    "max_tokens": sampling_params.max_tokens,
                    "repetition_penalty": sampling_params.repetition_penalty if hasattr(sampling_params, 'repetition_penalty') else 1.0,
                    "stop": sampling_params.stop if hasattr(sampling_params, 'stop') else None,
                    "include_stop_str_in_output": sampling_params.include_stop_str_in_output if hasattr(sampling_params, 'include_stop_str_in_output') else False,
                }
                try:
                    completion_token_ids_list = self.vllm_client.generate(
                        prompts=messages_to_process_vllm, # This is a list of strings
                        **client_sampling_params
                    )
                    # Assuming completion_token_ids_list is List[List[int]]
                    if isinstance(completion_token_ids_list, list) and \
                       all(isinstance(ids, list) for ids in completion_token_ids_list) and \
                       all(isinstance(token_id, int) for ids in completion_token_ids_list for token_id in ids):
                        for token_ids in completion_token_ids_list:
                            generated_texts.append(self.processing_class.decode(token_ids))
                    else: # Fallback parsing if client returned dicts (less likely with direct VLLMClient)
                        logger.warning(f"MCTS: VLLMClient.generate returned unexpected format. Expected List[List[int]], got {type(completion_token_ids_list)}. Attempting fallback parse.")
                        if isinstance(completion_token_ids_list, list) and all(isinstance(item, dict) for item in completion_token_ids_list):
                             for item_dict in completion_token_ids_list:
                                 if "text" in item_dict and isinstance(item_dict["text"], str):
                                     generated_texts.append(item_dict["text"])
                                 elif "text" in item_dict and isinstance(item_dict["text"], list) and item_dict["text"] and isinstance(item_dict["text"][0], str):
                                     generated_texts.append(item_dict["text"][0])
                                 else: generated_texts.append("") # Fallback for dict item
                        else: # Final fallback if parsing fails
                            logger.error(f"MCTS: Could not parse VLLMClient response. Content: {str(completion_token_ids_list)[:200]}")
                            generated_texts = [""] * len(messages_to_process_vllm)
                except Exception as e:
                    logger.error(f"[{self.__class__.__name__}] Error during VLLMClient generation: {e}", exc_info=True)
                    generated_texts = [""] * len(messages_to_process_vllm) # Fallback on error

            elif not self.use_vllm_server and self.llm: # Changed self.vllm_llm to self.llm
                logger.debug(f"[{self.__class__.__name__}] Using local LLM (self.llm) for MC generation. Prompts: {len(messages_to_process_vllm)}")
                llm_responses = self.llm.generate(messages_to_process_vllm, sampling_params=sampling_params, use_tqdm=False) # Changed self.vllm_llm to self.llm
                generated_texts = [response.outputs[0].text for response in llm_responses]
            # No 'else' needed here because active_vllm_backend_available check at start of function handles it.


            for j, idx in enumerate(indices_to_process_vllm):
                rollout = rollouts[idx]
                # The generated text should ideally contain <thinking>...</action>
                raw_action = generated_texts[j] if j < len(generated_texts) else ""

                # Ensure action starts with <thinking> (it should if generated correctly)
                think_token_str = "<thinking>"
                if raw_action.startswith(think_token_str):
                    action = raw_action
                else:
                    # Prepend if missing (might indicate generation issue)
                    action = think_token_str + raw_action

                # Use the history that was actually passed to vLLM (truncated one) for stats
                current_mc_step_history_for_stats = self._truncate_history(rollout["history"])

                # Get state ID *before* stepping, for MCTS update path
                try:
                    pre_step_state_id = rollout["env"].current_state_id
                    if pre_step_state_id is not None:
                        rollout["visited_state_ids"].append(pre_step_state_id)
                except AttributeError:
                    pass

                # Compute log probs and mask for the *generated action* using the *policy model*
                log_prob_np = np.array([])
                action_mask_np = np.array([])
                logits_np = np.array([]) # To store logits for entropy
                stats_computed_successfully = False
                try:
                    per_token_logp, ref_per_token_logp, action_mask, per_token_logits = self._compute_action_stats(
                        current_mc_step_history_for_stats, action, model_to_use=self.model
                    )

                    if self.ref_model is not None:
                        log_prob_np = ref_per_token_logp.detach().cpu().float().numpy()
                    else:
                        log_prob_np = per_token_logp.detach().cpu().float().numpy() # Fallback

                    action_mask_np = action_mask.detach().cpu().float().numpy()
                    logits_np = per_token_logits.detach().cpu().float().numpy() # Store logits
                    stats_computed_successfully = True
                except Exception as e:
                    console.print(f"[bold red]Error computing stats for MC action (Rollout {idx}, Step {rollout['step_count'] + 1}): {e}. Skipping step addition.[/bold red]")

                if not stats_computed_successfully:
                    rollout["truncated"] = True
                    console.print(f"[yellow]Marking Rollout {idx} as truncated due to stat computation error.[/yellow]")
                    continue

                # Step the environment
                next_state, reward, done, truncated, _ = rollout["env"].step(action)
                reward = reward if reward > 0 or reward == -1.0 else 0.0 # Existing reward logic
                rollout["rewards"].append(reward) # Store reward for this step

                # Add assistant action and user response to the *full* history for the *next* step
                rollout["history"].append({"role": "assistant", "content": action})
                rollout["history"].append({"role": "user", "content": next_state})

                # Get the state ID for the *intermediate* state reached in the rollout
                try:
                    mc_step_state_id = rollout["env"].current_state_id
                except AttributeError:
                    mc_step_state_id = None # Cannot get V-table value if no ID

                # Look up value from the *current* v_table for the intermediate state
                mc_step_value = self.v_table[mc_step_state_id] if mc_step_state_id is not None else 0.0

                # Store this step's data
                step_data = {
                    "state_history": current_mc_step_history_for_stats, # Store history used for stats
                    "action": action,
                    "reward": reward,
                    "value": mc_step_value,
                    "log_prob": log_prob_np,
                    "action_mask": action_mask_np,
                    "logits": logits_np, # Store logits for potential later use (entropy)
                    "done": done,
                    "truncated": truncated,
                    "mc_rollout_id": idx,
                    "mc_step_count": rollout["step_count"] + 1,
                    "env_state_id": mc_step_state_id # Store the state ID reached in this step
                }
                all_rollout_steps_data.append(step_data)

                # Update rollout state
                rollout["step_count"] += 1
                rollout["done"] = done
                rollout["truncated"] = truncated

        # --- MCTS Backpropagation for ALL Visited States ---
        total_return_for_log = 0.0 # Keep track for logging average
        num_valid_rollouts = 0
        updated_states_in_batch = set() # Track unique states updated in this call

        # Store state_ids and their final V-table values for updating MC step data
        updated_state_values = {}

        for i, rollout in enumerate(rollouts):
            # Calculate the total discounted return for logging/selection purposes if needed
            rollout_return_for_log = sum([(self.gamma ** t) * r for t, r in enumerate(rollout["rewards"])])
            total_return_for_log += rollout_return_for_log
            num_valid_rollouts += 1

            # Get unique states visited in this rollout
            visited_ids_in_order = rollout["visited_state_ids"]
            rollout_rewards = rollout["rewards"]

            # Update V(s) for every state visited in this rollout using the return *from that state onwards*
            num_steps_in_rollout = len(rollout_rewards)
            for t, state_id in enumerate(visited_ids_in_order):
                if state_id is None: continue # Skip if state ID is invalid

                # Calculate G_t: discounted return from step t onwards
                G_t = sum([(self.gamma ** k) * rollout_rewards[t+k] for k in range(num_steps_in_rollout - t)])

                self.visit_counts[state_id] += 1
                current_v = self.v_table[state_id] # Get current value (defaults to 0)
                # Incremental average update: V <- V + (G_t - V) / N
                new_v = current_v + (G_t - current_v) / self.visit_counts[state_id]
                self.v_table[state_id] = new_v
                updated_states_in_batch.add(state_id)
                updated_state_values[state_id] = new_v  # Store final V-table value

        # --- End Backpropagation ---

        # Update MC step data with final V-table values
        for step_data in all_rollout_steps_data:
            mc_state_id = step_data.get("env_state_id")
            if mc_state_id is not None and mc_state_id in updated_state_values:
                step_data["value"] = updated_state_values[mc_state_id]
            # Remove env_state_id as it's no longer needed after value update
            step_data.pop("env_state_id", None)


        # Calculate average return for logging (optional, doesn't affect V-table)
        if num_valid_rollouts == 0:
            avg_return_for_log = 0.0
            # If no rollouts completed, the V-table wasn't updated for the root.
            final_v_estimate = self.v_table[root_state_id] # Return existing value
        else:
            # Use the updated value from the V-table after all rollouts
            final_v_estimate = self.v_table[root_state_id]
            # Calculate avg return for logging separately
            avg_return_for_log = total_return_for_log / num_valid_rollouts


        console.print(f"[cyan]Updated V-table for {len(updated_states_in_batch)} states using {num_valid_rollouts} rollouts (Avg Rollout Return: {avg_return_for_log:.3f}). V({escape(str(root_state_id))}) = {final_v_estimate:.3f} (N={self.visit_counts[root_state_id]})[/cyan]")

        # Return the *updated V-table value* for the root state AND the collected step data
        return final_v_estimate, all_rollout_steps_data


    def _normalize_per_episode(self, tensor_to_normalize: torch.Tensor, dones: List[bool]) -> torch.Tensor:
        """Normalizes a tensor per episode based on done flags."""
        normalized_list = []
        start_idx = 0
        device = tensor_to_normalize.device

        if len(tensor_to_normalize) <= 1:
            return torch.zeros_like(tensor_to_normalize)

        for i in range(len(dones)):
            if dones[i]:
                episode_tensor = tensor_to_normalize[start_idx : i + 1]
                if len(episode_tensor) > 1:
                    mean = episode_tensor.mean()
                    std = episode_tensor.std()
                    norm_tensor = (episode_tensor - mean) / (std + 1e-8)
                elif len(episode_tensor) == 1:
                    norm_tensor = torch.zeros_like(episode_tensor)
                else:
                    norm_tensor = torch.tensor([], device=device)

                normalized_list.append(norm_tensor)
                start_idx = i + 1

        if start_idx < len(tensor_to_normalize):
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
            # Return empty tensor matching device and dtype if list is empty
            return torch.tensor([], device=device, dtype=tensor_to_normalize.dtype)

    def _generate_action(self, chat_history: List[Dict[str, str]]) -> Tuple[str, bool]:
        """Generate an action using the policy model and return whether the action tag was found.
           Uses history truncation, generation temperature, prepends <thinking>, and uses top_p.
        """
        truncated_history = self._truncate_history(chat_history)
        prompt = self.processing_class.apply_chat_template(truncated_history, tokenize=False)

        # --- Append <thinking> to the prompt string ---
        think_token_str = "<thinking>"
        prompt_with_thinking = prompt + think_token_str
        # --- End Append <thinking> ---

        # Tokenize the modified prompt
        inputs = self.processing_class(prompt_with_thinking, return_tensors="pt")

        device = self._get_device()
        inputs = {k: v.long().to(device) for k, v in inputs.items()}

        stopping_criteria_lower = StopOnActionTag(self.processing_class, stop_token="</action>")
        stopping_criteria_upper = StopOnActionTag(self.processing_class, stop_token="</Action>")
        stopping_criteria_list = StoppingCriteriaList([stopping_criteria_lower, stopping_criteria_upper])

        # print(f"INFO: Inputs device: {inputs['input_ids'].device}")
        # print(f"INFO: Model device: {self.model.device}") # Model is prepared by Accelerator

        with torch.no_grad():
            # Use the prepared model for generation
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=100, # Keep max_new_tokens reasonable
                pad_token_id=self.processing_class.eos_token_id,
                do_sample=True,
                temperature=self.generation_temperature, # Use configured temperature
                top_p=0.9, # Use top_p like in VinePPOTrainer
                stopping_criteria=stopping_criteria_list
            )

        prompt_length = inputs["input_ids"].shape[1] # Length of prompt_with_thinking
        # Decode only the newly generated tokens
        action_text_raw = self.processing_class.decode(
            outputs[0][prompt_length:],
            skip_special_tokens=True
        )

        # Clean assistant/system tags if they accidentally appear
        cleaned_action_text_raw = re.sub(r'(?i)\b(assistant|system)\b\s*', '', action_text_raw).strip()

        # Prepend "<thinking>" as it was part of the prompt but removed by slicing
        # Check if the cleaned text *already* starts with <thinking>... just in case
        if cleaned_action_text_raw.startswith(think_token_str):
            final_action_text = cleaned_action_text_raw
        else:
            final_action_text = think_token_str + cleaned_action_text_raw

        # Determine if stop token was found based on stopping criteria flags
        has_tag = stopping_criteria_lower.found_stop_token or stopping_criteria_upper.found_stop_token

        # Also double-check if both tags exist in the final text (case-insensitive) for robustness
        lower_text = final_action_text.lower()
        has_tag = has_tag and ("<action>" in lower_text and "</action>" in lower_text)

        return final_action_text, has_tag

    def _compute_action_stats(self, chat_history: List[Dict[str, str]], action: str, model_to_use=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes per-token log probabilities for policy/ref models, action mask, and policy logits.
           Scales logits by generation_temperature."""
        device = self._get_device()

        # Determine which model to use for the main calculation
        active_model = model_to_use if model_to_use is not None else self.model

        prompt = self.processing_class.apply_chat_template(chat_history, tokenize=False)
        chat_with_action = copy.deepcopy(chat_history)
        chat_with_action.append({"role": "assistant", "content": action})
        full_prompt = self.processing_class.apply_chat_template(chat_with_action, tokenize=False)

        prompt_tokens = {k: v.to(device) for k, v in self.processing_class(prompt, return_tensors="pt").items()}
        full_tokens = {k: v.to(device) for k, v in self.processing_class(full_prompt, return_tensors="pt").items()}

        prompt_len = prompt_tokens["input_ids"].size(1)
        full_len = full_tokens["input_ids"].size(1)
        action_len = full_len - prompt_len

        if action_len <= 0:
            raise ValueError(f"Action length is non-positive ({action_len}) after tokenization. Prompt len: {prompt_len}, Full len: {full_len}. Action text: '{action}'")


        # Use the selected model (policy or ref)
        outputs = active_model(input_ids=full_tokens["input_ids"], attention_mask=full_tokens["attention_mask"])

        # Slice logits from prompt_len-1 up to full_len-2
        logits = outputs.logits[:, prompt_len - 1 : full_len - 1, :]
        # --- Apply temperature scaling ---
        scaled_logits = logits / self.generation_temperature
        log_probs_dist = torch.log_softmax(scaled_logits, dim=-1)
        # --- End temperature scaling ---

        # Action tokens are from prompt_len up to full_len
        action_token_ids = full_tokens["input_ids"][:, prompt_len:full_len]

        # Ensure dimensions match for gather operation
        if action_token_ids.size(1) != log_probs_dist.size(1):
            raise ValueError(f"Logits/Token shape mismatch: log_probs_dist dim 1 ({log_probs_dist.size(1)}) vs action_token_ids dim 1 ({action_token_ids.size(1)}).")

        # Get the log probability of the actual action tokens generated
        per_token_log_probs = log_probs_dist.gather(2, action_token_ids.unsqueeze(-1)).squeeze(-1) # Shape: (batch_size=1, action_len)

        # Compute reference log probability using the reference model IF active_model is the policy model
        ref_per_token_log_probs = torch.zeros_like(per_token_log_probs) # Default
        if active_model is self.model and self.ref_model is not None:
            with torch.no_grad():
                ref_outputs = self.ref_model(input_ids=full_tokens["input_ids"], attention_mask=full_tokens["attention_mask"])
                ref_logits = ref_outputs.logits[:, prompt_len - 1 : full_len - 1, :]
                # --- Apply temperature scaling to ref logits ---
                scaled_ref_logits = ref_logits / self.generation_temperature
                # --- End temperature scaling ---
                if scaled_ref_logits.size(1) != action_token_ids.size(1):
                    raise ValueError(f"Ref Logits/Token shape mismatch: ref_logits dim 1 ({scaled_ref_logits.size(1)}) vs action_token_ids dim 1 ({action_token_ids.size(1)}).")

                ref_log_probs_dist = torch.log_softmax(scaled_ref_logits, dim=-1)
                ref_per_token_log_probs = ref_log_probs_dist.gather(2, action_token_ids.unsqueeze(-1)).squeeze(-1)
        elif active_model is self.model and self.ref_model is None:
            # If no ref model, use policy model's detached log probs as reference (zero KL)
            ref_per_token_log_probs = per_token_log_probs.detach()

        # Create the action mask
        action_mask = torch.ones_like(per_token_log_probs, device=device)

        # Return per-token log probs, ref log probs, mask, and the *unscaled* policy logits (needed for entropy)
        # Squeeze the batch dimension
        return per_token_log_probs.squeeze(0), ref_per_token_log_probs.squeeze(0), action_mask.squeeze(0), logits.squeeze(0)


    def _initialize_v_table_from_q_table(self):
        """
        Initializes the state-value table V(s) by averaging Q(s, a) for each state
        present in the provided self.q_table. Resets visit counts.
        """
        if not self.q_table:
            console.print("[yellow]Warning: Q-table is empty or None. Cannot initialize V-table from it.[/yellow]")
            self.v_table = defaultdict(float)
            self.visit_counts = defaultdict(int)
            return

        state_q_values = defaultdict(list)
        for (state_id, _), q_value in self.q_table.items():
            state_q_values[state_id].append(q_value)

        self.v_table = defaultdict(float) # Initialize V-table
        self.visit_counts = defaultdict(int) # Reset visit counts
        total_v_value = 0
        num_states = 0
        for state_id, q_values in state_q_values.items():
            if q_values:
                self.v_table[state_id] = sum(q_values) / len(q_values)
                total_v_value += self.v_table[state_id]
                num_states += 1

        avg_initial_v = total_v_value / num_states if num_states > 0 else 0.0
        console.print(f"[blue]INFO: Initialized V-table with {len(self.v_table)} states from Q-table. Average initial V(s) = {avg_initial_v:.3f}[/blue]")


    def compute_loss(
        self, model: nn.Module, inputs: dict, return_outputs: bool = False, num_items_in_batch: Optional[int] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        # Collect rollout episodes. Each _rollout_episode call adds main + MC steps to buffer.
        for _ in range(self.rollout_batch_size):
            self._rollout_episode()

        # Compute returns/advantages using values stored in the buffer (which came from V-table/MC)
        self.buffer.compute_returns_and_advantages()

        batch: VineBufferSample = self.buffer.get() # Get all data collected

        if len(batch.states) == 0:
            # Handle empty buffer case
            grad_param = next((p for p in model.parameters() if p.requires_grad), None)
            if grad_param is not None:
                computed_loss = (grad_param * 0.0).sum() # Zero loss with grad_fn
            else:
                computed_loss = torch.tensor(0.0, device=self._get_device(), requires_grad=True)
            print("WARNING: Buffer was empty after rollouts. Using zero loss.")
            metrics = {"loss": 0.0, "policy_loss": 0.0, "kl_loss": 0.0, "entropy": 0.0, "avg_reward": 0.0, "avg_episode_length": 0.0, "avg_advantage": 0.0, "avg_value": 0.0}
            return (computed_loss, metrics) if return_outputs else computed_loss

        else:
            device = self._get_device()

            # Advantages are always computed by the buffer (GAE or TD)
            advantages_source_tensor = torch.tensor(batch.advantages, device=device, dtype=torch.float32)

            # Normalize advantages based on type
            if self.buffer.advantage_type == "gae":
                # Normalize GAE advantages across the batch
                advantages_to_use = (advantages_source_tensor - advantages_source_tensor.mean()) / (advantages_source_tensor.std() + 1e-8)
                console.print(f"[blue]INFO: Using normalized GAE advantages ({len(advantages_source_tensor)} steps).[/blue]")
            elif self.buffer.advantage_type == "td":
                # TD advantages are typically used directly (or sometimes normalized too)
                # Let's normalize TD advantages as well for consistency
                # advantages_to_use = (advantages_source_tensor - advantages_source_tensor.mean()) / (advantages_source_tensor.std() + 1e-8)
                advantages_to_use = advantages_source_tensor # Option: use unnormalized TD
                console.print(f"[blue]INFO: Using normalized TD advantages ({len(advantages_source_tensor)} steps).[/blue]")
            else:
                raise ValueError(f"Unsupported advantage type: {self.buffer.advantage_type}")


            # --- Start Per-Token Loss Calculation ---
            all_per_token_losses = []
            all_per_token_kls = []
            all_masks = []
            all_advantages_expanded = []
            all_ppo_per_token_losses = []
            all_per_token_entropies = [] # List to store entropy

            current_advantage_idx = 0
            # Process each step (main or MC) stored in the batch
            for i in range(len(batch.states)):
                chat_history = batch.states[i]
                action = batch.actions[i]
                # Old log probs came from the model state when the action was generated
                old_per_token_logp = torch.tensor(batch.old_log_probs[i], device=device)
                action_mask = torch.tensor(batch.action_masks[i], device=device)

                if action_mask.sum().item() <= 0:
                    # console.print(f"[yellow]Warning: Skipping step {i} due to zero action mask length.[/yellow]")
                    current_advantage_idx += 1
                    continue # Skip loss calculation for this step


                try:
                    # Recompute log probs, ref log probs, and get policy logits
                    per_token_logp, ref_per_token_logp, _, per_token_logits = self._compute_action_stats(
                        chat_history, action, model_to_use=self.model
                    )

                    # Align shapes
                    if per_token_logp.shape != old_per_token_logp.shape or \
                       per_token_logp.shape != ref_per_token_logp.shape or \
                       per_token_logp.shape != action_mask.shape:
                        raise ValueError(f"Shape mismatch at step {i}: p={per_token_logp.shape}, old={old_per_token_logp.shape}, ref={ref_per_token_logp.shape}, mask={action_mask.shape}")

                except Exception as e:
                    console.print(f"[bold red]Error recomputing stats in compute_loss for step {i}: {e}. Skipping step.[/bold red]")
                    current_advantage_idx += 1
                    continue


                # Get the advantage corresponding to this step
                if current_advantage_idx >= len(advantages_to_use):
                    raise IndexError(f"Advantage index {current_advantage_idx} out of bounds ({len(advantages_to_use)}). Check buffer/advantage logic.")
                advantage = advantages_to_use[current_advantage_idx]


                # Calculate per-token KL divergence (policy vs ref)
                per_token_kl = torch.zeros_like(per_token_logp)
                if self.beta != 0.0 and self.ref_model is not None:
                    logp_policy = per_token_logp
                    logp_ref = ref_per_token_logp.detach()
                    # KL = sum [ p_policy * (logp_policy - logp_ref) ]
                    # Approximation using log probs: ratio - 1 - log_ratio
                    log_ratio_for_kl = logp_policy - logp_ref
                    ratio_for_kl = torch.exp(log_ratio_for_kl)
                    # Clamp KL to avoid extreme values? Not typically done here.
                    per_token_kl = ratio_for_kl - 1.0 - log_ratio_for_kl # Ensures positivity


                # --- Calculate per-token entropy ---
                probs = torch.softmax(per_token_logits, dim=-1) # Use unscaled logits
                log_probs_full_dist = torch.log_softmax(per_token_logits, dim=-1)
                per_token_entropy = -(probs * log_probs_full_dist).sum(dim=-1)
                all_per_token_entropies.append(per_token_entropy)
                # --- End entropy calculation ---

                # Calculate PPO ratio and surrogate losses
                log_ratio = per_token_logp - old_per_token_logp.detach()
                ratio = torch.exp(log_ratio)
                # Use low/high clip values
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_eps_low, 1 + self.clip_eps_high)


                # Expand advantage to match the shape of per-token tensors
                expanded_advantage = advantage.expand_as(per_token_logp)

                per_token_loss1 = ratio * expanded_advantage
                per_token_loss2 = clipped_ratio * expanded_advantage
                ppo_per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

                # Combine PPO loss, KL penalty, and entropy bonus
                combined_per_token_loss = ppo_per_token_loss
                if self.beta != 0.0:
                    combined_per_token_loss += self.beta * per_token_kl
                if self.entropy_coeff != 0.0:
                    combined_per_token_loss -= self.entropy_coeff * per_token_entropy # Subtract entropy bonus


                all_per_token_losses.append(combined_per_token_loss)
                all_ppo_per_token_losses.append(ppo_per_token_loss)
                all_per_token_kls.append(per_token_kl)
                all_masks.append(action_mask)
                all_advantages_expanded.append(expanded_advantage)

                # Move to the next step's advantage
                current_advantage_idx += 1


            # Concatenate all tensors from the batch
            if all_per_token_losses:
                cat_per_token_losses = torch.cat(all_per_token_losses)
                cat_per_token_kls = torch.cat(all_per_token_kls)
                cat_masks = torch.cat(all_masks)
                cat_advantages = torch.cat(all_advantages_expanded)
                cat_ppo_per_token_losses = torch.cat(all_ppo_per_token_losses)
                cat_entropies = torch.cat(all_per_token_entropies) # Concatenated entropies

                # Ensure masks sum is not zero
                valid_mask_sum = cat_masks.sum().clamp(min=1e-8)

                # Compute final loss with different normalization options
                # loss_type = getattr(self.args, 'loss_type', 'default') # Use self.loss_type
                if self.loss_type == 'grpo':
                    # Average loss per sequence, then average over sequences
                    # This requires knowing sequence boundaries, which we don't have easily here
                    # Revert to default token averaging for now
                    masked_loss = cat_per_token_losses * cat_masks
                    computed_loss = masked_loss.sum() / valid_mask_sum
                    # console.print("[yellow]Warning: 'grpo' loss type not fully implemented without sequence boundaries, using default token averaging.[/yellow]")
                elif self.loss_type == 'dr_grpo':
                    # Divide sum of losses by total number of steps * max_completion_length
                    masked_loss = cat_per_token_losses * cat_masks
                    computed_loss = masked_loss.sum() / (len(batch.states) * getattr(self.args, 'max_completion_length', 100))
                else:  # default
                    # Average over all valid tokens in the batch
                    masked_loss = cat_per_token_losses * cat_masks
                    computed_loss = masked_loss.sum() / valid_mask_sum

                # Compute metrics
                mean_kl = (cat_per_token_kls * cat_masks).sum() / valid_mask_sum
                mean_policy_loss = (cat_ppo_per_token_losses * cat_masks).sum() / valid_mask_sum
                avg_advantage = (cat_advantages * cat_masks).sum() / valid_mask_sum
                mean_entropy = (cat_entropies * cat_masks).sum() / valid_mask_sum # Calculate mean entropy

                # Calculate average value estimate from the batch values
                batch_values = torch.tensor(batch.values, device=device)
                avg_value = batch_values.mean().item() if len(batch_values) > 0 else 0.0

                metrics = {
                    "loss": computed_loss.item(),
                    "policy_loss": mean_policy_loss.item(),
                    "kl_loss": mean_kl.item(),
                    "entropy": mean_entropy.item(), # Log average entropy
                    "avg_reward": sum(self.episode_rewards) / len(self.episode_rewards) if self.episode_rewards else 0.0,
                    "avg_episode_length": sum(self.episode_lengths) / len(self.episode_lengths) if self.episode_lengths else 0.0,
                    "avg_advantage": avg_advantage.item(),
                    "avg_value": avg_value,
                }
            else:
                # Handle case where batch processing resulted in no valid data
                computed_loss = torch.tensor(0.0, device=device, requires_grad=True)
                metrics = {"loss": 0.0, "policy_loss": 0.0, "kl_loss": 0.0, "entropy": 0.0, "avg_reward": 0.0, "avg_episode_length": 0.0, "avg_advantage": 0.0, "avg_value": 0.0}


            self.log(metrics)
            self.buffer.clear()
            self.episode_rewards = []
            self.episode_lengths = []

        if return_outputs:
            # Ensure metrics dict is always returned if loss computation happened
            if 'metrics' not in locals():
                metrics = {"loss": computed_loss.item(), "policy_loss": 0.0, "kl_loss": 0.0, "entropy": 0.0, "avg_reward": 0.0, "avg_episode_length": 0.0, "avg_advantage": 0.0, "avg_value": 0.0}
            return computed_loss, metrics
        else:
            return computed_loss


    def _rollout_episode(self) -> float:
        """
        Generate an episode using the chat template.
        Collects steps (including from MC rollouts) and adds them to the buffer.
        Updates V-table via _mc_value calls.
        Includes enhanced action retries.
        """
        env = self.env_factory()
        init_text, _ = env.reset()

        system_prompt = ""
        if hasattr(env, 'system_prompt') and env.system_prompt:
            system_prompt = env.system_prompt

        chat_history = []
        if system_prompt:
            chat_history.append({"role": "system", "content": system_prompt})

        chat_history.append({"role": "user", "content": init_text})

        console.print(f"[bold green]Starting new episode[/bold green]")
        console.print(f"[cyan]Initial state:[/cyan] {init_text}")

        done = False
        truncated = False
        episode_reward = 0
        step_count = 0

        episode_data = [] # Stores dicts for steps (main + MC) before adding to buffer

        max_action_retries = 5
        while not done and not truncated and step_count < env.max_steps:
            action = ""
            has_action_tag = False
            is_valid_action = False # Track validity

            for attempt in range(max_action_retries):
                action, has_action_tag = self._generate_action(chat_history)
                escaped_action = escape(action)
                # console.print(f"[yellow](Attempt {attempt + 1}/{max_action_retries}) Generated Action:[/yellow] {escaped_action}")
                # console.print(f"[{'green' if has_action_tag else 'red'}]Action Tag Found: {has_action_tag}[/{'green' if has_action_tag else 'red'}]")

                # Check if action is valid (exists in available actions)
                parsed_action = self._extract_action_tag(action)
                available_actions = env.get_available_actions() if hasattr(env, 'get_available_actions') else []
                is_valid_action = parsed_action in available_actions if parsed_action else False
                # console.print(f"[{'green' if is_valid_action else 'red'}]Is Valid Action: {is_valid_action}[/{'green' if is_valid_action else 'red'}]")

                if has_action_tag and is_valid_action:
                    break # Stop retrying if action has tag and is valid
                elif attempt == max_action_retries - 1:
                    console.print(f"[yellow]Warning: Max retries reached. Proceeding with action (Tag: {has_action_tag}, Valid: {is_valid_action}): {escaped_action}[/yellow]")


            # --- Determine Value Estimate V(s) & Run MC ---
            estimated_value = 0.0
            mc_steps_data = []
            try:
                state_id = env.current_state_id # Get state ID before MC/V-table lookup
            except AttributeError:
                console.print("[bold red]Error: Environment does not have 'current_state_id'. Cannot use V-table or MC updates. Ending episode.[/bold red]")
                break # End episode if state ID is missing

            # Perform MCTS-like value update and get MC step data
            # Pass has_action_tag AND is_valid_action to mc_value
            console.print(f"[blue]Performing MC rollouts/update for state {escape(str(state_id))}[/blue]")
            estimated_value, mc_steps_data = self._mc_value(env, chat_history, has_action_tag and is_valid_action)
            # --- End Value Estimate ---

            # Calculate per-token log probs, mask, and logits for the main action
            try:
                per_token_logp, _, action_mask, _ = self._compute_action_stats(
                    chat_history, action, model_to_use=self.model
                )
                log_prob_np = per_token_logp.detach().cpu().float().numpy()
                action_mask_np = action_mask.detach().cpu().float().numpy()
                main_stats_computed = True
            except Exception as e:
                console.print(f"[bold red]Error computing stats for main action: {e}. Ending episode.[/bold red]")
                break # End episode

            current_chat_history = copy.deepcopy(chat_history) # History *before* adding assistant action
            chat_history.append({"role": "assistant", "content": action})

            next_state_text, reward, done, truncated, _ = env.step(action)

            # Apply penalty based on original has_action_tag/is_valid_action checks
            if not has_action_tag or not is_valid_action:
                # Reward should already be negative from env.step in these cases
                console.print(f"[bold red]Action invalid or tag missing. Reward: {reward}[/bold red]")


            chat_history.append({"role": "user", "content": next_state_text})

            # Store data for the *main episode step*
            episode_data.append({
                "step_num": step_count + 1,
                "state_history": current_chat_history,
                "action": action,
                "reward": reward,
                "value": estimated_value, # Updated V(s) from MC
                "log_prob": log_prob_np,
                "action_mask": action_mask_np,
                "has_action_tag": has_action_tag, # Store original tag status
                "is_valid_action": is_valid_action, # Store validity status
                "next_state_text": next_state_text,
                "done": done,
                "truncated": truncated,
                "is_mc_step": False
            })

            # --- Add MC rollout steps data ---
            for mc_step in mc_steps_data:
                # Add MC steps to episode data, ensure they have required fields
                if "log_prob" in mc_step and "action_mask" in mc_step and \
                    len(mc_step["log_prob"]) > 0 and len(mc_step["action_mask"]) > 0:
                    episode_data.append({
                        "step_num": f"MC_{mc_step['mc_rollout_id']}_{mc_step['mc_step_count']}",
                        "state_history": mc_step["state_history"],
                        "action": mc_step["action"],
                        "reward": mc_step["reward"],
                        "value": mc_step["value"],
                        "log_prob": mc_step["log_prob"],
                        "action_mask": mc_step["action_mask"],
                        "has_action_tag": True, # Assume True for MC steps
                        "is_valid_action": True, # Assume True for MC steps
                        "next_state_text": "N/A (MC step)",
                        "done": mc_step["done"],
                        "truncated": mc_step["truncated"],
                        "is_mc_step": True
                    })
                else:
                    console.print(f"[yellow]Warning: Skipping MC step {mc_step.get('mc_rollout_id', '?')}_{mc_step.get('mc_step_count', '?')} due to missing data.[/yellow]")

            # --- End adding MC data ---

            episode_reward += reward
            step_count += 1
        # --- End main episode loop ---

        # --- Add collected data (main + MC) to buffer ---
        start_buffer_idx = self.buffer.size
        items_added_count = 0
        for i, step_data in enumerate(episode_data):
            # Determine buffer 'done' flag (end of sequence for GAE/TD)
            buffer_done = step_data["done"] or step_data["truncated"]

            # Ensure log_prob and action_mask are numpy arrays before adding
            log_prob_add = step_data["log_prob"]
            action_mask_add = step_data["action_mask"]
            if not isinstance(log_prob_add, np.ndarray): log_prob_add = np.array(log_prob_add)
            if not isinstance(action_mask_add, np.ndarray): action_mask_add = np.array(action_mask_add)

            self.buffer.add(
                state=step_data["state_history"],
                action=step_data["action"],
                reward=step_data["reward"],
                value=step_data["value"],
                log_prob=log_prob_add,
                action_mask=action_mask_add,
                done=buffer_done
            )
            items_added_count += 1
        # --- End buffer add ---

        # Compute GAE/TD over the *entire sequence* just added
        self.buffer.compute_returns_and_advantages()

        # --- Extract advantages/returns for logging ---
        end_buffer_idx = self.buffer.size
        if end_buffer_idx - start_buffer_idx != items_added_count:
            console.print(f"[bold red]Error: Mismatch between items added ({items_added_count}) and buffer size increase ({end_buffer_idx - start_buffer_idx}). Check buffer logic.[/bold red]")
            episode_advantages = []
            episode_returns = []
        else:
            episode_advantages = self.buffer.advantages[start_buffer_idx : end_buffer_idx]
            episode_returns = self.buffer.returns[start_buffer_idx : end_buffer_idx]
        # --- End GAE/TD Extraction ---

        # --- Print the table ---
        table = Table(title="Episode Trajectory (incl. MC steps)", box=box.ROUNDED)
        table.add_column("Step", justify="right", style="cyan")
        table.add_column("Type", justify="center")
        table.add_column("Assistant (Action)", style="yellow", no_wrap=False, max_width=80)
        table.add_column("Reward", justify="right", style="magenta")
        table.add_column("Value (V_tbl)", justify="right", style="blue")
        table.add_column("Return (" + self.buffer.advantage_type.upper() + ")", justify="right", style="purple")
        table.add_column("Adv (" + self.buffer.advantage_type.upper() + ")", justify="right", style="orange_red1")
        table.add_column("Tag", justify="center", style="red")
        table.add_column("Valid", justify="center", style="green") # New column for validity
        table.add_column("Done", justify="center")
        table.add_column("Trunc", justify="center")

        processed_step_idx = 0
        for i, step_data in enumerate(episode_data):
            # Check if we have corresponding advantage/return data
            if processed_step_idx >= len(episode_advantages):
                console.print(f"[yellow]Warning: Log index mismatch for step {step_data['step_num']}. Skipping log row.[/yellow]")
                continue # Skip logging this row

            advantage = episode_advantages[processed_step_idx]
            ret = episode_returns[processed_step_idx]

            step_type = "MC" if step_data["is_mc_step"] else "Main"
            action_tag_symbol = "✓" if step_data.get("has_action_tag", False) else "✗"
            valid_action_symbol = "✓" if step_data.get("is_valid_action", False) else "✗" # Check validity
            display_action = escape(step_data["action"])
            done_symbol = "✓" if step_data["done"] else ""
            trunc_symbol = "✓" if step_data["truncated"] else ""

            table.add_row(
                str(step_data["step_num"]),
                step_type,
                display_action,
                f"{step_data['reward']:.2f}",
                f"{step_data['value']:.3f}",
                f"{ret:.2f}",
                f"{advantage:.2f}",
                action_tag_symbol,
                valid_action_symbol, # Add validity symbol
                done_symbol,
                trunc_symbol
            )
            processed_step_idx += 1

        console.print(table)
        # --- End table print ---

        console.print(Panel(f"Episode finished with reward: {episode_reward:.2f}", style="bold green"))
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(step_count) # Length based on main steps

        return episode_reward


    def train(self, num_ppo_updates: int = 100):
        """
        Custom training loop for PPO with MCTS value updates.
        Includes periodic reference model updates.
        """
        self.model.train()
        gradient_accumulation_steps = getattr(self.args, 'gradient_accumulation_steps', 1)
        print(f"INFO: Using gradient accumulation with {gradient_accumulation_steps} steps.")

        total_optimizer_steps = num_ppo_updates
        total_accumulation_steps = total_optimizer_steps * gradient_accumulation_steps

        print(f"INFO: Total optimizer steps: {total_optimizer_steps}")
        print(f"INFO: Total accumulation steps: {total_accumulation_steps}")

        for step in range(total_accumulation_steps):
            # compute_loss handles rollouts, buffer management, and loss calculation
            loss = self.compute_loss(self.model, {}, return_outputs=False)

            # Scale loss for gradient accumulation
            scaled_loss = loss / max(gradient_accumulation_steps, 1)

            self.accelerator.backward(scaled_loss)

            # Log scaled loss for this accumulation step
            self.log({"train/accumulation_step_loss": scaled_loss.item()})

            # Optimizer step after accumulating gradients
            if (step + 1) % gradient_accumulation_steps == 0:
                effective_step = (step + 1) // gradient_accumulation_steps

                grad_norm = None
                if self.accelerator.sync_gradients:
                    max_norm = getattr(self.args, 'max_grad_norm', 1.0)
                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm)

                if grad_norm is not None:
                    # Ensure grad_norm is logged correctly if it's a tensor
                    log_grad_norm = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
                    self.log({"train/grad_norm": log_grad_norm})


                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                    lr_list = self.lr_scheduler.get_last_lr()
                    current_lr = lr_list[0] if lr_list else 0.0
                    self.log({"train/learning_rate": current_lr})

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

                # Update vLLM model weights periodically (e.g., after each optimizer step)
                self._update_vllm_model()

                print(f"INFO: Optimizer Step {effective_step}/{total_optimizer_steps} completed. Grad Norm: {f'{log_grad_norm:.4f}' if grad_norm is not None else 'N/A'}")

                # Save model checkpoint
                save_frequency = getattr(self.args, 'save_steps', 10)
                if self.accelerator.is_main_process and effective_step % save_frequency == 0:
                    save_path = os.path.join(self.args.output_dir, f"Step_{effective_step}")
                    # Use accelerator.save_state for full training state
                    self.accelerator.save_state(save_path)
                    # Optionally save just the model weights using save_pretrained or accelerator's method
                    # self.save_model(save_path) # This saves HF model/config
                    console.print(f"[green]Saved accelerator state at effective step {effective_step} to {save_path}[/green]")

        console.print("[bold green]Training complete.[/bold green]") 