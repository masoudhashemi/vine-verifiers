import argparse
import os
import sys
import random
import time
from typing import List, Dict, Any, Optional, Tuple
import gc
from collections import defaultdict
import re
import logging # Added logging

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import verifiers as vf
from verifiers.utils.imports import LLM, SamplingParams  # type: ignore
from verifiers.envs.two_treasures_maze_gym_env import TwoTreasuresMazeGymEnv, MAZE_GYM_SYSTEM_PROMPT, VALID_ACTIONS_LIST, T1_REWARD, T2_REWARD

# Configure logging for the demo
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def print_messages(messages: List[Dict[str, str]], skip_system: bool = False, skip_initial_prompt: bool = False):
    """Print the conversation messages in a readable format."""
    start_idx = 0
    if skip_system and messages and messages[0]["role"] == "system":
        start_idx = 1
    
    # In this demo, the first user message IS the initial state, so we might want to skip it
    # if we've already printed it or if skip_initial_prompt is True.
    # Let's adjust to handle the case where the "few_shot" concept doesn't directly apply
    # but we want to control printing the first user message which contains the initial state.
    
    # For this demo, the typical pattern is:
    # 1. System Prompt (optional, handled by env)
    # 2. User Prompt (Initial State from env.reset())
    # 3. Assistant Action
    # 4. User Prompt (Next State from env.step())
    # ...

    # If skip_initial_prompt is true, and we are past the system prompt, skip the next user message.
    if skip_initial_prompt and start_idx < len(messages) and messages[start_idx]["role"] == "user":
        start_idx +=1

    for i, msg in enumerate(messages[start_idx:], start=start_idx):
        role = msg["role"].upper()
        print(f"\n[{role}]:")
        # For maze env, the observation can be long. We might want to parse it or summarize.
        # For now, just print.
        if role == "USER" and "<state>" in msg["content"]:
            # Extract content within <state> for cleaner printing if desired
            # match = re.search(r"<state>(.*?)</state>", msg["content"], re.DOTALL)
            # if match:
            #     print(match.group(1).strip())
            # else:
            #     print(msg["content"])
            print(msg["content"]) # Print full for now
        else:
            print(msg["content"])


def _get_start_maze(env: TwoTreasuresMazeGymEnv, preferred_maze_idx: Optional[str]) -> str:
    """Helper to determine the start maze, prompting if necessary."""
    if preferred_maze_idx and preferred_maze_idx in env.starting_states:
        return preferred_maze_idx
    elif preferred_maze_idx:
        logger.warning(f"Provided start maze index '{preferred_maze_idx}' is not valid. Choose from list or let system choose randomly.")

    logger.info("\nAvailable starting mazes (by index):")
    for i, maze_id_str in enumerate(env.starting_states):
        # We don't have descriptions for mazes in the same way as states,
        # so we'll just show the index. The user can inspect MAZE_X definitions if curious.
        logger.info(f"{i}. Maze {maze_id_str}")
    
    while True:
        try:
            choice = input(f"Choose a starting maze by index (0-{len(env.starting_states)-1}), or press Enter for random: ").strip()
            if not choice:
                # env.reset() will pick randomly if start_state_id is None
                return random.choice(env.starting_states)
            
            # Ensure the choice is a valid index string present in starting_states
            if choice in env.starting_states:
                return choice
            else:
                logger.warning("Invalid choice. Please pick a valid index from the list.")
        except ValueError:
            logger.warning("Invalid input. Enter a number or press Enter.")


def run_human_interactive(max_steps: int = 25, start_maze_idx: Optional[str] = None):
    """Run an interactive game where a human player makes all decisions for TwoTreasuresMazeGymEnv."""
    continue_playing = True
    
    while continue_playing:
        logger.info("\n" + "="*80)
        logger.info("TWO TREASURES MAZE: HUMAN INTERACTIVE MODE")
        logger.info("="*80)
        
        env = TwoTreasuresMazeGymEnv(max_steps_gym=max_steps, seed=random.randint(0, 2**30))
        
        chosen_maze_idx_str = _get_start_maze(env, start_maze_idx)
        logger.info(f"Starting game with Maze Index: {chosen_maze_idx_str}")
        
        # env.reset() returns observation, info
        # The observation is already formatted with <state> tags and "What to do next"
        observation, info = env.reset(start_state_id=chosen_maze_idx_str) 
        
        logger.info("\nINITIAL STATE:")
        # The observation from TwoTreasuresMazeGymEnv is the full prompt for the LLM.
        # We can print it directly.
        print(observation) 
        
        current_gym_step = 0
        done_gym = False
        truncated_gym = False
        
        while not done_gym and not truncated_gym and current_gym_step < max_steps:
            current_gym_step += 1
            logger.info(f"\n--- GYM STEP {current_gym_step} ---")
            
            # Display available actions to the human
            logger.info("\nAvailable actions:")
            for i, action_name in enumerate(VALID_ACTIONS_LIST, 1):
                logger.info(f"{i}. {action_name}")
            
            chosen_action_for_llm = ""
            while True:
                try:
                    choice = input("\nChoose an action (number): ")
                    choice_idx = int(choice) - 1
                    
                    if 0 <= choice_idx < len(VALID_ACTIONS_LIST):
                        raw_action = VALID_ACTIONS_LIST[choice_idx]
                        chosen_action_for_llm = f"<thinking>{raw_action}</thinking>"
                        break
                    else:
                        logger.warning("Invalid choice. Please pick a valid number.")
                except ValueError:
                    logger.warning("Invalid input. Enter a number corresponding to an action.")
            
            # Pass the LLM-formatted action to env.step()
            observation, reward, done_gym, truncated_gym, info = env.step(chosen_action_for_llm)
            
            logger.info("\n" + "="*50)
            logger.info(f"OBSERVATION (Prompt for next step):\n{observation}")
            logger.info(f"REWARD: {reward:.2f}")
            logger.info(f"INFO: {info}")
            
            if done_gym or truncated_gym:
                final_game_message = info.get("final_message", env.final_message) # env.final_message should be set
                if "found Treasure 2" in final_game_message:
                    logger.info(f"\n🎉 GAME OVER: You found Treasure 2 (T2)! Highest value! Steps: {env.current_step_internal}")
                elif "found Treasure 1" in final_game_message:
                    logger.info(f"\n⚠️ GAME OVER: You found Treasure 1 (T1)! Good, but T2 is better. Steps: {env.current_step_internal}")
                elif "Maximum steps reached" in final_game_message or "truncated by gym step limit" in final_game_message:
                     logger.info(f"\n⏱️ GAME OVER: Ran out of time. Steps: {env.current_step_internal}")
                else: # Generic failure or other end state
                    logger.info(f"\n💀 GAME OVER: {final_game_message}. Steps: {env.current_step_internal}")

        if current_gym_step >= max_steps and not (done_gym or truncated_gym):
            # This case should be handled by truncated_gym=True from the env
            logger.info(f"\n⏱️ GAME OVER: Reached max demo steps ({current_gym_step}).")
        
        while True:
            play_again = input("\nDo you want to play again? (y/n): ").strip().lower()
            if play_again in ['y', 'yes']:
                break
            elif play_again in ['n', 'no']:
                continue_playing = False
                break
            else:
                logger.warning("Invalid input. Please enter 'y' or 'n'.")
    
    # Return true if T2 was found in the last game.
    # This requires inspecting the 'info' or 'env.final_message' from the last step of the last game.
    # For simplicity, we won't track explicit success like this for human mode.
    return True 


def run_llm_interactive(model_name: str, max_steps: int = 25, start_maze_idx: Optional[str] = None,
                        vllm_device="cuda", gpu_memory_utilization=0.9, 
                        vllm_dtype="auto", enable_prefix_caching=True, 
                        max_model_len=None):
    """Run an interactive game where an LLM makes all decisions for TwoTreasuresMazeGymEnv."""
    logger.info(f"Loading model: {model_name}")
    model = LLM(
        model=model_name,
        device=vllm_device,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=vllm_dtype,
        enable_prefix_caching=enable_prefix_caching,
        max_model_len=max_model_len,
    )
    
    continue_playing = True
    last_game_won_t2 = False
    
    while continue_playing:
        logger.info("\n" + "="*80)
        logger.info(f"TWO TREASURES MAZE: LLM INTERACTIVE MODE (Model: {model_name})")
        logger.info("="*80)
        
        env = TwoTreasuresMazeGymEnv(max_steps_gym=max_steps, seed=random.randint(0, 2**30))
        
        chosen_maze_idx_str = _get_start_maze(env, start_maze_idx)
        logger.info(f"Starting game with Maze Index: {chosen_maze_idx_str}")
        
        # The system prompt is part of TwoTreasuresMazeGymEnv config
        messages = [{"role": "system", "content": env.system_prompt}]
        
        observation_gym, info_gym = env.reset(start_state_id=chosen_maze_idx_str)
        # The observation_gym is already the user prompt for the LLM, including <state> tags.
        messages.append({"role": "user", "content": observation_gym})
        
        logger.info("\nINITIAL STATE (as presented to LLM):")
        # Print messages, skipping system, but show initial user prompt
        print_messages(messages, skip_system=True, skip_initial_prompt=False) 
        
        current_gym_step = 0
        done_gym = False
        truncated_gym = False
        last_game_won_t2 = False # Reset for current game

        while not done_gym and not truncated_gym and current_gym_step < max_steps:
            current_gym_step += 1
            logger.info(f"\n--- GYM STEP {current_gym_step} ---")
            
            logger.info("Model thinking...")
            sampling_params = SamplingParams(
                temperature=0.1, top_p=0.7, max_tokens=50, # <thinking>action</thinking> is still short
                stop=["</thinking>"], 
                include_stop_str_in_output=True
            )
            
            # Model gets the history including the latest user message (current state)
            response = model.chat(messages=messages, sampling_params=sampling_params)
            
            assistant_msg_content = response[0].outputs[0].text.strip()
            # Ensure the response is wrapped in <thinking>...</thinking>
            if not assistant_msg_content.lower().startswith("<thinking>"):
                assistant_msg_content = "<thinking>" + assistant_msg_content
            if not assistant_msg_content.lower().endswith("</thinking>"):
                assistant_msg_content += "</thinking>"

            assistant_msg = {"role": "assistant", "content": assistant_msg_content}
            messages.append(assistant_msg)
            
            logger.info("\nMODEL RESPONSE:")
            print(assistant_msg["content"])
            
            # Pass LLM's action to the environment
            observation_gym, reward_gym, done_gym, truncated_gym, info_gym = env.step(assistant_msg["content"])
            
            # Add environment's response (next state) as a user message for the LLM
            env_user_msg = {"role": "user", "content": observation_gym}
            messages.append(env_user_msg)

            logger.info("\nENVIRONMENT RESPONSE (Prompt for next LLM step):")
            print(env_user_msg["content"]) # Print the new state/prompt
            logger.info(f"REWARD: {reward_gym:.2f}, INFO: {info_gym}")
            
            if done_gym or truncated_gym:
                final_game_message = info_gym.get("final_message", env.final_message)
                event = info_gym.get("game_event", "")
                if event == "found_t2":
                    logger.info(f"\n🎉 GAME OVER: LLM found Treasure 2 (T2)! Highest value! Steps: {env.current_step_internal}")
                    last_game_won_t2 = True
                elif event == "found_t1":
                    logger.info(f"\n⚠️ GAME OVER: LLM found Treasure 1 (T1). Steps: {env.current_step_internal}")
                elif event == "max_steps_internal" or info_gym.get("gym_truncated"):
                    logger.info(f"\n⏱️ GAME OVER: LLM ran out of time. Steps: {env.current_step_internal}")
                else: # Generic failure or other end state
                    logger.info(f"\n💀 GAME OVER: {final_game_message}. Steps: {env.current_step_internal}")

        if current_gym_step >= max_steps and not (done_gym or truncated_gym):
            logger.info(f"\n⏱️ GAME OVER: LLM reached max demo steps ({current_gym_step}).")
            if not done_gym: # If game hadn't ended naturally by this point
                 last_game_won_t2 = False


        logger.info("\n" + "="*80)
        logger.info("FINAL CONVERSATION (LLM Game):")
        print_messages(messages, skip_system=True, skip_initial_prompt=True) # Skip system and first user prompt
        logger.info("="*80)
        
        while True:
            play_again = input("\nDo you want to run another LLM game? (y/n): ").strip().lower()
            if play_again in ['y', 'yes']:
                break
            elif play_again in ['n', 'no']:
                continue_playing = False
                break
            else:
                logger.warning("Invalid input. Please enter 'y' or 'n'.")
    
    return last_game_won_t2


def run_random_interactive(max_steps: int = 25, start_maze_idx: Optional[str] = None):
    """Run an interactive game where actions are chosen randomly for TwoTreasuresMazeGymEnv."""
    continue_playing = True
    last_game_won_t2 = False

    while continue_playing:
        logger.info("\n" + "="*80)
        logger.info("TWO TREASURES MAZE: RANDOM AGENT INTERACTIVE MODE")
        logger.info("="*80)
        
        env = TwoTreasuresMazeGymEnv(max_steps_gym=max_steps, seed=random.randint(0, 2**30))
        
        chosen_maze_idx_str = _get_start_maze(env, start_maze_idx)
        logger.info(f"Starting game with Maze Index: {chosen_maze_idx_str}")
        
        observation, info = env.reset(start_state_id=chosen_maze_idx_str)
        logger.info("\nINITIAL STATE:")
        print(observation)
        
        current_gym_step = 0
        done_gym = False
        truncated_gym = False
        last_game_won_t2 = False # Reset for current game

        while not done_gym and not truncated_gym and current_gym_step < max_steps:
            current_gym_step += 1
            logger.info(f"\n--- GYM STEP {current_gym_step} ---")
            
            # Get available actions (though for this env, it's always VALID_ACTIONS_LIST)
            # We still need to format it for the env.step() method
            raw_action = env.rng.choice(VALID_ACTIONS_LIST) # Use env's rng for consistency if seed is used
            action_for_llm_format = f"<thinking>{raw_action}</thinking>"
            
            logger.info(f"\nRandom agent chooses: {raw_action} (formatted as: {action_for_llm_format})")
            time.sleep(0.5) # Pause to make it watchable
            
            observation, reward, done_gym, truncated_gym, info = env.step(action_for_llm_format)
            
            logger.info("\n" + "="*50)
            logger.info(f"OBSERVATION (Prompt for next step):\n{observation}")
            logger.info(f"REWARD: {reward:.2f}")
            logger.info(f"INFO: {info}")
            
            if done_gym or truncated_gym:
                final_game_message = info.get("final_message", env.final_message)
                event = info.get("game_event", "")
                if event == "found_t2":
                    logger.info(f"\n🎉 GAME OVER: Random Agent found Treasure 2 (T2)! Steps: {env.current_step_internal}")
                    last_game_won_t2 = True
                elif event == "found_t1":
                    logger.info(f"\n⚠️ GAME OVER: Random Agent found Treasure 1 (T1). Steps: {env.current_step_internal}")
                elif event == "max_steps_internal" or info.get("gym_truncated"):
                    logger.info(f"\n⏱️ GAME OVER: Random Agent ran out of time. Steps: {env.current_step_internal}")
                else:
                    logger.info(f"\n💀 GAME OVER: {final_game_message}. Steps: {env.current_step_internal}")
        
        if current_gym_step >= max_steps and not (done_gym or truncated_gym):
            logger.info(f"\n⏱️ GAME OVER: Random Agent reached max demo steps ({current_gym_step}).")
            if not done_gym: last_game_won_t2 = False


        while True:
            play_again = input("\nDo you want to run another random agent game? (y/n): ").strip().lower()
            if play_again in ['y', 'yes']:
                break
            elif play_again in ['n', 'no']:
                continue_playing = False
                break
            else:
                logger.warning("Invalid input. Please enter 'y' or 'n'.")
    
    return last_game_won_t2

def run_batch_evaluation(mode: str, num_trials: int = 10, model_name: Optional[str] = None,
                        max_steps: int = 25, 
                        vllm_device="cuda", gpu_memory_utilization=0.9, vllm_dtype="auto",
                        enable_prefix_caching=True, max_model_len=None,
                        start_maze_idx_filter: Optional[str] = None): # Filter for a specific maze
    """Run batch evaluation for the specified mode and report success rate (finding T2)."""
    logger.info("\n" + "="*80)
    logger.info(f"TWO TREASURES MAZE: BATCH EVALUATION - {mode.upper()} MODE")
    
    temp_env_for_mazes = TwoTreasuresMazeGymEnv(max_steps_gym=max_steps)
    all_possible_maze_indices = temp_env_for_mazes.starting_states # list of strings: "0", "1", ...
    del temp_env_for_mazes
    gc.collect()

    mazes_to_evaluate = []
    if start_maze_idx_filter:
        if start_maze_idx_filter in all_possible_maze_indices:
            mazes_to_evaluate = [start_maze_idx_filter]
            logger.info(f"Batch evaluation will run for the specified maze index: {start_maze_idx_filter}")
        else:
            logger.warning(f"Warning: Specified maze index '{start_maze_idx_filter}' is invalid. Batch will run for ALL mazes.")
            mazes_to_evaluate = all_possible_maze_indices
    else:
        mazes_to_evaluate = all_possible_maze_indices
        logger.info(f"Batch evaluation will run for all {len(mazes_to_evaluate)} maze layouts.")

    logger.info(f"Running {num_trials} trials per maze layout with max {max_steps} steps each.")
    logger.info("="*80)
    
    overall_t2_found_count = 0
    overall_t1_found_count = 0
    overall_timed_out_count = 0
    overall_total_steps_taken = 0
    overall_trials_run = 0

    # {maze_idx_str: {"trials": 0, "t2_successes": 0, "t1_successes":0, "timed_out":0, "total_steps": 0}}
    per_maze_stats = defaultdict(lambda: {"trials": 0, "t2_successes": 0, "t1_successes": 0, "timed_out": 0, "total_steps": 0})
    
    model_loaded = None
    if mode == "llm":
        if not model_name:
            logger.error("Model name must be provided for LLM batch evaluation.")
            return 0,0,0,0, {}
        logger.info(f"Loading model: {model_name}")
        try:
            model_loaded = LLM(
                model=model_name, device=vllm_device, gpu_memory_utilization=gpu_memory_utilization,
                dtype=vllm_dtype, enable_prefix_caching=enable_prefix_caching, max_model_len=max_model_len,
            )
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            return 0,0,0,0, {}
    
    for current_maze_idx_str in mazes_to_evaluate:
        logger.info(f"\n----- Evaluating Maze Index: {current_maze_idx_str} -----")
        maze_t2_success_count = 0
        maze_t1_success_count = 0
        maze_timed_out_count = 0
        maze_total_steps = 0

        for trial in range(num_trials):
            logger.info(f"  Trial {trial+1}/{num_trials} for maze {current_maze_idx_str}")
            gc.collect() # Explicit garbage collection before a trial
            
            current_trial_steps_internal = 0
            trial_found_t2 = False
            trial_found_t1 = False
            trial_timed_out = False # Specifically for max_steps condition

            env_batch = TwoTreasuresMazeGymEnv(max_steps_gym=max_steps, seed=random.randint(0, 2**30) + trial)
            
            if mode == "llm":
                try:
                    messages_batch = [{"role": "system", "content": env_batch.system_prompt}]
                    obs_batch, _ = env_batch.reset(start_state_id=current_maze_idx_str)
                    messages_batch.append({"role": "user", "content": obs_batch})
                    
                    done_batch, truncated_batch = False, False
                    gym_step_this_trial = 0
                    while not done_batch and not truncated_batch and gym_step_this_trial < max_steps:
                        gym_step_this_trial += 1
                        sampling_params_batch = SamplingParams(
                            temperature=0.1, top_p=0.7, max_tokens=50,
                            stop=["</thinking>"], 
                            include_stop_str_in_output=True
                        )
                        response_batch = model_loaded.chat(messages=messages_batch, sampling_params=sampling_params_batch)
                        
                        assistant_content = response_batch[0].outputs[0].text.strip()
                        if not assistant_content.lower().startswith("<thinking>"):
                            assistant_content = "<thinking>" + assistant_content
                        if not assistant_content.lower().endswith("</thinking>"):
                            assistant_content += "</thinking>"
                            
                        messages_batch.append({"role": "assistant", "content": assistant_content})
                        
                        obs_batch, _, done_batch, truncated_batch, info_batch = env_batch.step(assistant_content)
                        messages_batch.append({"role": "user", "content": obs_batch})
                    
                    # After loop, check outcome based on info_batch or env_batch state
                    event = info_batch.get("game_event", "")
                    current_trial_steps_internal = env_batch.current_step_internal
                    if event == "found_t2":
                        trial_found_t2 = True
                    elif event == "found_t1":
                        trial_found_t1 = True
                    elif event == "max_steps_internal" or info_batch.get("gym_truncated", False) or gym_step_this_trial >= max_steps :
                        trial_timed_out = True
                    # If no specific event, it implies timeout due to gym_step_this_trial >= max_steps
                    if not (trial_found_t1 or trial_found_t2):
                        trial_timed_out = True


                except Exception as e:
                    logger.error(f"    Error in LLM trial {trial+1} for maze {current_maze_idx_str}: {e}")
                    trial_timed_out = True # Mark as timeout on error
                    current_trial_steps_internal = env_batch.current_step_internal if env_batch.current_step_internal > 0 else max_steps
            
            elif mode == "random":
                obs_rand, _ = env_batch.reset(start_state_id=current_maze_idx_str)
                done_rand, truncated_rand = False, False
                gym_step_rand = 0
                info_rand = {}
                while not done_rand and not truncated_rand and gym_step_rand < max_steps:
                    gym_step_rand += 1
                    raw_action_rand = env_batch.rng.choice(VALID_ACTIONS_LIST)
                    action_fmt_rand = f"<thinking>Random</thinking><action>{raw_action_rand}</action>"
                    obs_rand, _, done_rand, truncated_rand, info_rand = env_batch.step(action_fmt_rand)

                event_rand = info_rand.get("game_event", "")
                current_trial_steps_internal = env_batch.current_step_internal
                if event_rand == "found_t2":
                    trial_found_t2 = True
                elif event_rand == "found_t1":
                    trial_found_t1 = True
                elif event_rand == "max_steps_internal" or info_rand.get("gym_truncated", False) or gym_step_rand >= max_steps:
                    trial_timed_out = True
                if not (trial_found_t1 or trial_found_t2):
                        trial_timed_out = True
            
            # Update stats for this trial
            if trial_found_t2:
                maze_t2_success_count += 1
                per_maze_stats[current_maze_idx_str]["t2_successes"] += 1
            elif trial_found_t1: # Only count as T1 success if T2 was not found
                maze_t1_success_count += 1
                per_maze_stats[current_maze_idx_str]["t1_successes"] += 1
            
            if trial_timed_out and not (trial_found_t1 or trial_found_t2) : # Count as timed out only if no treasure found
                maze_timed_out_count += 1
                per_maze_stats[current_maze_idx_str]["timed_out"] += 1
            
            maze_total_steps += current_trial_steps_internal
            per_maze_stats[current_maze_idx_str]["trials"] += 1
            per_maze_stats[current_maze_idx_str]["total_steps"] += current_trial_steps_internal
            
            result_str = "T2_FOUND" if trial_found_t2 else ("T1_FOUND" if trial_found_t1 else "TIMED_OUT")
            logger.info(f"    Result: {result_str} in {current_trial_steps_internal} internal steps.")

        # Update overall stats from this maze's batch
        overall_t2_found_count += maze_t2_success_count
        overall_t1_found_count += maze_t1_success_count
        overall_timed_out_count += maze_timed_out_count
        overall_total_steps_taken += maze_total_steps
        overall_trials_run += per_maze_stats[current_maze_idx_str]['trials']

        # Report for current maze
        trials_this_maze = per_maze_stats[current_maze_idx_str]['trials']
        if trials_this_maze > 0:
            avg_steps_for_maze = maze_total_steps / trials_this_maze
            t2_rate_for_maze = maze_t2_success_count / trials_this_maze
            t1_rate_for_maze = maze_t1_success_count / trials_this_maze
            timeout_rate_for_maze = maze_timed_out_count / trials_this_maze
            logger.info(f"  Summary for Maze {current_maze_idx_str}: T2 Rate: {t2_rate_for_maze:.2%}, T1 Rate: {t1_rate_for_maze:.2%}, Timeout Rate: {timeout_rate_for_maze:.2%}, Avg Steps: {avg_steps_for_maze:.2f} (over {trials_this_maze} trials)")
        else:
            logger.info(f"  Summary for Maze {current_maze_idx_str}: No trials completed.")

    overall_t2_success_rate = overall_t2_found_count / overall_trials_run if overall_trials_run > 0 else 0
    overall_t1_success_rate = overall_t1_found_count / overall_trials_run if overall_trials_run > 0 else 0
    overall_timeout_rate = overall_timed_out_count / overall_trials_run if overall_trials_run > 0 else 0
    overall_avg_steps = overall_total_steps_taken / overall_trials_run if overall_trials_run > 0 else 0
    
    logger.info("\n" + "="*80)
    logger.info("OVERALL BATCH EVALUATION SUMMARY (TWO TREASURES MAZE):")
    logger.info(f"Mode: {mode}")
    logger.info(f"Total Trials Run (across all evaluated mazes): {overall_trials_run}")
    logger.info(f"Overall T2 Found (Success): {overall_t2_found_count} ({overall_t2_success_rate:.2%})")
    logger.info(f"Overall T1 Found (Partial): {overall_t1_found_count} ({overall_t1_success_rate:.2%})")
    logger.info(f"Overall Timed Out (No Treasure): {overall_timed_out_count} ({overall_timeout_rate:.2%})")
    logger.info(f"Overall Average Internal Steps: {overall_avg_steps:.2f}")
    logger.info("="*80)

    logger.info("\nPER MAZE LAYOUT SUMMARY:")
    header = f"{'Maze Index':<12} | {'Trials':<7} | {'T2 Found':<10} | {'T1 Found':<10} | {'Timed Out':<10} | {'T2 Rate':<9} | {'Avg Steps':<10}"
    logger.info(header)
    logger.info("-" * len(header))
    for maze_idx_str_sorted, data in sorted(per_maze_stats.items()):
        trials_s = data['trials']
        t2_s = data['t2_successes']
        t1_s = data['t1_successes']
        to_s = data['timed_out']
        total_steps_s = data['total_steps']
        s_rate_t2 = t2_s / trials_s if trials_s > 0 else 0
        avg_s_val = total_steps_s / trials_s if trials_s > 0 else 0
        logger.info(f"{maze_idx_str_sorted:<12} | {trials_s:<7} | {t2_s:<10} | {t1_s:<10} | {to_s:<10} | {s_rate_t2:<9.2%} | {avg_s_val:<10.2f}")
    logger.info("="*80)
    
    if model_loaded is not None:
        del model_loaded
        gc.collect()
    
    return overall_t2_success_rate, overall_avg_steps, dict(per_maze_stats)


def main():
    parser = argparse.ArgumentParser(description="Run a demo of the Two Treasures Maze game")
    parser.add_argument("--mode", type=str, default="human", 
                        choices=["human", "llm", "random", "batch"],
                        help="Mode to run the demo in")
    parser.add_argument("--model", type=str, default="Qwen/Qwen1.5-1.8B-Chat", 
                        help="Model to use for LLM modes (e.g., 'mistralai/Mistral-7B-Instruct-v0.1')")
    parser.add_argument("--max_steps", type=int, default=25,
                        help="Maximum number of steps per episode/trial")
    parser.add_argument("--start_maze", type=str, default=None,
                        help="Specify a starting maze index (e.g., '0', '1'). If not provided or invalid, will be random or prompted in interactive modes. In batch mode, if not specified, all mazes are used.")
    
    # Batch mode specific arguments
    parser.add_argument("--batch_mode_agent", type=str, default="llm",
                        choices=["llm", "random"],
                        help="Agent type to run in batch evaluation (used if --mode=batch)")
    parser.add_argument("--num_trials", type=int, default=10,
                        help="Number of trials to run in batch mode (per maze layout if applicable)")

    # VLLM-specific arguments (copied from imprisoned_demo.py)
    parser.add_argument("--vllm_device", type=str, default="cuda", 
                        help="Device to run VLLM on (e.g., cuda, cuda:0, cpu)")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9,
                        help="Fraction of GPU memory to use")
    parser.add_argument("--vllm_dtype", type=str, default="auto",
                        help="Data type for model weights (auto, float16, bfloat16)")
    parser.add_argument("--vllm_enable_prefix_caching", action="store_true", default=True,
                        help="Enable prefix caching for faster generation")
    parser.add_argument("--no_vllm_enable_prefix_caching", action="store_false", dest="vllm_enable_prefix_caching",
                        help="Disable prefix caching")
    parser.add_argument("--vllm_max_model_len", type=int, default=None,
                        help="Maximum model sequence length")
    
    args = parser.parse_args()

    if args.mode == "human":
        run_human_interactive(
            max_steps=args.max_steps,
            start_maze_idx=args.start_maze
        )
    elif args.mode == "llm":
        run_llm_interactive(
            model_name=args.model,
            max_steps=args.max_steps,
            start_maze_idx=args.start_maze,
            vllm_device=args.vllm_device,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            vllm_dtype=args.vllm_dtype,
            enable_prefix_caching=args.vllm_enable_prefix_caching,
            max_model_len=args.vllm_max_model_len
        )
    elif args.mode == "random":
        run_random_interactive(
            max_steps=args.max_steps,
            start_maze_idx=args.start_maze
        )
    elif args.mode == "batch":
        run_batch_evaluation(
            mode=args.batch_mode_agent, # Use the agent type for batch
            num_trials=args.num_trials,
            model_name=args.model if args.batch_mode_agent == "llm" else None,
            max_steps=args.max_steps,
            start_maze_idx_filter=args.start_maze, # Pass the single maze filter
            vllm_device=args.vllm_device,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            vllm_dtype=args.vllm_dtype,
            enable_prefix_caching=args.vllm_enable_prefix_caching,
            max_model_len=args.vllm_max_model_len
        )

if __name__ == "__main__":
    main() 