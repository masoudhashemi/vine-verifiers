import argparse
import os
import sys
import random
import time
from typing import List, Dict, Any, Optional, Tuple
import gc  # Add garbage collection import
from collections import defaultdict # For batch reporting
import re # Add re import

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import verifiers as vf
from verifiers.utils.imports import LLM, SamplingParams  # type: ignore
from verifiers.envs.imprisoned.imprisoned_multistep_env import ImprisonedMultistepEnv
from verifiers.envs.imprisoned.imprisoned_env import ImprisonedEnv
from verifiers.envs.imprisoned.qlearning_agent import QLearningAgent

# Define the system prompt once to ensure consistency between demo and training
TRAINING_SYSTEM_PROMPT = """You are playing a text-based adventure game where you are trying to escape from prison.
You will be given a description of the current state you are in, and a list of actions you can take.
Choose ONE action from the available actions list. You must select an action that is explicitly listed.

Your response must be in the following format:
<thinking>
[Your thoughts on what to do next]
</thinking>
<action>
[The exact action you choose from the available actions]
</action>

Do not include any other text in your response. You only should think about the action and then output the action in the format specified above.
Only output one action and finish there.

Remember: Only choose actions that are explicitly listed in the "Available actions" section.
"""

class TrainingConsistentMultistepEnv(ImprisonedMultistepEnv):
    """
    A version of ImprisonedMultistepEnv with error handling that matches
    the ImprisonedGymEnv used in training.
    """
    
    def _parse_action(self, action_text: str) -> Optional[str]:
        """Extract the action from the XML format using regex for robustness, matching ImprisonedGymEnv."""
        match = re.search(r"<action>(.*?)</action>", action_text, re.DOTALL | re.IGNORECASE)
        if match:
            action = match.group(1).strip()
            # Match ImprisonedGymEnv's behavior of splitting on ":"
            if ":" in action:
                action = action.split(":", 1)[0].strip()
            return action
        return None
    
    def env_response(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, str]:
        """Generate the environment's response with training-consistent error handling."""
        # Find the last user message to identify the current state
        user_messages = [msg for msg in messages if msg["role"] == "user"]
        
        # First interaction or reset needed
        if len(user_messages) <= 1 or not hasattr(self, 'current_state') or self.current_state is None:
            # Handle initial state/reset like the parent class
            return super().env_response(messages, **kwargs)
        
        # Get the last assistant message
        assistant_messages = [msg for msg in messages if msg["role"] == "assistant"]
        if not assistant_messages:
            # First turn, no assistant message yet
            return super().env_response(messages, **kwargs)
        
        # Get the last assistant message
        user_message = assistant_messages[-1]["content"]
        user_action = self._parse_action(user_message)
        
        # Check if the format was correct - MATCH TRAINING ERROR FORMAT
        if user_action is None:
            error_msg = (
                "❌ ERROR: You didn't use the correct format for your response.\n"
                "You MUST use correct tags for thinking and action:\n"
                "<thinking>Your reasoning</thinking>\n"
                "<action>a valid action from the list</action>\n"
                "Let's try again with the correct format.\n"
                f"<state>\n{self.render()}\n</state>\n\n"
                "What do you do next?"
            )
            return {"role": "user", "content": error_msg}
        
        # Get available actions for the current state
        available_actions = self._get_available_actions(self.current_state)
        
        # Match the user's action to an available action
        matched_action = self._match_action(user_action, available_actions)
        
        # If the action was invalid, inform the user with the available actions - MATCH TRAINING ERROR FORMAT
        if not matched_action:
            action_text = self._format_available_actions()
            error_msg = (
                f"❌ ERROR: Invalid action selected\n\n"
                f"You chose: '{user_action}'\n\n"
                f"{action_text}\n\n"
                f"Please select EXACTLY one of the available actions listed above.\n"
                f"Remember to use the format:\n"
                f"<thinking>Your reasoning</thinking>\n"
                f"<action>a valid action from the list</action>"
            )
            return {"role": "user", "content": error_msg}
        
        # For valid actions, use the parent implementation
        return super().env_response(messages, **kwargs)
    
    def _format_available_actions(self) -> str:
        """Format available actions with descriptions like ImprisonedGymEnv."""
        available_actions = self._get_available_actions(self.current_state)
        action_text = "\nAvailable actions:\n"
        for action, details in available_actions.items():
            action_text += f"- {action}: {details.get('description', '')}\n"
        return action_text

def print_messages(messages: List[Dict[str, str]], skip_system: bool = False, skip_few_shot: bool = False):
    """Print the conversation messages in a readable format."""
    start_idx = 0
    if skip_system:
        start_idx = 1
    
    if skip_few_shot and len(messages) > 2:
        # Find where the actual conversation starts (after system and few-shot)
        for i, msg in enumerate(messages):
            if msg["role"] == "user" and i > start_idx:
                start_idx = i
                break
    
    for i, msg in enumerate(messages[start_idx:], start=start_idx):
        role = msg["role"].upper()
        print(f"\n[{role}]:")
        print(msg["content"])

def _get_start_state(env: ImprisonedMultistepEnv, preferred_start_state: Optional[str]) -> str:
    """Helper to determine the start state, prompting if necessary."""
    if preferred_start_state and preferred_start_state in env.starting_states:
        return preferred_start_state
    elif preferred_start_state:
        print(f"Warning: Provided start state '{preferred_start_state}' is not valid. Please choose from the list or let the system choose randomly.")

    print("\nAvailable starting states:")
    for i, state_id in enumerate(env.starting_states):
        print(f"{i+1}. {state_id} - {env.states[state_id].get('description', 'No description').splitlines()[0]}")
    
    while True:
        try:
            choice = input(f"Choose a starting state by number (1-{len(env.starting_states)}), or press Enter for random: ").strip()
            if not choice:
                return random.choice(env.starting_states)
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(env.starting_states):
                return env.starting_states[choice_idx]
            else:
                print("Invalid choice. Please pick a valid number.")
        except ValueError:
            print("Invalid input. Enter a number or press Enter.")

def run_human_interactive(env_config_path: Optional[str] = None, max_steps: int = 20, start_state: Optional[str] = None):
    """Run an interactive game where a human player makes all decisions."""
    continue_playing = True
    
    while continue_playing:
        print("\n" + "="*80)
        print("IMPRISONED: HUMAN INTERACTIVE MODE")
        print("="*80)
        
        # Initialize the environment
        env = ImprisonedMultistepEnv(
            config_path=env_config_path,
            max_steps=max_steps,
            sleep_time=0.0
        )
        
        chosen_start_state = _get_start_state(env, start_state)
        print(f"Starting game from: {chosen_start_state}")
        observation = env.reset(start_state_id=chosen_start_state)
        print("\nINITIAL STATE:")
        print(observation)
        
        step = 0
        done = False
        
        while not done and step < max_steps:
            step += 1
            print(f"\n--- STEP {step} ---")
            
            available_actions = env._get_available_actions(env.current_state)
            if not available_actions:
                print("No actions available. Game over!")
                break
            
            print("\nAvailable actions:")
            for i, (action, details) in enumerate(available_actions.items(), 1):
                print(f"{i}. {action}: {details.get('description', 'No description')}")
            
            while True:
                try:
                    choice = input("\nChoose an action (number): ")
                    choice_idx = int(choice) - 1
                    
                    if 0 <= choice_idx < len(available_actions):
                        action = list(available_actions.keys())[choice_idx]
                        break
                    else:
                        print("Invalid choice. Please pick a valid number.")
                except ValueError:
                    print("Invalid input. Enter a number corresponding to an action.")
            
            observation, reward, done, truncated, info = env.step_env(action)
            
            print("\n" + "="*50)
            print(observation)
            
            if done:
                if env.current_state == "escape_success":
                    print("\n🎉 GAME OVER: You have successfully escaped! Congratulations!")
                else:
                    print(f"\n💀 GAME OVER: Your escape attempt has failed in {env.current_state}.")
        
        if step >= max_steps and not done:
            print(f"\n⏱️ GAME OVER: You ran out of time to escape after {step} attempts.")
        
        while True:
            play_again = input("\nDo you want to play again? (y/n): ").strip().lower()
            if play_again in ['y', 'yes']:
                break
            elif play_again in ['n', 'no']:
                continue_playing = False
                break
            else:
                print("Invalid input. Please enter 'y' or 'n'.")
    
    return env.current_state == "escape_success"

def run_llm_interactive(model_name: str, env_config_path: Optional[str] = None, 
                        max_steps: int = 20, start_state: Optional[str] = None,
                        vllm_device="cuda", gpu_memory_utilization=0.9, 
                        vllm_dtype="auto", enable_prefix_caching=True, 
                        max_model_len=None):
    """Run an interactive game where an LLM makes all decisions."""
    print(f"Loading model: {model_name}")
    model = LLM(
        model=model_name,
        device=vllm_device,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=vllm_dtype,
        enable_prefix_caching=enable_prefix_caching,
        max_model_len=max_model_len,
    )
    
    continue_playing = True
    
    while continue_playing:
        print("\n" + "="*80)
        print(f"IMPRISONED: LLM INTERACTIVE MODE (Model: {model_name})")
        print("="*80)
        
        env = TrainingConsistentMultistepEnv(
            config_path=env_config_path,
            max_steps=max_steps,
            sleep_time=0.0,
            system_prompt=TRAINING_SYSTEM_PROMPT
        )
        
        chosen_start_state = _get_start_state(env, start_state)
        print(f"Starting game from: {chosen_start_state}")
        
        messages = [{"role": "system", "content": env.system_prompt}]
        if env.few_shot:
            messages.extend(env.few_shot)
        
        observation = env.reset(start_state_id=chosen_start_state)
        initial_prompt = (
            f"<state>You are in the following state:\\n{observation}\\n</state>\\n\\n"
            "Remember to use <thinking></thinking> and <action></action> tags in your response.\n"
        )
        messages.append({"role": "user", "content": initial_prompt})
        
        print("\nINITIAL STATE:")
        print_messages(messages, skip_system=True)
        
        step = 0
        completed = False
        
        while not completed and step < max_steps:
            step += 1
            print(f"\n--- STEP {step} ---")
            
            # Get model response
            print("Model thinking...")
            sampling_params = SamplingParams(
                temperature=0.1, top_p=0.7, max_tokens=256,
                stop=["</action>", "/Action"], include_stop_str_in_output=True
            )
            
            response = model.chat(messages=messages, sampling_params=sampling_params)
            
            # Add model response to messages
            assistant_msg = {"role": "assistant", "content": response[0].outputs[0].text}
            messages.append(assistant_msg)
            
            # Print the model's response
            print("\nMODEL RESPONSE:")
            print(assistant_msg["content"])
            
            # Check if we're done
            if env.is_completed(messages):
                completed = True
                print("\nConversation completed!")
            else:
                # Get environment response
                env_msg = env.env_response(messages)
                messages.append(env_msg)

                # Print the environment response
                print("\nENVIRONMENT RESPONSE:")
                print(env_msg["content"])
        
        # Print final result
        print("\n" + "="*80)
        print("FINAL CONVERSATION:")
        print_messages(messages, skip_system=True, skip_few_shot=True)
        print("="*80)
        
        # Check if the agent successfully escaped
        success = env.current_state == "escape_success"
        if success:
            print("\n🎉 GAME OVER: The LLM successfully escaped! Congratulations!")
        else:
            print("\n💀 GAME OVER: The LLM failed to escape.")
        
        while True:
            play_again = input("\nDo you want to run another LLM game? (y/n): ").strip().lower()
            if play_again in ['y', 'yes']:
                break
            elif play_again in ['n', 'no']:
                continue_playing = False
                break
            else:
                print("Invalid input. Please enter 'y' or 'n'.")
    
    return success

def run_qlearning_interactive(q_table_path: str = "q_table.pkl", 
                            env_config_path: Optional[str] = None, 
                            max_steps: int = 20, start_state: Optional[str] = None):
    """Run an interactive game where a Q-learning agent makes all decisions."""
    continue_playing = True
    
    while continue_playing:
        print("\n" + "="*80)
        print("IMPRISONED: Q-LEARNING INTERACTIVE MODE")
        print("="*80)
        
        # Initialize the base ImprisonedEnv for QLearningAgent
        # QLearningAgent expects the simpler ImprisonedEnv, not MultistepEnv for its internal logic
        base_env_for_q_agent = ImprisonedEnv(config_path=env_config_path if env_config_path else None)
        agent = QLearningAgent(base_env_for_q_agent)

        # We still need a MultistepEnv for interaction if we want to display descriptions etc.
        # Or we can simplify the Q-learning display if not needed.
        # For now, let's use the base_env_for_q_agent for Q-learning state setting.
        
        # Allow choosing start state for the Q-learning agent's play_game method
        # The QLearningAgent.play_game internally calls env.reset()
        # We need to ensure that reset can take a start_state_id
        # This means ImprisonedEnv.reset() also needs to support start_state_id
        # Let's assume ImprisonedEnv.reset() has been updated or QLearningAgent can handle it.
        # If not, QLearningAgent would need adjustment, or we pass the start state to play_game.

        # For a consistent experience, let's create a dummy MultistepEnv just to use _get_start_state
        # This is a bit clunky but avoids modifying QLearningAgent for now.
        temp_display_env = ImprisonedMultistepEnv(config_path=env_config_path, max_steps=max_steps)
        chosen_start_state = _get_start_state(temp_display_env, start_state)
        print(f"Starting Q-learning game from: {chosen_start_state}")
        
        try:
            agent.load_q_table(q_table_path)
        except FileNotFoundError:
            print(f"Q-table file {q_table_path} not found. Starting with an empty Q-table.")
        
        # The QLearningAgent's play_game method needs to be able to start from a specific state.
        # Let's assume agent.play_game(start_state_id=chosen_start_state, ...)

        reward, steps = agent.play_game(max_steps=max_steps, render=True, start_state_id=chosen_start_state)
        
        success = reward > 0
        
        if success:
            print("\n🎉 GAME OVER: The Q-learning agent successfully escaped! Congratulations!")
        else:
            print("\n💀 GAME OVER: The Q-learning agent failed to escape.")
        
        while True:
            play_again = input("\nDo you want to run another Q-learning game? (y/n): ").strip().lower()
            if play_again in ['y', 'yes']:
                break
            elif play_again in ['n', 'no']:
                continue_playing = False
                break
            else:
                print("Invalid input. Please enter 'y' or 'n'.")
    
    return success

def run_random_interactive(env_config_path: Optional[str] = None, max_steps: int = 20, start_state: Optional[str] = None):
    """Run an interactive game where actions are chosen randomly."""
    continue_playing = True
    
    while continue_playing:
        print("\n" + "="*80)
        print("IMPRISONED: RANDOM AGENT INTERACTIVE MODE")
        print("="*80)
        
        env = TrainingConsistentMultistepEnv(
            config_path=env_config_path,
            max_steps=max_steps,
            sleep_time=0.0,
            system_prompt=TRAINING_SYSTEM_PROMPT
        )
        
        chosen_start_state = _get_start_state(env, start_state)
        print(f"Starting game from: {chosen_start_state}")
        observation = env.reset(start_state_id=chosen_start_state)
        print("\nINITIAL STATE:")
        print(observation)
        
        step = 0
        done = False
        
        while not done and step < max_steps:
            step += 1
            print(f"\n--- STEP {step} ---")
            
            available_actions = env._get_available_actions(env.current_state)
            if not available_actions:
                print("No actions available. Game over!")
                break
            
            print("\nAvailable actions:")
            for i, (action, details) in enumerate(available_actions.items(), 1):
                print(f"{i}. {action}: {details.get('description', 'No description')}")
            
            action = random.choice(list(available_actions.keys()))
            print(f"\nRandom agent chooses: {action}")
            time.sleep(1)
            observation, reward, done, truncated, info = env.step_env(action)
            
            print("\n" + "="*50)
            print(observation)
            
            if done:
                if env.current_state == "escape_success":
                    print("\n🎉 GAME OVER: Random agent has successfully escaped! Congratulations!")
                else:
                    print(f"\n💀 GAME OVER: Random agent's escape attempt has failed in {env.current_state}.")
        
        if step >= max_steps and not done:
            print(f"\n⏱️ GAME OVER: Random agent ran out of time to escape after {step} attempts.")
        
        while True:
            play_again = input("\nDo you want to run another random agent game? (y/n): ").strip().lower()
            if play_again in ['y', 'yes']:
                break
            elif play_again in ['n', 'no']:
                continue_playing = False
                break
            else:
                print("Invalid input. Please enter 'y' or 'n'.")
    
    return env.current_state == "escape_success"

def run_qlearning_llm_interactive(model_name: str, q_table_path: str = "q_table.pkl",
                                env_config_path: Optional[str] = None, max_steps: int = 20,
                                start_state: Optional[str] = None, top_k: int = 3, 
                                vllm_device="cuda", gpu_memory_utilization=0.9,
                                vllm_dtype="auto", enable_prefix_caching=True, 
                                max_model_len=None):
    """Run an interactive game where Q-learning suggests top-k actions and LLM chooses."""
    print(f"Loading model: {model_name}")
    model_loaded = LLM( # Renamed model to model_loaded to avoid conflict if any
        model=model_name, device=vllm_device, gpu_memory_utilization=gpu_memory_utilization,
        dtype=vllm_dtype, enable_prefix_caching=enable_prefix_caching, max_model_len=max_model_len,
    )
    
    continue_playing = True
    
    while continue_playing:
        print("\n" + "="*80)
        print(f"IMPRISONED: Q-LEARNING + LLM INTERACTIVE MODE (Model: {model_name}, Top-{top_k})")
        print("="*80)
        
        # Base env for Q-agent
        gym_env_qllm = ImprisonedEnv(config_path=env_config_path if env_config_path else None) # Unique name
        agent_qllm = QLearningAgent(gym_env_qllm) # Unique name
        
        # Multistep env for interaction
        env_qllm = TrainingConsistentMultistepEnv( # Unique name
            config_path=env_config_path, max_steps=max_steps, sleep_time=0.0,
            system_prompt=TRAINING_SYSTEM_PROMPT
        )
        
        chosen_start_state = _get_start_state(env_qllm, start_state)
        print(f"Starting game from: {chosen_start_state}")

        try:
            agent_qllm.load_q_table(q_table_path)
        except FileNotFoundError:
            print(f"Q-table file {q_table_path} not found. Starting with an empty Q-table.")
        
        q_system_prompt = env_qllm.system_prompt + "\n\nYou will be given suggestions from a Q-learning agent about which actions might be best. Consider these suggestions, but make your own decision based on your understanding of the game."
        messages_qllm = [{"role": "system", "content": q_system_prompt}] # Unique name
        if env_qllm.few_shot:
            messages_qllm.extend(env_qllm.few_shot)
        
        observation_qllm = env_qllm.reset(start_state_id=chosen_start_state)
        gym_env_qllm.current_state = env_qllm.current_state 
        
        q_suggestions = ""
        available_actions_dict = env_qllm._get_available_actions(env_qllm.current_state)
        if available_actions_dict:
            q_values = [(action, agent_qllm.get_q_value(env_qllm.current_state, action)) for action in available_actions_dict]
            q_values.sort(key=lambda x: x[1], reverse=True)
            q_suggestions = "\n\nQ-learning agent suggestions (ranked by expected value):\n"
            for i, (act, q_val) in enumerate(q_values[:top_k], 1):
                q_suggestions += f"{i}. {act}: Q-value = {q_val:.2f}\n"
        
        initial_prompt = (
            f"<state>You are in the following state:\\n{observation_qllm}{q_suggestions}\\n</state>\\n\\n"
            "Remember to use <thinking></thinking> and <action></action> tags in your response.\n"
        )
        messages_qllm.append({"role": "user", "content": initial_prompt})
        
        print("\nINITIAL STATE:")
        print_messages(messages_qllm, skip_system=True)
        
        step_qllm = 0 # Unique name
        completed_qllm = False # Unique name
        success_this_game = False # Variable for game outcome

        while not completed_qllm and step_qllm < max_steps:
            step_qllm += 1
            print(f"\n--- STEP {step_qllm} ---")
            
            # Get model response
            print("Model thinking...")
            sampling_params_qllm = SamplingParams(
                temperature=0.1, top_p=0.7, max_tokens=256,
                stop=["</action>", "/Action"], include_stop_str_in_output=True
            )
            response = model_loaded.chat(messages=messages_qllm, sampling_params=sampling_params_qllm) # Use model_loaded
            
            # Add model response to messages
            assistant_msg = {"role": "assistant", "content": response[0].outputs[0].text}
            messages_qllm.append(assistant_msg)
            
            # Print the model's response
            print("\nMODEL RESPONSE:")
            print(assistant_msg["content"])
            
            # Check if we're done
            if env_qllm.is_completed(messages_qllm):
                completed_qllm = True
                print("\nConversation completed!")
            else:
                # Get environment response
                env_msg = env_qllm.env_response(messages_qllm)
                
                # If we're continuing, add Q-learning suggestions to the next message
                if not env_qllm.is_completed([*messages_qllm, env_msg]):
                    gym_env_qllm.current_state = env_qllm.current_state 
                    available_actions_dict_next = env_qllm._get_available_actions(env_qllm.current_state)
                    q_suggestions_next = ""
                    if available_actions_dict_next:
                        q_values_next = [(action, agent_qllm.get_q_value(env_qllm.current_state, action)) for action in available_actions_dict_next]
                        q_values_next.sort(key=lambda x: x[1], reverse=True)
                        q_suggestions_next = "\n\nQ-learning agent suggestions (ranked by expected value):\n"
                        for i, (act_n, q_val_n) in enumerate(q_values_next[:top_k], 1):
                            q_suggestions_next += f"{i}. {act_n}: Q-value = {q_val_n:.2f}\n"
                        env_msg["content"] += q_suggestions_next
                
                messages_qllm.append(env_msg)

                # Print the environment response
                print("\nENVIRONMENT RESPONSE:")
                print(env_msg["content"])
        
        # Print final result
        print("\n" + "="*80)
        print("FINAL CONVERSATION:")
        print_messages(messages_qllm, skip_system=True, skip_few_shot=True)
        print("="*80)
        
        # Check if the agent successfully escaped
        success_this_game = env_qllm.current_state == "escape_success"
        
        if success_this_game:
            print("\n🎉 GAME OVER: The LLM successfully escaped! Congratulations!")
        else:
            print("\n💀 GAME OVER: The LLM failed to escape.")

        while True:
            play_again = input("\nDo you want to run another Q-learning + LLM game? (y/n): ").strip().lower()
            if play_again in ['y', 'yes']:
                break
            elif play_again in ['n', 'no']:
                continue_playing = False
                break
            else:
                print("Invalid input. Please enter 'y' or 'n'.")
    
    return success_this_game # Return the outcome of the last game played in the loop

def run_batch_evaluation(mode: str, num_trials: int = 10, model_name: str = None,
                        q_table_path: str = "q_table.pkl", env_config_path: Optional[str] = None,
                        max_steps: int = 20, top_k: int = 3, vllm_device="cuda",
                        gpu_memory_utilization=0.9, vllm_dtype="auto",
                        enable_prefix_caching=True, max_model_len=None,
                        start_state: Optional[str] = None): # Added start_state
    """Run batch evaluation for the specified mode and report success rate."""
    print("\n" + "="*80)
    print(f"IMPRISONED: BATCH EVALUATION - {mode.upper()} MODE")
    # num_trials now means trials *per starting state* if multiple start states are used.
    # If a specific start_state is given, it's num_trials for that state.
    # If no start_state is given, it cycles through all available start states.
    
    # Initialize a temporary env to get starting_states list
    temp_env_for_states = ImprisonedMultistepEnv(config_path=env_config_path, max_steps=max_steps)
    all_possible_start_states = temp_env_for_states.starting_states
    del temp_env_for_states # free memory
    gc.collect()

    states_to_evaluate = []
    if start_state: # User specified a single start state for batch
        if start_state in all_possible_start_states:
            states_to_evaluate = [start_state]
            print(f"Batch evaluation will run for the specified starting state: {start_state}")
        else:
            print(f"Warning: Specified start state '{start_state}' is invalid. Batch will run for all starting states.")
            states_to_evaluate = all_possible_start_states
    else: # No specific start state, cycle through all
        states_to_evaluate = all_possible_start_states
        print(f"Batch evaluation will run for all {len(states_to_evaluate)} starting states.")

    print(f"Running {num_trials} trials per starting state with max {max_steps} steps each")
    print("="*80)
    
    overall_success_count = 0
    overall_total_steps = 0
    overall_trials_run = 0

    # For grouped reporting
    # {state_id: {"trials": 0, "successes": 0, "total_steps": 0}}
    per_start_state_stats = defaultdict(lambda: {"trials": 0, "successes": 0, "total_steps": 0})
    
    model_loaded = None
    if mode in ["llm", "qlearning_llm"]:
        print(f"Loading model: {model_name}")
        try:
            model_loaded = LLM(
                model=model_name, device=vllm_device, gpu_memory_utilization=gpu_memory_utilization,
                dtype=vllm_dtype, enable_prefix_caching=enable_prefix_caching, max_model_len=max_model_len,
            )
        except Exception as e:
            print(f"Error initializing model: {e}")
            return 0, 0, {} # Return empty dict for per_start_state_stats
    
    for current_start_state_id in states_to_evaluate:
        print(f"\n----- Evaluating Start State: {current_start_state_id} -----")
        start_state_success_count = 0
        start_state_total_steps = 0

        for trial in range(num_trials):
            print(f"  Trial {trial+1}/{num_trials} for state {current_start_state_id}")
            gc.collect()
            
            current_trial_steps = 0
            success_this_trial = False

            if mode == "human": # Should not happen due to arg parsing, but good check
                print("Human mode not available for batch evaluation.")
                return 0,0, {}

            elif mode == "llm":
                try:
                    env = TrainingConsistentMultistepEnv(
                        config_path=env_config_path, max_steps=max_steps, sleep_time=0.0,
                        system_prompt=TRAINING_SYSTEM_PROMPT
                    )
                    messages = [{"role": "system", "content": env.system_prompt}]
                    if env.few_shot: messages.extend(env.few_shot)
                    
                    observation = env.reset(start_state_id=current_start_state_id) # Use current start state
                    initial_prompt = (
                        f"<state>You are in the following state:\\n{observation}\\n</state>\\n\\n"
                        "Remember to use <thinking></thinking> and <action></action> tags in your response.\n"
                    )
                    messages.append({"role": "user", "content": initial_prompt})
                    
                    step_this_trial = 0
                    while step_this_trial < max_steps: # Iterate up to max_steps
                        step_this_trial += 1
                        sampling_params = SamplingParams(
                            temperature=0.1, top_p=0.7, max_tokens=256,
                            stop=["</action>", "/Action"], include_stop_str_in_output=True
                        )
                        response = model_loaded.chat(messages=messages, sampling_params=sampling_params)
                        assistant_msg = {"role": "assistant", "content": response[0].outputs[0].text}
                        messages.append(assistant_msg)
                        
                        # Get environment response to the assistant's action
                        env_msg = env.env_response(messages)
                        messages.append(env_msg)
                        
                        # Now check if the game is completed based on the environment's response
                        if env.is_completed(messages):
                            break # Game ended based on its own rules (win/loss)
                    
                    # After the loop (either max_steps reached or game completed earlier),
                    # check the final state for success.
                    success_this_trial = env.current_state == "escape_success"
                    current_trial_steps = step_this_trial 
                except Exception as e:
                    print(f"    Error in trial {trial+1} for state {current_start_state_id}: {e}")
                    success_this_trial = False # Ensure it's marked as failure on error
                    current_trial_steps = step_this_trial if step_this_trial > 0 else max_steps # Record steps taken or max_steps if error was early
            
            elif mode == "qlearning":
                gym_env_q = ImprisonedEnv(config_path=env_config_path if env_config_path else None)
                agent_q = QLearningAgent(gym_env_q)
                try:
                    agent_q.load_q_table(q_table_path)
                except FileNotFoundError:
                    print(f"    Q-table file {q_table_path} not found for trial {trial+1}, state {current_start_state_id}. Starting empty.")
                
                reward_q, steps_q = agent_q.play_game(max_steps=max_steps, render=False, start_state_id=current_start_state_id)
                success_this_trial = reward_q > 0
                current_trial_steps = steps_q
            
            elif mode == "random":
                env_random = TrainingConsistentMultistepEnv(
                    config_path=env_config_path, max_steps=max_steps, sleep_time=0.0,
                    system_prompt=TRAINING_SYSTEM_PROMPT
                )
                env_random.reset(start_state_id=current_start_state_id) # Use current start state
                step_random = 0
                done_random = False
                while not done_random and step_random < max_steps:
                    step_random += 1
                    available_actions_random = env_random._get_available_actions(env_random.current_state)
                    if not available_actions_random: break
                    action_random = random.choice(list(available_actions_random.keys()))
                    _, _, done_random, _, _ = env_random.step_env(action_random)
                success_this_trial = env_random.current_state == "escape_success"
                current_trial_steps = step_random

            elif mode == "qlearning_llm":
                try:
                    gym_env_qllm = ImprisonedEnv(config_path=env_config_path if env_config_path else None)
                    agent_qllm = QLearningAgent(gym_env_qllm)
                    env_qllm = TrainingConsistentMultistepEnv(
                        config_path=env_config_path, max_steps=max_steps, sleep_time=0.0,
                        system_prompt=TRAINING_SYSTEM_PROMPT
                    )
                    try:
                        agent_qllm.load_q_table(q_table_path)
                    except FileNotFoundError:
                         print(f"    Q-table file {q_table_path} not found for trial {trial+1}, state {current_start_state_id}. Starting empty.")

                    q_system_prompt_batch = env_qllm.system_prompt + "\n\nYou will be given suggestions from a Q-learning agent about which actions might be best. Consider these suggestions, but make your own decision based on your understanding of the game."
                    messages_qllm = [{"role": "system", "content": q_system_prompt_batch}]
                    if env_qllm.few_shot: messages_qllm.extend(env_qllm.few_shot)
                    
                    observation_qllm = env_qllm.reset(start_state_id=current_start_state_id) # Use current start state
                    gym_env_qllm.current_state = env_qllm.current_state # Sync
                    
                    q_suggestions_batch = ""
                    available_actions_qllm = env_qllm._get_available_actions(env_qllm.current_state)
                    if available_actions_qllm:
                        q_values_batch = [(act, agent_qllm.get_q_value(env_qllm.current_state, act)) for act in available_actions_qllm]
                        q_values_batch.sort(key=lambda x: x[1], reverse=True)
                        q_suggestions_batch = "\n\nQ-learning agent suggestions (ranked by expected value):\n"
                        for i, (act_b, q_val_b) in enumerate(q_values_batch[:top_k], 1):
                            q_suggestions_batch += f"{i}. {act_b}: Q-value = {q_val_b:.2f}\n"
                    
                    initial_prompt_qllm = (
                        f"<state>You are in the following state:\\n{observation_qllm}{q_suggestions_batch}\\n</state>\\n\\n"
                        "Remember to use <thinking></thinking> and <action></action> tags in your response.\n"
                    )
                    messages_qllm.append({"role": "user", "content": initial_prompt_qllm})
                    
                    step_qllm = 0
                    while step_qllm < max_steps: # Iterate up to max_steps
                        step_qllm += 1
                        sampling_params_qllm = SamplingParams(
                            temperature=0.1, top_p=0.7, max_tokens=256,
                            stop=["</action>", "/Action"], include_stop_str_in_output=True
                        )
                        response_qllm = model_loaded.chat(messages=messages_qllm, sampling_params=sampling_params_qllm)
                        assistant_msg_qllm = {"role": "assistant", "content": response_qllm[0].outputs[0].text}
                        messages_qllm.append(assistant_msg_qllm)
                        
                        # Get environment response
                        env_msg_qllm = env_qllm.env_response(messages_qllm)
                        
                        # Check if the game is completed based on the current messages + env_msg_qllm
                        # Need to create a temporary list to pass to is_completed
                        temp_messages_for_completion_check = messages_qllm + [env_msg_qllm]
                        game_just_ended = env_qllm.is_completed(temp_messages_for_completion_check)

                        if not game_just_ended: # Only add suggestions if the game will continue
                            gym_env_qllm.current_state = env_qllm.current_state # Sync for Q-values
                            available_actions_next_qllm = env_qllm._get_available_actions(env_qllm.current_state)
                            q_suggestions_next_qllm = ""
                            if available_actions_next_qllm:
                                q_values_next_qllm = [(act_n, agent_qllm.get_q_value(env_qllm.current_state, act_n)) for act_n in available_actions_next_qllm]
                                q_values_next_qllm.sort(key=lambda x: x[1], reverse=True)
                                q_suggestions_next_qllm = "\n\nQ-learning agent suggestions (ranked by expected value):\n"
                                for i, (act_n_b, q_val_n_b) in enumerate(q_values_next_qllm[:top_k], 1):
                                    q_suggestions_next_qllm += f"{i}. {act_n_b}: Q-value = {q_val_n_b:.2f}\n"
                                env_msg_qllm["content"] += q_suggestions_next_qllm
                        
                        messages_qllm.append(env_msg_qllm) # Now add the potentially modified env_msg_qllm
                        
                        if game_just_ended: # Break if the game ended after this environment response
                            break
                        
                    # After the loop, check success
                    success_this_trial = env_qllm.current_state == "escape_success"
                    current_trial_steps = step_qllm
                except Exception as e:
                    print(f"    Error in trial {trial+1} for state {current_start_state_id}: {e}")
                    success_this_trial = False # Ensure it's marked as failure on error
                    current_trial_steps = step_qllm if step_qllm > 0 else max_steps # Record steps taken or max_steps if error was early
            
            # Update stats for this trial
            if success_this_trial:
                start_state_success_count += 1
                per_start_state_stats[current_start_state_id]["successes"] += 1
            
            start_state_total_steps += current_trial_steps
            per_start_state_stats[current_start_state_id]["trials"] += 1
            per_start_state_stats[current_start_state_id]["total_steps"] += current_trial_steps
            
            print(f"    Result: {'✓ SUCCESS' if success_this_trial else '✗ FAILURE'} in {current_trial_steps} steps")

        # Update overall stats from this start state's batch
        overall_success_count += start_state_success_count
        overall_total_steps += start_state_total_steps
        overall_trials_run += per_start_state_stats[current_start_state_id]['trials'] # Use actual trials run for this state

        # Report for current start state
        start_state_trials_completed = per_start_state_stats[current_start_state_id]['trials']
        if start_state_trials_completed > 0:
            avg_steps_for_state = start_state_total_steps / start_state_trials_completed
            success_rate_for_state = start_state_success_count / start_state_trials_completed
            print(f"  Summary for {current_start_state_id}: Success Rate: {success_rate_for_state:.2%}, Avg Steps: {avg_steps_for_state:.2f} (over {start_state_trials_completed} trials)")
        else:
            print(f"  Summary for {current_start_state_id}: No trials completed.")


    # Calculate overall success rate and average steps
    overall_success_rate = overall_success_count / overall_trials_run if overall_trials_run > 0 else 0
    overall_avg_steps = overall_total_steps / overall_trials_run if overall_trials_run > 0 else 0
    
    print("\n" + "="*80)
    print(f"OVERALL BATCH EVALUATION SUMMARY:")
    print(f"Mode: {mode}")
    print(f"Total Trials Run (across all evaluated start states): {overall_trials_run}")
    print(f"Overall Success count: {overall_success_count}")
    print(f"Overall Success rate: {overall_success_rate:.2%}")
    print(f"Overall Average steps: {overall_avg_steps:.2f}")
    print("="*80)

    print("\nPER STARTING STATE SUMMARY:")
    print(f"{'Starting State':<30} | {'Trials':<7} | {'Successes':<9} | {'Success Rate':<12} | {'Avg Steps':<10}")
    print("-" * 80)
    for state_id, data in sorted(per_start_state_stats.items()):
        trials = data['trials']
        successes = data['successes']
        total_s = data['total_steps']
        s_rate = successes / trials if trials > 0 else 0
        avg_s = total_s / trials if trials > 0 else 0
        print(f"{state_id:<30} | {trials:<7} | {successes:<9} | {s_rate:<12.2%} | {avg_s:<10.2f}")
    print("="*80)
    
    if model_loaded is not None:
        del model_loaded
        gc.collect()
    
    return overall_success_rate, overall_avg_steps, dict(per_start_state_stats)

def main():
    parser = argparse.ArgumentParser(description="Run a demo of the Imprisoned text adventure game")
    parser.add_argument("--mode", type=str, default="human", 
                        choices=["human", "llm", "qlearning", "random", "qlearning_llm", "batch"],
                        help="Mode to run the demo in")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", 
                        help="Model to use for LLM modes")
    parser.add_argument("--q_table", type=str, default="q_table.pkl",
                        help="Path to the Q-table file for Q-learning modes")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to the environment configuration file")
    parser.add_argument("--max_steps", type=int, default=20,
                        help="Maximum number of steps to run")
    parser.add_argument("--start_state", type=str, default=None,
                        help="Specify a starting state ID. If not provided or invalid, will be random or prompted in interactive modes. In batch mode, if not specified, all starting states are used.")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Number of top actions to suggest in qlearning_llm mode")
    parser.add_argument("--batch_mode", type=str, default="llm",
                        choices=["llm", "qlearning", "random", "qlearning_llm"],
                        help="Mode to run in batch evaluation (used if --mode=batch)")
    parser.add_argument("--num_trials", type=int, default=10,
                        help="Number of trials to run in batch mode (per starting state if applicable)")
    # Add VLLM-specific arguments
    parser.add_argument("--vllm_device", type=str, default="cuda:2", # Changed default for safety if user has multiple GPUs
                        help="Device to run VLLM on (e.g., cuda, cuda:0, cuda:1, cpu)")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9,
                        help="Fraction of GPU memory to use")
    parser.add_argument("--vllm_dtype", type=str, default="auto",
                        help="Data type for model weights (auto, float16, bfloat16)")
    parser.add_argument("--vllm_enable_prefix_caching", action="store_true", default=True, # Default to True, can be negated
                        help="Enable prefix caching for faster generation")
    parser.add_argument("--no_vllm_enable_prefix_caching", action="store_false", dest="vllm_enable_prefix_caching",
                        help="Disable prefix caching")
    parser.add_argument("--vllm_max_model_len", type=int, default=None,
                        help="Maximum model sequence length")
    
    args = parser.parse_args()


    if args.mode == "human":
        run_human_interactive(
            env_config_path=args.config,
            max_steps=args.max_steps,
            start_state=args.start_state
        )
    elif args.mode == "llm":
        run_llm_interactive(
            model_name=args.model,
            env_config_path=args.config,
            max_steps=args.max_steps,
            start_state=args.start_state,
            vllm_device=args.vllm_device,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            vllm_dtype=args.vllm_dtype,
            enable_prefix_caching=args.vllm_enable_prefix_caching,
            max_model_len=args.vllm_max_model_len
        )
    elif args.mode == "qlearning":
        # Note: This mode's start_state handling depends on ImprisonedEnv and QLearningAgent supporting it.
        print("Warning: Q-learning mode's start_state behavior depends on underlying ImprisonedEnv and QLearningAgent supporting start_state_id in their reset/play_game methods.")
        run_qlearning_interactive(
            q_table_path=args.q_table,
            env_config_path=args.config,
            max_steps=args.max_steps,
            start_state=args.start_state
        )
    elif args.mode == "random":
        run_random_interactive(
            env_config_path=args.config,
            max_steps=args.max_steps,
            start_state=args.start_state
        )
    elif args.mode == "qlearning_llm":
        run_qlearning_llm_interactive(
            model_name=args.model,
            q_table_path=args.q_table,
            env_config_path=args.config,
            max_steps=args.max_steps,
            start_state=args.start_state,
            top_k=args.top_k,
            vllm_device=args.vllm_device,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            vllm_dtype=args.vllm_dtype,
            enable_prefix_caching=args.vllm_enable_prefix_caching,
            max_model_len=args.vllm_max_model_len
        )
    elif args.mode == "batch":
        run_batch_evaluation(
            mode=args.batch_mode,
            num_trials=args.num_trials,
            model_name=args.model,
            q_table_path=args.q_table,
            env_config_path=args.config,
            max_steps=args.max_steps,
            start_state=args.start_state,
            top_k=args.top_k,
            vllm_device=args.vllm_device,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            vllm_dtype=args.vllm_dtype,
            enable_prefix_caching=args.vllm_enable_prefix_caching,
            max_model_len=args.vllm_max_model_len
        )

if __name__ == "__main__":
    main()