import random
import yaml
import os
import re
from typing import List, Dict, Any, Tuple

from datasets import Dataset
from trl.trainer.grpo_trainer import RewardFunc

from verifiers.envs.multistep_env import MultiStepEnv
from verifiers.envs.imprisoned.text_game_rubric import TextGameRubric

SYSTEM_PROMPT = """You are playing a text-based adventure game where you are trying to escape from prison.
You will be given a description of the current state you are in, and a list of actions you can take.
Choose ONE action from the available actions list. You must select an action that is explicitly listed.

Your response must be in the following format:
<thinking>
[Your thoughts on what to do next]
</thinking>
<action>
[The exact action you choose from the available actions]
</action>

Remember: Only choose actions that are explicitly listed in the "Available actions" section.
"""

class ImprisonedMultistepEnv(MultiStepEnv):
    def __init__(self, 
                 config_path=None,
                 system_prompt=SYSTEM_PROMPT,
                 few_shot=None,
                 **kwargs):
        
        if few_shot is None:
            few_shot = []
            
        sampling_args = {
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
        }
        
        super().__init__(system_prompt=system_prompt, few_shot=few_shot, sampling_args=sampling_args, **kwargs)
        
        # If no config path is provided, use the one in the same directory as this module
        if config_path is None:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(module_dir, "imprisoned.yaml")
        
        # Ensure the file exists
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"YAML file not found at {config_path}")
        
        print(f"Loading game data from: {config_path}")
        
        # Load the YAML file
        with open(config_path, "r") as file:
            self.game_data = yaml.safe_load(file)
            
        self.states = self.game_data.get("states", {})
        self.starting_states = self.game_data.get("starting_states", [])
        
        # Verify starting states exist in the states dictionary
        self.starting_states = [s for s in self.starting_states if s in self.states]
        
        if not self.starting_states:
            raise ValueError("No valid starting states found in the YAML!")
            
        # Initialize environment state variables
        self.current_state = None
        self.inventory = set()
        self.rubric = TextGameRubric()
        self.dataset = None
        self.eval_dataset = None

    def get_dataset(self, num_samples=1000, **kwargs: Any) -> Dataset | None:
        """Generate a dataset for training with a specified number of samples."""
        if self.dataset is None:
            # Create a simple dataset with starting prompts
            examples = []
            
            # Create initial examples from each starting state
            for state in self.starting_states:
                # Reset the environment to this specific state
                self.current_state = state
                self.inventory = set()
                
                # Get the state description using render()
                observation = self.render()
                
                # Create the prompt with system message and initial observation
                prompt = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"{observation}\n\nWhat do you want to do?"}
                ]
                examples.append({"prompt": prompt})
            
            # Repeat examples to reach desired number of samples
            if len(examples) < num_samples:
                # Calculate how many times to repeat each example
                repeat_count = num_samples // len(examples) + 1
                examples = examples * repeat_count
                # Trim to exact number requested
                examples = examples[:num_samples]
            
            import pandas as pd
            from datasets import Dataset
            
            df = pd.DataFrame(examples)
            self.dataset = Dataset.from_pandas(df)
            
        return self.dataset
    
    def get_eval_dataset(self, num_samples=50, **kwargs: Any) -> Dataset | None:
        """Generate a dataset for evaluation with a specified number of samples."""
        if self.eval_dataset is None:
            # Create a simple dataset with starting prompts
            examples = []
            
            # Create initial examples from each starting state
            for state in self.starting_states:
                # Reset the environment to this specific state
                self.current_state = state
                self.inventory = set()

                # Get the state description using render()
                observation = self.render()
                
                # Create the prompt with system message and initial observation
                prompt = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"{observation}\n\nWhat do you want to do?"}
                ]
                examples.append({"prompt": prompt, "answer": ""})  # Empty answer for evaluation
            
            # Repeat examples to reach desired number of samples
            if len(examples) < num_samples:
                # Calculate how many times to repeat each example
                repeat_count = num_samples // len(examples) + 1
                examples = examples * repeat_count
                # Trim to exact number requested
                examples = examples[:num_samples]
            
            import pandas as pd
            from datasets import Dataset
            
            df = pd.DataFrame(examples)
            self.eval_dataset = Dataset.from_pandas(df)
            
        return self.eval_dataset
    
    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        """Get the reward functions for evaluation."""
        return self.rubric.get_reward_funcs()
    
    def _get_step_count(self, messages: List[Dict[str, str]]) -> int:
        """Count the number of actual interaction steps in the message history."""
        # Skip messages that are part of system prompt or few-shot examples
        conversation_start = 1  # Start after system message
        if self.few_shot:
            # Account for all few-shot messages
            conversation_start += len(self.few_shot)
        
        # Count assistant messages after the conversation start
        step_count = 0
        for message in messages[conversation_start:]:
            if message.get("role") == "assistant":
                step_count += 1
                
        return step_count

    def is_completed(self, messages: List[Dict[str, str]], **kwargs: Any) -> bool:
        """Check if the conversation has reached a terminal state."""
        # Need at least a system message, user message, and assistant response
        if len(messages) < 3:
            return False
        
        # Check if we've reached the maximum number of steps
        step_count = self._get_step_count(messages)
        if step_count >= self.max_steps:
            print(f"Step count: {step_count}, max steps: {self.max_steps}")
            return True
        
        # Check if we've reached a terminal state
        user_messages = [msg for msg in messages if msg["role"] == "user"]
        if user_messages:
            last_env_message = user_messages[-1]["content"]
            
            # Check for explicit GAME OVER messages
            if "GAME OVER" in last_env_message:
                print(f"GAME OVER message found: {last_env_message}")
                return True
            
            # Check for successful escape message
            if "successfully escape" in last_env_message.lower():
                print(f"Successfully escape message found: {last_env_message}")
                return True
            
            # Check for terminal state marker
            state_marker_match = re.search(r'\[STATE: ([^\]]+)\]', last_env_message)
            if state_marker_match:
                state_name = state_marker_match.group(1)
                if state_name in self.states and self.states[state_name].get("terminal", False):
                    print(f"Terminal state reached: {state_name}")
                    return True
        
        return False
    
    def _parse_action(self, message: str) -> str:
        """Parse the user's action from their message using the XML-like format."""
        message = message.strip()
        
        # Try to extract action from the XML-like format
        action_tag_start = message.find("<action>")
        action_tag_end = message.find("</action>")
        
        if action_tag_start != -1 and action_tag_end != -1 and action_tag_start < action_tag_end:
            # Extract content between <action> and </action> tags
            action = message[action_tag_start + len("<action>"):action_tag_end].strip()
            return action
        
        # Return None to indicate format error
        return None
    
    def _get_available_actions(self, state: str) -> Dict[str, Dict[str, Any]]:
        """Get available actions for the current state, considering inventory."""
        if state not in self.states:
            return {}
            
        actions = self.states[state].get("actions", {})
        available_actions = {}
        
        for action, details in actions.items():
            conditions = details.get("conditions", {})
            required_item = conditions.get("requires")
            
            if not required_item or required_item in self.inventory:
                available_actions[action] = details
                
        return available_actions
    
    def _match_action(self, user_action: str, available_actions: Dict[str, Dict[str, Any]]) -> str:
        """Match the user's action to an available action."""
        if not available_actions:
            return None
        
        # Clean up user action - remove whitespace and convert to lowercase
        user_action = user_action.strip().lower()
        
        # Try exact match first (case-insensitive)
        for action in available_actions:
            if action.lower() == user_action:
                return action
        
        # Try partial match where the user's action contains the full action name
        for action in available_actions:
            if action.lower() in user_action:
                return action
        
        # Try partial match where the action contains the user's input
        # Only if user input is at least 4 characters to avoid too loose matching
        if len(user_action) >= 4:
            for action in available_actions:
                if user_action in action.lower():
                    return action
        
        # Try matching against descriptions
        for action, details in available_actions.items():
            description = details.get("description", "").lower()
            if user_action in description:
                return action
        
        # If no match found, return None
        return None
    
    def reset(self, seed=None, start_state_id: str = None) -> str:
        """Reset the environment to a specified or random starting state."""
        if seed is not None:
            random.seed(seed)
        
        if start_state_id and start_state_id in self.starting_states:
            self.current_state = start_state_id
        else:
            if start_state_id:
                print(f"Warning: Provided start_state_id '{start_state_id}' is not valid or not in starting_states. Choosing a random starting state.")
            self.current_state = random.choice(self.starting_states)
            
        self.inventory = set()
        
        return self.render()

    def _identify_initial_state(self, initial_message: str) -> str:
        """Identify the initial state from the message content."""
        found_state = None
        
        # First look for the state identifier marker
        state_marker_match = re.search(r'\[STATE: ([^\]]+)\]', initial_message)
        if state_marker_match:
            state_name = state_marker_match.group(1)
            if state_name in self.states:
                found_state = state_name
                return found_state
        
        # Try exact description match - look for longer descriptions first to avoid partial matches
        state_matches = []
        for state_name, state_data in self.states.items():
            state_desc = state_data.get("description", "")
            if state_desc and state_desc in initial_message:
                # Store the state name and the length of its description
                state_matches.append((state_name, len(state_desc)))
        
        # If we found matches, use the one with the longest description
        if state_matches:
            state_matches.sort(key=lambda x: x[1], reverse=True)
            found_state = state_matches[0][0]
            return found_state
        
        # If no match, try matching first line
        if not found_state and initial_message:
            initial_lines = [line.strip() for line in initial_message.split('\n') if line.strip()]
            if initial_lines:
                initial_first_line = initial_lines[0]
                for state_name, state_data in self.states.items():
                    state_desc = state_data.get("description", "")
                    if state_desc:
                        state_lines = [line.strip() for line in state_desc.split('\n') if line.strip()]
                        if state_lines and initial_first_line == state_lines[0]:
                            found_state = state_name
                            break
        
        # If still no match, use a default starting state
        if not found_state:
            found_state = random.choice(self.starting_states)
        
        return found_state

    def env_response(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, str]:
        """Generate the environment's response based on the user's action."""
        
        # Find the last user message to identify the current state
        user_messages = [msg for msg in messages if msg["role"] == "user"]
        
        # First interaction or reset needed
        if len(user_messages) <= 1 or not hasattr(self, 'current_state') or self.current_state is None:
            # The first message should come from the dataset
            initial_message = user_messages[0]["content"] if user_messages else ""
            
            # Identify the initial state
            found_state = self._identify_initial_state(initial_message)
            
            # Initialize the game state
            self.current_state = found_state
            self.inventory = set()
        else:
            # For subsequent interactions, try to identify state from the last user message
            last_user_message = user_messages[-1]["content"]
            
            # Check if we need to re-identify the state (e.g., after an invalid action)
            state_marker_match = re.search(r'\[STATE: ([^\]]+)\]', last_user_message)
            if state_marker_match:
                state_name = state_marker_match.group(1)
                if state_name in self.states and state_name != self.current_state:
                    self.current_state = state_name
        
        # Parse the user's action from the most recent assistant message
        assistant_messages = [msg for msg in messages if msg["role"] == "assistant"]
        if not assistant_messages:
            # First turn, no assistant message yet
            # Get available actions for the current state
            available_actions = self._get_available_actions(self.current_state)
            
            # Format the action list
            action_text = "\n\nAvailable actions:\n"
            for action, details in available_actions.items():
                action_text += f"- {action}: {details.get('description', '')}\n"
            
            # Get the state description
            state_desc = self.states[self.current_state].get("description", "")
            
            # Add state identifier marker
            state_id_marker = f"[STATE: {self.current_state}]"
            
            # Ensure content is properly formatted as a string
            content = f"{state_id_marker}\n{state_desc}\n{action_text}\n\nWhat do you want to do?"
            return {"role": "user", "content": content}
        
        # Get the last assistant message
        user_message = assistant_messages[-1]["content"]
        user_action = self._parse_action(user_message)
        
        # Check if the format was correct
        if user_action is None:
            content = "Please use the correct format for your response:\n\n<thinking>\n[Your thoughts on what to do next]\n</thinking>\n<action>\n[The exact action you choose from the available actions]\n</action>"
            return {"role": "user", "content": content}
        
        # Get available actions for the current state
        available_actions = self._get_available_actions(self.current_state)
        
        # Match the user's action to an available action
        matched_action = self._match_action(user_action, available_actions)
        
        # If the action was invalid, inform the user with the available actions
        if not matched_action:
            # Format the action list
            action_text = "\n\nAvailable actions:\n"
            for action, details in available_actions.items():
                action_text += f"- {action}: {details.get('description', '')}\n"
            
            # Return the current state description along with available actions
            state_desc = self.states[self.current_state].get("description", "")
            inventory_text = f"\n📜 Inventory: {', '.join(self.inventory) if self.inventory else 'Empty'}"
            
            # Add state identifier marker to ensure state consistency
            state_id_marker = f"[STATE: {self.current_state}]"
            
            content = f"That action is not available. Please choose from the available actions:{action_text}\n\n{state_id_marker}\n{state_desc}{inventory_text}{action_text}"
            return {"role": "user", "content": content}
        
        # Use the step_env method to process the action and get the new state
        observation, reward, terminated, truncated, info = self.step_env(matched_action)
        
        # Handle terminal states
        if terminated:           
            if self.current_state == "escape_success":
                content = f"{observation}\n\n🎉 GAME OVER: You have successfully escaped! Congratulations!"
                return {"role": "user", "content": content}
            else:
                content = f"{observation}\n\n💀 GAME OVER: Your escape attempt has failed in {self.current_state}."
                return {"role": "user", "content": content}
        
        # Check if we've reached the maximum number of steps
        step_count = self._get_step_count(messages)
        if step_count >= self.max_steps:
            content = f"⏱️ GAME OVER: You ran out of time to escape after {step_count} attempts."
            return {"role": "user", "content": content}
        
        # Return the observation (which includes state description, inventory, and available actions)
        content = f"{observation}\n\nWhat do you want to do?"
        return {"role": "user", "content": content}

    def render(self) -> str:
        """Render the current state as a string."""
        if not hasattr(self, 'current_state') or self.current_state is None:
            return "Environment not initialized"
        
        # Add a state identifier marker that can be easily parsed
        state_id_marker = f"[STATE: {self.current_state}]"
        state_desc = self.states[self.current_state].get("description", "")
        inventory_text = f"\n📜 Inventory: {', '.join(self.inventory) if self.inventory else 'Empty'}"
        
        # For terminal states, don't show available actions
        if self.states[self.current_state].get("terminal", False):
            return f"{state_id_marker}\n{state_desc}{inventory_text}"
        
        available_actions = self._get_available_actions(self.current_state)
        action_text = "\n\nAvailable actions:\n"
        for action, details in available_actions.items():
            action_text += f"- {action}: {details.get('description', '')}\n"
        
        return f"{state_id_marker}\n{state_desc}{inventory_text}{action_text}"

    def step_env(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment with the given action."""
        # Check if the environment has been initialized
        if not hasattr(self, 'current_state') or self.current_state is None:
            self.reset()
        
        # Get available actions for the current state
        available_actions = self._get_available_actions(self.current_state)
        
        # Match the user's action to an available action
        matched_action = self._match_action(action, available_actions)
        
        # Initialize info dictionary
        info = {
            "action_taken": matched_action,
            "valid_action": matched_action is not None,
            "inventory": list(self.inventory),
            "previous_state": self.current_state  # Track previous state
        }
        
        # If no matching action is found, stay in current state with negative reward
        if not matched_action:
            # Don't change state, just return current state with no reward
            return self.render(), 0.0, False, False, info
        
        # Get the action details
        action_data = available_actions[matched_action]
        
        # Handle inventory item acquisition
        if "grants" in action_data:
            item = action_data["grants"]
            if item not in self.inventory:  # Only add if not already present
                self.inventory.add(item)
                info["item_acquired"] = item
        
        # Determine the next state
        next_state = None
        if "probabilities" in action_data:
            next_states = list(action_data["probabilities"].keys())
            probabilities = list(action_data["probabilities"].values())
            next_state = random.choices(next_states, probabilities)[0]
        elif "next_state" in action_data:
            next_state = action_data["next_state"]
        else:
            # If no next state specified, stay in current state
            next_state = self.current_state
        
        # Ensure the next state exists and is valid
        if next_state not in self.states:
            next_state = self.current_state
        
        # Update current state
        self.current_state = next_state
        
        # Check if the new state is terminal
        is_terminal = self.states[self.current_state].get("terminal", False)
        
        # Add state info to info dict
        info["state"] = self.current_state
        info["terminal"] = is_terminal
        
        # Return with no reward - rewards will be handled by the rubric
        return self.render(), 0.0, is_terminal, False, info 