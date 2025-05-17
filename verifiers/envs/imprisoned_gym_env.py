from typing import Any, Dict, List, Tuple, Optional
import copy
import numpy as np
import random
import re
import logging

from verifiers.envs.imprisoned.imprisoned_env import ImprisonedEnv

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

Do not include any other text in your response. You only should think about the action and then output the action in the format specified above.
Only output one action and finish there.

Remember: Only choose actions that are explicitly listed in the "Available actions" section.
"""

logging.basicConfig(level=logging.INFO)


class ImprisonedGymEnv:
    """
    A Gymnasium-like wrapper for ImprisonedEnv that supports:
    - Cloning the environment state
    - Starting from intermediate states
    - Standard gym interface (reset, step)
    """
    
    def __init__(self, max_steps=20, **kwargs):
        self.env = ImprisonedEnv(**kwargs)
        self.current_state_id = None
        self.current_state_desc = None
        self.step_count = 0
        self.max_steps = max_steps
        self.terminal = False
        self.inventory = set()
        self.system_prompt = kwargs.get("system_prompt", SYSTEM_PROMPT)
        self._system_prompt_added = False
        
    def get_train_dataloader(self):
        # This remains unchanged (dummy dataloader)
        pass

    def reset(self, seed=None, start_state_id: Optional[str] = None):
        """Reset the environment and return the initial state with proper formatting."""
        state_id = self.env.reset(seed=seed, start_state_id=start_state_id)
        
        state_desc = ""
        if hasattr(self.env, 'render'):
            rendered_state = self.env.render()
            if rendered_state:
                state_desc = rendered_state
        
        if not state_desc and hasattr(self.env, 'get_state_description'):
            desc_from_method = self.env.get_state_description()
            if desc_from_method:
                state_desc = desc_from_method
        
        if not state_desc and hasattr(self.env, 'states') and hasattr(self.env, 'current_state'):
            if self.env.current_state in self.env.states:
                state_desc = self.env.states[self.env.current_state].get("description", "")
                if state_desc:
                    state_desc = f"[STATE: {self.env.current_state}]\n{state_desc}"
        
        if not state_desc:
            state_desc = "You are in a prison cell. You need to find a way to escape."
        
        # Modularized formatting for inventory and available actions.
        state_desc += self._format_inventory()
        state_desc += self._format_available_actions()
        
        self.current_state_id = state_id
        self.current_state_desc = state_desc
        self.step_count = 0
        self.terminal = False
        self.inventory = set()
        
        # Format the observation with XML tags and an explicit delimiter between environment and LLM output.
        formatted_observation = (
            f"<state>You are in the following state:\n{state_desc}\n</state>\n\n"
            "Remember to use <thinking></thinking> and <action></action> tags in your response.\n"
        )
        
        # We no longer add the system prompt to the formatted observation
        # as it will be handled by the VinePPOTrainer as a separate message
        self._system_prompt_added = True
        
        return formatted_observation, {}
    
    def step(self, action):
        """Take a step in the environment using the parsed action."""
        parsed_action = self._parse_action(action)
        
        if parsed_action is None:
            error_msg = (
                "❌ ERROR: You didn't use the correct format for your response.\n"
                "You MUST use correct tags for thinking and action:\n"
                "<thinking>Your reasoning</thinking>\n"
                "<action>a valid action from the list</action>\n"
                "Let's try again with the correct format.\n"
                f"<state>\n{self.current_state_desc}\n</state>\n\n"
                "What do you do next?"
            )
            return error_msg, 0.0, False, False, {"error": "format_error"}
        
        available_actions = self.env.get_available_actions()
        action_index = self._find_action_index(parsed_action, available_actions)
        
        if action_index is None:
            error_msg = (
                f"❌ ERROR: Invalid action selected\n\n"
                f"You chose: '{parsed_action}'\n\n"
                f"{self._format_available_actions()}\n\n"
                f"Please select EXACTLY one of the available actions listed above.\n"
                f"Remember to use the format:\n"
                f"<thinking>Your reasoning</thinking>\n"
                f"<action>a valid action from the list</action>"
            )
            return error_msg, 0.0, False, False, {"error": "invalid_action"}
        
        result = self.env.step(action_index)
        if len(result) == 4:
            state_id, reward, done, info = result
            truncated = False
        else:
            state_id, reward, done, truncated, info = result
        
        self.current_state_id = state_id
        self.current_state_desc = self.env.get_state_description()
        # Always add inventory info
        self.current_state_desc += self._format_inventory() 
        if not done and "Available actions:" not in self.current_state_desc:
            self.current_state_desc += self._format_available_actions()
        
        self.step_count += 1
        self.terminal = done
        
        if hasattr(self.env, 'inventory'):
            self.inventory = self.env.inventory.copy()
        
        formatted_observation = f"<state>\n{self.current_state_desc}\n</state>\n\nWhat do you do next?"
        
        return formatted_observation, reward, done, truncated, info
    
    def _parse_action(self, action_text):
        """Extract the action from the XML format using regex for robustness."""
        match = re.search(r"<action>(.*?)</action>", action_text, re.DOTALL | re.IGNORECASE)
        if match:
            action = match.group(1).strip()
            if ":" in action:
                action = action.split(":", 1)[0].strip()
            return action
        return None
    
    def _find_action_index(self, action_text: str, available_actions: List[str]) -> Optional[int]:
        """Find the closest matching action index from available actions."""
        if not available_actions:
            return None
        for i, avail_action in enumerate(available_actions):
            if action_text.lower() == avail_action.lower():
                return i
        for i, avail_action in enumerate(available_actions):
            if avail_action.lower() in action_text.lower() or action_text.lower() in avail_action.lower():
                return i
        return None
    
    def _format_inventory(self) -> str:
        """Return formatted inventory information."""
        if hasattr(self.env, 'inventory'):
            return f"\n📜 Inventory: {', '.join(self.env.inventory) if self.env.inventory else 'Empty'}"
        return ""
    
    def _format_available_actions(self) -> str:
        """Return formatted available actions with descriptions."""
        action_text = self._get_available_actions_with_descriptions()
        if not action_text and hasattr(self.env, 'get_available_actions'):
            available_actions = self.env.get_available_actions()
            if available_actions:
                action_text = f"\n\nAvailable actions: {', '.join(available_actions)}"
        return action_text
    
    def _get_available_actions_with_descriptions(self) -> str:
        """Format available actions with descriptions in the same format as ImprisonedMultistepEnv."""
        if not hasattr(self.env, 'states') or not hasattr(self.env, 'current_state'):
            return ""
        
        current_state = self.env.current_state
        if not hasattr(self.env, 'states') or current_state not in self.env.states:
            return ""
        
        actions = self.env.states[current_state].get("actions", {})
        action_text = "\nAvailable actions:\n"
        for action, details in actions.items():
            conditions = details.get("conditions", {})
            required_item = conditions.get("requires")
            if not required_item or (hasattr(self.env, 'inventory') and required_item in self.env.inventory):
                action_text += f"- {action}: {details.get('description', '')}\n"
        return action_text
    
    def clone(self) -> 'ImprisonedGymEnv':
        """Create a deep copy of the environment."""
        new_env = ImprisonedGymEnv(max_steps=self.max_steps)
        new_env.env = copy.deepcopy(self.env)
        new_env.current_state_id = self.current_state_id
        new_env.current_state_desc = self.current_state_desc
        new_env.step_count = self.step_count
        new_env.terminal = self.terminal
        new_env.inventory = self.inventory.copy()
        return new_env
        
    def set_state(self, state: Any) -> None:
        """Set the environment to a specific state."""
        if isinstance(state, dict):
            self.current_state_id = state.get("state_id")
            self.current_state_desc = state.get("state_desc")
            self.step_count = state.get("step_count", 0)
            self.terminal = state.get("terminal", False)
            self.inventory = set(state.get("inventory", []))
            if self.current_state_id in self.env.states:
                self.env.current_state = self.current_state_id
                self.env.terminal = self.terminal
                self.env.inventory = self.inventory.copy()
        elif isinstance(state, str):
            for state_id, state_data in self.env.states.items():
                if state_data.get("description", "") == state:
                    self.current_state_id = state_id
                    self.current_state_desc = state
                    self.env.current_state = state_id
                    break
            else:
                self.current_state_id = random.choice(self.env.starting_states)
                self.current_state_desc = self.env.get_state_description()
                self.env.current_state = self.current_state_id
            
            self.step_count = 0
            self.terminal = False
            self.inventory = set()
            self.env.terminal = False
            self.env.inventory = set()
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state as a dictionary."""
        return {
            "state_id": self.current_state_id,
            "state_desc": self.current_state_desc,
            "step_count": self.step_count,
            "terminal": self.terminal,
            "inventory": list(self.inventory)
        }
    
    def get_available_actions(self) -> List[str]:
        """Get the list of available actions in the current state."""
        return self.env.get_available_actions()

    def __copy__(self):
        """Create a shallow copy of the environment."""
        new_env = ImprisonedGymEnv(max_steps=self.max_steps)
        new_env.env = copy.copy(self.env)
        new_env.current_state_id = self.current_state_id
        new_env.current_state_desc = self.current_state_desc
        new_env.step_count = self.step_count
        new_env.terminal = self.terminal
        new_env.inventory = self.inventory.copy()
        new_env._system_prompt_added = self._system_prompt_added
        new_env.system_prompt = self.system_prompt
        
        if hasattr(self.env, 'current_state'):
            new_env.env.current_state = self.env.current_state
        if hasattr(self.env, 'terminal'):
            new_env.env.terminal = self.env.terminal
        if hasattr(self.env, 'inventory') and hasattr(self.env.inventory, 'copy'):
            new_env.env.inventory = self.env.inventory.copy()
        
        if new_env.current_state_desc is None and new_env.current_state_id is not None:
            if hasattr(new_env.env, 'get_state_description'):
                new_env.current_state_desc = new_env.env.get_state_description()
        
        return new_env

    def __deepcopy__(self, memo):
        """Create a deep copy of the environment for Monte Carlo rollouts."""
        new_env = ImprisonedGymEnv(max_steps=self.max_steps)
        memo[id(self)] = new_env
        new_env.env = copy.deepcopy(self.env, memo)
        new_env.current_state_id = self.current_state_id
        new_env.current_state_desc = self.current_state_desc
        new_env.step_count = self.step_count
        new_env.terminal = self.terminal
        new_env.inventory = copy.deepcopy(self.inventory, memo)
        new_env.system_prompt = self.system_prompt
        new_env._system_prompt_added = self._system_prompt_added
        
        if hasattr(self.env, 'current_state'):
            new_env.env.current_state = self.env.current_state
        if hasattr(self.env, 'terminal'):
            new_env.env.terminal = self.env.terminal
        if hasattr(self.env, 'inventory'):
            new_env.env.inventory = copy.deepcopy(self.env.inventory, memo)
        
        if new_env.current_state_desc is None and new_env.current_state_id is not None:
            if hasattr(new_env.env, 'get_state_description'):
                new_env.current_state_desc = new_env.env.get_state_description()
        
        return new_env
