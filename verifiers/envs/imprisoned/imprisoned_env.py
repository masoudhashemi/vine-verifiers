import os
import random
import logging
import yaml
from gymnasium import spaces
import gymnasium as gym
from typing import Optional

logging.basicConfig(level=logging.INFO)

class ImprisonedEnv(gym.Env):
    """A custom OpenAI Gym environment for the text-based game 'Imprisoned'."""

    def __init__(self, config_path=None):
        super(ImprisonedEnv, self).__init__()

        if config_path is None:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(module_dir, "imprisoned.yaml")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"YAML file not found at {config_path}")
        
        logging.info(f"Loading game data from: {config_path}")
        
        with open(config_path, "r") as file:
            self.game_data = yaml.safe_load(file)

        self.states = self.game_data.get("states", {})
        self.starting_states = self.game_data.get("starting_states", [])
        self.starting_states = [s for s in self.starting_states if s in self.states]

        if not self.starting_states:
            raise ValueError("No valid starting states found in the YAML!")

        self.current_state = random.choice(self.starting_states)
        self.terminal = False
        self.inventory = set()

        self.action_space = spaces.Discrete(len(self.get_available_actions()))
        self.observation_space = spaces.Discrete(len(self.states))

    def get_available_actions(self):
        if self.current_state not in self.states:
            logging.warning(f"⚠️ WARNING: State '{self.current_state}' not found in YAML!")
            return []

        actions = self.states[self.current_state].get("actions", {})
        available_actions = []

        for action, details in actions.items():
            conditions = details.get("conditions", {})
            required_item = conditions.get("requires")

            if not required_item or required_item in self.inventory:
                available_actions.append(action)

        return available_actions

    def get_state_description(self):
        return self.states.get(self.current_state, {}).get("description", "Unknown state.")

    def step(self, action_index):
        actions = self.get_available_actions()
        if not actions:
            logging.warning(f"⚠️ WARNING: No actions available in state '{self.current_state}'")
            return self.current_state, -0.1, True, {}
        
        if action_index >= len(actions):
            logging.warning("⚠️ WARNING: Invalid action index selected.")
            return self.current_state, -0.1, True, {}

        chosen_action = actions[action_index]
        action_data = self.states[self.current_state]["actions"][chosen_action]

        if "grants" in action_data:
            self.inventory.add(action_data["grants"])

        if "probabilities" in action_data:
            next_states = list(action_data["probabilities"].keys())
            probabilities = list(action_data["probabilities"].values())
            new_state = random.choices(next_states, probabilities)[0]
        else:
            new_state = action_data.get("next_state", self.current_state)

        if new_state not in self.states:
            logging.warning(f"⚠️ WARNING: Next state '{new_state}' not found! Returning to a safe fallback.")
            new_state = random.choice(self.starting_states)

        self.current_state = new_state
        self.terminal = self.states[self.current_state].get("terminal", False)

        if not self.get_available_actions() and not self.terminal:
            logging.warning(f"⚠️ WARNING: State '{self.current_state}' has no actions!")
            self.current_state = random.choice(self.starting_states)

        reward = (
            1
            if self.current_state == "escape_success"
            else (-1 if self.terminal else 0)
        )

        return self.current_state, reward, self.terminal, {}

    def reset(self, seed: Optional[int] = None, start_state_id: Optional[str] = None):
        if seed is not None:
            random.seed(seed) # For reproducibility if needed

        if start_state_id and start_state_id in self.starting_states:
            self.current_state = start_state_id
        else:
            if start_state_id:
                logging.warning(f"⚠️ WARNING: Provided start_state_id '{start_state_id}' is not valid or not in starting_states. Choosing a random starting state.")
            self.current_state = random.choice(self.starting_states)
        
        # Ensure the chosen state is valid
        while self.current_state not in self.states:
            logging.warning(f"⚠️ WARNING: State '{self.current_state}' (chosen for reset) not in YAML states! Choosing another random starting state.")
            self.current_state = random.choice(self.starting_states)

        self.terminal = False
        self.inventory.clear()
        return self.current_state

    def render(self):
        logging.info(f"\n🔹 {self.get_state_description()}")
        logging.info(f"📜 Inventory: {', '.join(self.inventory) if self.inventory else 'Empty'}")

    def close(self):
        pass
