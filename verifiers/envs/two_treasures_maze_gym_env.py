from typing import Any, Dict, List, Tuple, Optional
import copy
import numpy as np
import random
import re
import logging

# --- Maze Definitions and Constants ---
MAZE_1 = [
    "#######",
    "#A..1.#",
    "#.#...#",
    "###.#2#",
    "#######",
]

MAZE_2 = [
    "########",
    "#A....1#",
    "#.#.##.#",
    "#.#... #",
    "#.####.#",
    "#.....2#",
    "########",
]

MAZE_3 = [
    "#######",
    "#A#...#",
    "#..1#.#",
    "#....2#",
    "#######",
]

MAZE_4 = [
    "#########",
    "#A....#.#",
    "#.###.#.#",
    "#1..#..2#",
    "#########",
]

MAZE_5 = [
    "#######",
    "#1....#",
    "#.A.#.#",
    "#...2.#",
    "#######",
]

MAZE_6 = [
    "########",
    "#.1....#",
    "#.#.##.#",
    "#.#.A. #",
    "#.####.#",
    "#.....2#",
    "########",
]

ALL_MAZES = [MAZE_1, MAZE_2, MAZE_3, MAZE_4, MAZE_5, MAZE_6]

# Rewards and Penalties
T1_REWARD = 1.0
T2_REWARD = 2.0
STEP_PENALTY = -0.05
WALL_PENALTY = -0.2
MAX_STEPS_PENALTY = -1.0
INCORRECT_FORMAT_PENALTY = -0.1
INVALID_ACTION_PENALTY = -0.1

VALID_ACTIONS_LIST = ["move up", "move down", "move left", "move right"]

# Define orientations (row_delta, col_delta) directly
NORTH = (-1, 0) # Up
SOUTH = (1, 0)  # Down
WEST = (0, -1)  # Left
EAST = (0, 1)   # Right

MAZE_GYM_SYSTEM_PROMPT = ("""You are playing a text-based maze game. Your goal is to find the treasure with the highest value.
Your response must be in the following format:
<thinking>
move up/down/left/right
</thinking>

Only output one action within the thinking tag.
Remember: Only choose actions that are explicitly listed.""")

logging.basicConfig(level=logging.INFO)


class TwoTreasuresMazeGymEnv:
    """
    A standalone Gymnasium-like wrapper for a text-based maze environment with two treasures.
    Supports:
    - Cloning the environment state
    - Starting from specific maze layouts
    - Standard gym interface (reset, step) for PPO training
    - Step-wise rewards for PPO
    """

    def __init__(self, max_steps: int = 25, seed: Optional[int] = None, **kwargs):
        self.logger = logging.getLogger(f"verifiers.envs.{self.__class__.__name__}")
        self.rng = random.Random(seed)
        self.max_steps: int = max_steps
        self.mazes: List[List[str]] = ALL_MAZES # Store available mazes

        # Internal game state attributes (will be set in _reset_internal_state)
        self.current_maze_layout: List[str] = []
        self.rows: int = 0
        self.cols: int = 0
        self.agent_pos: Optional[Tuple[int, int]] = None
        self.t1_pos: Optional[Tuple[int, int]] = None
        self.t2_pos: Optional[Tuple[int, int]] = None
        self.walls: set[Tuple[int, int]] = set()
        self.current_step_internal: int = 0 # For game logic steps
        self.game_over_internal: bool = False # For internal game termination
        self.final_message: str = ""

        # Gym wrapper specific attributes
        self.current_llm_prompt: Optional[str] = None
        self.current_step_gym: int = 0 # For gym episode steps
        self.terminal_gym: bool = False

        self.system_prompt = kwargs.get("system_prompt", MAZE_GYM_SYSTEM_PROMPT)
        
        # For VinePPOTrainer compatibility to select starting states
        self.possible_start_state_ids = [str(i) for i in range(len(self.mazes))]
        self.starting_states = self.possible_start_state_ids

        self._reset_internal_state() # Initialize with a random maze by default

    def _reset_internal_state(self, specific_maze_layout: Optional[List[str]] = None):
        """Resets the internal game state for a new episode."""
        if specific_maze_layout:
            self.current_maze_layout = [row[:] for row in specific_maze_layout] # Deep copy
        elif self.mazes:
            self.current_maze_layout = [row[:] for row in self.rng.choice(self.mazes)] # Deep copy
        else:
            raise ValueError("No mazes available to choose from for a new episode.")
            
        self.rows = len(self.current_maze_layout)
        self.cols = len(self.current_maze_layout[0])
        self.agent_pos = None
        self.t1_pos = None
        self.t2_pos = None
        self.walls = set()
        self.current_step_internal = 0
        self.game_over_internal = False
        self.final_message = ""

        for r, row_str in enumerate(self.current_maze_layout):
            for c, char in enumerate(row_str):
                if char == '#':
                    self.walls.add((r, c))
                elif char == 'A':
                    self.agent_pos = (r, c)
                elif char == '1':
                    self.t1_pos = (r, c)
                elif char == '2':
                    self.t2_pos = (r, c)

        if self.agent_pos is None or self.t1_pos is None or self.t2_pos is None:
            self.logger.error(f"Maze parsing failed for layout: {self.current_maze_layout}. Agent: {self.agent_pos}, T1: {self.t1_pos}, T2: {self.t2_pos}")
            raise ValueError(f"Maze parsing failed. A, T1, or T2 missing in layout.")

    def get_board_as_string(self) -> str:
        """
        Generates a string representation of the current maze board,
        with the agent's position marked as 'P'.
        """
        if not self.current_maze_layout:
            return "Error: Maze layout not initialized."
        
        board_lines = []
        agent_r, agent_c = self.agent_pos if self.agent_pos else (-1, -1) 

        for r_idx, row_str in enumerate(self.current_maze_layout):
            display_row_chars = list(row_str) # Work with a copy
            if r_idx == agent_r and 0 <= agent_c < len(display_row_chars):
                display_row_chars[agent_c] = "P"
            board_lines.append("".join(display_row_chars))
        
        if not board_lines:
            return "Error: Could not generate board."
            
        separator = "-" * len(board_lines[0]) if board_lines else ""
        return "\n".join(board_lines) + "\n" + separator

    def _get_state_description(self) -> str:
        """Generates the textual description of the current state."""
        if self.game_over_internal:
            return self.final_message

        if self.agent_pos is None:
            return "Error: Agent position is unknown."

        r, c = self.agent_pos
        description = f"You are at ({r},{c}). "
        details = []
        visible = ""

        moves = {"move up": NORTH, "move down": SOUTH, "move left": WEST, "move right": EAST}
        for direction, (dr, dc) in moves.items():
            nr, nc = r + dr, c + dc
            if (nr, nc) in self.walls:
                details.append(f"{direction} is a wall.")
            else:
                details.append(f"{direction} is open path.")
                if (nr, nc) == self.t1_pos:
                    visible = f" Treasure 1 (T1) is {direction}."
                elif (nr, nc) == self.t2_pos:
                    visible = f" Treasure 2 (T2) is {direction}."

        description += " ".join(details)
        description += visible
        return description.strip()

    def _format_prompt_for_agent(self, raw_observation: str) -> str:
        """
        Formats the raw observation string into a detailed prompt for the LLM agent.
        """
        board_string = self.get_board_as_string()
        valid_actions_str = ", ".join(f"'{a}'" for a in VALID_ACTIONS_LIST)

        prompt = f"""The current maze layout is:
{board_string}

Note: 'A' represents your starting position in the maze.

To navigate, you must choose *one* of the following actions: {valid_actions_str}.
If you choose 'move up', your action response must be: <thinking>move up</thinking>.

Actions correspond to cardinal directions:
- "move up": decrease row index
- "move down": increase row index
- "move left": decrease column index
- "move right": increase column index

Your current observation (position and open paths):
{raw_observation}

Ensure your chosen action corresponds to an 'open path'.

Chosen action:"""
        return prompt

    def reset(self, seed: Optional[int] = None, start_state_id: Optional[str] = None) -> Tuple[str, Dict]:
        """Reset the environment and return the initial state with proper formatting."""
        if seed is not None:
            self.rng.seed(seed)

        chosen_maze_layout: Optional[List[str]] = None
        if start_state_id is not None:
            try:
                maze_idx = int(start_state_id)
                if 0 <= maze_idx < len(self.mazes):
                    chosen_maze_layout = self.mazes[maze_idx]
                else:
                    self.logger.warning(f"Invalid start_state_id index {maze_idx}. Choosing a random maze.")
            except ValueError:
                self.logger.warning(f"Invalid start_state_id format {start_state_id}. Choosing a random maze.")
        
        # If no specific layout chosen or found, _reset_internal_state will pick one randomly
        self._reset_internal_state(specific_maze_layout=chosen_maze_layout) 
        
        raw_initial_state = self._get_state_description()
        self.current_llm_prompt = self._format_prompt_for_agent(raw_initial_state)
        
        self.current_step_gym = 0
        self.terminal_gym = False

        formatted_observation = self.current_llm_prompt
        
        return formatted_observation, {}

    def _parse_action(self, action_text_from_llm: str) -> Optional[str]:
        """Extract the action from the <thinking>...</thinking> tags using regex."""
        match = re.search(r"<thinking>(.*?)</thinking>", action_text_from_llm, re.DOTALL | re.IGNORECASE)
        if match:
            action = match.group(1).strip().lower()
            # Check if the extracted content is one of the valid actions
            if action in VALID_ACTIONS_LIST:
                return action
            # If not directly a valid action, return the extracted content for further checking in step method
            # This handles cases where the LLM might put extra text inside <thinking> along with a valid action.
            # For stricter parsing, you might only return `action` if `action in VALID_ACTIONS_LIST`
            # and `None` otherwise, forcing an error if any extra text exists.
            # However, for now, let's see if the step method's check is sufficient.
            # Update: Let's be stricter. Only return if it's a valid action. Otherwise, it's a format/content error.
            return action # The step method will check if this is in VALID_ACTIONS_LIST
        return None # Return None if <thinking> tags are not found or are empty

    def step(self, action_from_llm: str) -> Tuple[str, float, bool, bool, Dict]:
        """Take a step in the environment using the parsed action."""
        parsed_action_content = self._parse_action(action_from_llm)
        info = {}
        current_reward = STEP_PENALTY # Base penalty for taking a step

        next_llm_prompt_content: str
        action_error_message = "" # For LLM feedback

        if self.game_over_internal: # Should not happen if PPO loop respects done, but safety check
            self.logger.warning("Step called after game is over.")
            formatted_observation = f"{self.final_message}\nGame already ended."
            return formatted_observation, 0.0, self.terminal_gym, True, {"error": "stepped_after_done"}

        if parsed_action_content is None: # This now means <thinking> tags were missing/malformed
            action_error_message = (
                "❌ ERROR: You didn't use the correct format for your response.\n"
                "You MUST use the <thinking> tag around your action:\n"
                "<thinking>move up/down/left/right</thinking>\n"
                "Let's try again with the correct format."
            )
            current_reward += INCORRECT_FORMAT_PENALTY 
            info["error"] = "format_error"
            next_llm_prompt_content = f"{action_error_message}\n\n{self._format_prompt_for_agent(self._get_state_description())}"
        elif parsed_action_content not in VALID_ACTIONS_LIST:
            available_actions_str = ", ".join(f"'{a}'" for a in VALID_ACTIONS_LIST)
            action_error_message = (
                f"❌ ERROR: Invalid action command '{parsed_action_content}' found within <thinking> tags.\n"
                f"Valid actions are: {available_actions_str}.\n"
                f"Please select EXACTLY one of the available actions listed inside the <thinking> tags.\n"
                f"For example: <thinking>move up</thinking>"
            )
            current_reward += INVALID_ACTION_PENALTY # Keep penalty for invalid action command
            info["error"] = "invalid_action_command"
            current_raw_observation = self._get_state_description()
            next_llm_prompt_content = f"{action_error_message}\n\n{self._format_prompt_for_agent(current_raw_observation)}"
        else:
            # Process valid action format
            r, c = self.agent_pos # Current agent position
            nr, nc = r, c # New position, default to current
            move_made = False

            action_map = {
                "move up": NORTH, "move down": SOUTH,
                "move left": WEST, "move right": EAST
            }
            dr, dc = action_map[parsed_action_content]
            nr, nc = r + dr, c + dc

            if (nr, nc) in self.walls:
                current_reward += WALL_PENALTY 
                info["game_event"] = "hit_wall"
                # Agent stays, nr, nc remain r, c effectively
                action_error_message = f"You tried to '{parsed_action_content}' but hit a wall! You remain at ({r},{c})."
            else:
                self.agent_pos = (nr, nc) # Update agent position
                move_made = True
                # STEP_PENALTY already applied
                info["game_event"] = "moved_successfully"

            self.current_step_internal += 1 # Increment internal game step

            # Check for game end conditions after moving
            if move_made:
                if self.agent_pos == self.t1_pos:
                    current_reward += T1_REWARD
                    self.game_over_internal = True
                    self.final_message = "You found Treasure 1 (T1)! Game Over."
                    info["game_event"] = "found_t1"
                elif self.agent_pos == self.t2_pos:
                    current_reward += T2_REWARD
                    self.game_over_internal = True
                    self.final_message = "You found Treasure 2 (T2)! Game Over."
                    info["game_event"] = "found_t2"
            
            if not self.game_over_internal and self.current_step_internal >= self.max_steps:
                current_reward += MAX_STEPS_PENALTY
                self.game_over_internal = True
                self.final_message = "Maximum steps reached. Game Over."
                info["game_event"] = "max_steps_internal"
            
            # Prepare next LLM prompt content
            current_raw_observation = self._get_state_description() # This will be final_message if game_over_internal
            if action_error_message: # If agent hit a wall
                next_llm_prompt_content = f"{action_error_message}\n\n{self._format_prompt_for_agent(current_raw_observation)}"
            else:
                next_llm_prompt_content = self._format_prompt_for_agent(current_raw_observation)
        
        self.current_step_gym += 1 # Increment gym step count
        self.current_llm_prompt = next_llm_prompt_content # Update stored LLM prompt

        is_truncated_gym = self.current_step_gym >= self.max_steps
        
        # If gym truncates, and game didn't naturally end via internal logic (treasure/max_steps_internal)
        if is_truncated_gym and not self.game_over_internal:
            # Apply gym's own truncation penalty only if not already game over with a specific outcome
            # Check info["game_event"] to avoid double penalizing for MAX_STEPS_PENALTY
            if info.get("game_event") not in ["found_t1", "found_t2", "max_steps_internal"]:
                 current_reward += MAX_STEPS_PENALTY 
            info["gym_truncated"] = True
            # If game wasn't over internally, but gym truncates, set final message for truncation
            if not self.final_message: # Ensure we don't overwrite a treasure message
                self.final_message = "Episode truncated by gym step limit."
                self.current_llm_prompt = f"<state>\n{self.final_message}\n</state>\n\nWhat do you do next?"
                if not self.game_over_internal:
                    self.game_over_internal = True # Mark internal game as over due to gym truncation
                    self.current_llm_prompt = self._format_prompt_for_agent(self._get_state_description()) # This will now use final_message


        self.terminal_gym = self.game_over_internal or is_truncated_gym

        if self.terminal_gym:
            formatted_observation = f"<state>\n{self.final_message}\n</state>\n\nEpisode has ended."
        else:
            formatted_observation = f"<state>\n{self.current_llm_prompt}\n</state>\n\nWhat do you do next?"
        
        return formatted_observation, current_reward, self.terminal_gym, is_truncated_gym, info

    def get_available_actions(self) -> List[str]:
        """Get the list of available actions in the current state."""
        return copy.deepcopy(VALID_ACTIONS_LIST)

    def clone(self) -> 'TwoTreasuresMazeGymEnv':
        """Create a deep copy of the environment for rollouts."""
        new_gym_env = TwoTreasuresMazeGymEnv(max_steps=self.max_steps, seed=self.rng.randint(0, 2**32 - 1))
        
        # Deepcopy relevant attributes
        new_gym_env.current_maze_layout = copy.deepcopy(self.current_maze_layout)
        new_gym_env.rows = self.rows
        new_gym_env.cols = self.cols
        new_gym_env.agent_pos = self.agent_pos
        new_gym_env.t1_pos = self.t1_pos
        new_gym_env.t2_pos = self.t2_pos
        new_gym_env.walls = copy.deepcopy(self.walls)
        new_gym_env.current_step_internal = self.current_step_internal
        new_gym_env.game_over_internal = self.game_over_internal
        new_gym_env.final_message = self.final_message
        
        new_gym_env.current_llm_prompt = self.current_llm_prompt
        new_gym_env.current_step_gym = self.current_step_gym
        new_gym_env.terminal_gym = self.terminal_gym
        new_gym_env.system_prompt = self.system_prompt # system_prompt is not state, but part of config
        
        new_gym_env.rng = copy.deepcopy(self.rng)
        new_gym_env.possible_start_state_ids = self.possible_start_state_ids
        new_gym_env.starting_states = self.starting_states
        new_gym_env.logger = self.logger # Logger can be shared or re-initialized

        return new_gym_env

    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the Gym environment as a dictionary."""
        return {
            "current_maze_layout": self.current_maze_layout,
            "rows": self.rows,
            "cols": self.cols,
            "agent_pos": self.agent_pos,
            "t1_pos": self.t1_pos,
            "t2_pos": self.t2_pos,
            "walls": list(self.walls), # Convert set to list for serialization if needed
            "current_step_internal": self.current_step_internal,
            "game_over_internal": self.game_over_internal,
            "final_message": self.final_message,
            "current_llm_prompt_gym": self.current_llm_prompt,
            "current_step_gym": self.current_step_gym,
            "max_steps": self.max_steps,
            "terminal_gym": self.terminal_gym,
            "rng_state_gym": self.rng.getstate()
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set the Gym environment to a specific state."""
        self.current_maze_layout = state["current_maze_layout"]
        self.rows = state["rows"]
        self.cols = state["cols"]
        self.agent_pos = tuple(state["agent_pos"]) if state["agent_pos"] else None
        self.t1_pos = tuple(state["t1_pos"]) if state["t1_pos"] else None
        self.t2_pos = tuple(state["t2_pos"]) if state["t2_pos"] else None
        self.walls = set(map(tuple, state["walls"])) 

        self.current_step_internal = state["current_step_internal"]
        self.game_over_internal = state["game_over_internal"]
        self.final_message = state["final_message"]
        
        self.current_llm_prompt = state["current_llm_prompt_gym"]
        self.current_step_gym = state["current_step_gym"]
        self.max_steps = state["max_steps"]
        self.terminal_gym = state["terminal_gym"]
        self.rng.setstate(state["rng_state_gym"])
        
        # Ensure logger is available
        if not hasattr(self, 'logger') or self.logger is None:
            self.logger = logging.getLogger(f"verifiers.envs.{self.__class__.__name__}")

    @property
    def current_state_id(self) -> Optional[str]:
        """Returns a unique identifier for the current game state (maze layout + agent position)."""
        if self.agent_pos:
            try:
                maze_index = ALL_MAZES.index(self.current_maze_layout)
                return f"maze{maze_index}_pos{self.agent_pos[0]}-{self.agent_pos[1]}"
            except ValueError: 
                 return f"custommaze_pos{self.agent_pos[0]}-{self.agent_pos[1]}"
        return None

    @property
    def current_state_desc(self) -> Optional[str]: 
        """Returns the current LLM prompt, which acts as the state description for the agent."""
        return self.current_llm_prompt

# Example Usage
if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.info("Creating TwoTreasuresMazeGymEnv instance...")
    gym_env = TwoTreasuresMazeGymEnv(max_steps=15, seed=42)
    logger.info("Successfully created TwoTreasuresMazeGymEnv instance.")

    logger.info(f"System Prompt:\n{gym_env.system_prompt}")
    
    obs, info = gym_env.reset(start_state_id='0') 
    logger.info(f"\nInitial Observation (Maze 0):\n{obs}")
    logger.info(f"Initial Info: {info}")
    logger.info(f"Available actions: {gym_env.get_available_actions()}")
    logger.info(f"Current state ID: {gym_env.current_state_id}")


    actions_to_test = [
        "<thinking>move right</thinking>",
        "<thinking>move down</thinking>", 
        "<thinking>fly</thinking>", # Invalid action inside thinking tags
        "move up", # Missing thinking tags (format error)
        "<thinking>move right</thinking>", # Valid: (2,2) -> (2,3) should be T1 in MAZE_1
        "<action>move left</action>" # Wrong tags (format error)
    ]

    for i, action_str in enumerate(actions_to_test):
        if gym_env.terminal_gym:
            logger.info("Episode ended early.")
            break
        logger.info(f"\n--- Step {gym_env.current_step_gym + 1} ---")
        logger.info(f"Gym Env Current State ID: {gym_env.current_state_id}")
        logger.info(f"Taking action: {action_str}")
        obs, reward, done, truncated, info_step = gym_env.step(action_str)
        
        logger.info(f"Next Observation:\n{obs}")
        logger.info(f"Reward: {reward}")
        logger.info(f"Done: {done}")
        logger.info(f"Truncated: {truncated}")
        logger.info(f"Info: {info_step}")
        logger.info(f"Internal agent_pos: {gym_env.agent_pos}, game_over: {gym_env.game_over_internal}, step: {gym_env.current_step_internal}")
        if done:
            logger.info("Episode finished.")
            

    logger.info("\n--- Testing Clone and State Setting ---")
    gym_env.reset(start_state_id='1', seed=55) 
    gym_env.step("<thinking>move right</thinking>")
    gym_env.step("<thinking>move right</thinking>")
    
    original_state_data = gym_env.get_state()
    original_llm_prompt_val = gym_env.current_llm_prompt
    original_agent_pos_val = gym_env.agent_pos
    original_gym_step_val = gym_env.current_step_gym

    cloned_env = gym_env.clone()
    
    cloned_env.step("<thinking>move down</thinking>")
    logger.info(f"Original env agent_pos after clone steps: {gym_env.agent_pos} (Should be {original_agent_pos_val})")
    logger.info(f"Cloned env agent_pos after its step: {cloned_env.agent_pos}")

    assert gym_env.agent_pos == original_agent_pos_val, "Clone affected original env's agent_pos"
    assert gym_env.current_llm_prompt == original_llm_prompt_val, "Clone affected original LLM prompt"
    assert cloned_env.agent_pos != original_agent_pos_val, "Cloned env did not step independently"


    gym_env.reset(start_state_id='0') 
    gym_env.set_state(original_state_data)

    assert gym_env.current_llm_prompt == original_llm_prompt_val, "State setting failed for llm_prompt"
    assert gym_env.agent_pos == original_agent_pos_val, "State setting failed for agent_pos"
    assert gym_env.current_step_gym == original_gym_step_val, "State setting failed for current_step_gym"
    logger.info("Clone and state setting test passed (basic checks).")

    logger.info("\n--- Testing Max Steps ---")
    gym_env.reset(start_state_id='0', seed=77)
    logger.info(f"Max steps for this run: {gym_env.max_steps}")
    # Store done and truncated from the last step in the loop to check after loop finishes
    last_done_in_loop = False
    last_truncated_in_loop = False

    for i in range(gym_env.max_steps + 2):
        action_to_take = gym_env.rng.choice([
             "<thinking>move right</thinking>",
             "<thinking>move down</thinking>",
        ])
        obs, reward, done, truncated, info_step = gym_env.step(action_to_take)
        logger.info(f"Step {gym_env.current_step_gym}: Action: {gym_env._parse_action(action_to_take)}, Reward: {reward:.2f}, Done: {done}, Truncated: {truncated}, Info: {info_step.get('game_event', info_step.get('error'))}")
        last_done_in_loop = done
        last_truncated_in_loop = truncated
        if done:
            logger.info(f"Episode ended at step {gym_env.current_step_gym}. Done: {done}, Truncated: {truncated}")
            logger.info(f"Final LLM Prompt (should indicate game over):\n{gym_env.current_llm_prompt}")
            break
    
    if not (last_done_in_loop or last_truncated_in_loop): 
        logger.error("Episode did not end after max_steps + 2 !!")
