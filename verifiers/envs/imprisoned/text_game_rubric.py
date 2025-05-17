from typing import List, Dict, Any, Callable
import re

from trl.trainer.grpo_trainer import RewardFunc


class TextGameRubric:
    """Rubric for evaluating performance in text-based adventure games."""

    def __init__(self):
        """Initialize the text game rubric."""
        pass

    def get_reward_funcs(self) -> List[RewardFunc]:
        """Get the reward functions for evaluation."""
        return [
            self.escape_success_reward,
            self.exploration_reward,
            self.state_transition_reward
        ]

    def escape_success_reward(self, prompts: List[List[Dict[str, str]]], 
                             completions: List[List[Dict[str, str]]], 
                             answer: List[Any] = None, 
                             **kwargs) -> List[float]:
        """Reward function for successful escape.
        
        Args:
            prompts: List of conversation prompts
            completions: List of model completions (conversation turns)
            answer: List of reference answers (not used)
            
        Returns:
            List of rewards (3.0 for successful escape, 0.0 otherwise)
        """
        rewards = []
        
        for completion in completions:
            # Check if any message indicates successful escape
            # Look for success message in user messages (environment responses)
            success = any("GAME OVER: You have successfully escaped" in msg.get("content", "") 
                         for msg in completion if msg.get("role") == "user")
            
            rewards.append(3.0 if success else 0.0)
            
        return rewards

    def exploration_reward(self, prompts: List[List[Dict[str, str]]], 
                          completions: List[List[Dict[str, str]]], 
                          answer: List[Any] = None, 
                          **kwargs) -> List[float]:
        """Reward function for exploration (visiting different states).
        
        Args:
            prompts: List of conversation prompts
            completions: List of model completions (conversation turns)
            answer: List of reference answers (not used)
            
        Returns:
            List of rewards based on unique states visited
        """
        rewards = []
        
        for completion in completions:
            # Extract state descriptions from environment messages
            env_messages = [msg.get("content", "") for msg in completion if msg.get("role") == "user"]
            
            # Count unique state descriptions (first paragraph of each message)
            unique_states = set()
            for msg in env_messages:
                # Get the first paragraph as a proxy for state description
                first_para = msg.split("\n\n")[0] if "\n\n" in msg else msg
                unique_states.add(first_para)
            
            # Reward based on unique states visited (normalized)
            # More unique states = more exploration
            num_unique = len(unique_states)
            reward = min(1.0, num_unique / 10.0)  # Cap at 1.0, normalize by expecting ~10 states
            
            rewards.append(reward)
            
        return rewards
        
    def state_transition_reward(self, prompts: List[List[Dict[str, str]]], 
                               completions: List[List[Dict[str, str]]], 
                               answer: List[Any] = None, 
                               **kwargs) -> List[float]:
        """Reward function for successful state transitions.
        
        Args:
            prompts: List of conversation prompts
            completions: List of model completions (conversation turns)
            answer: List of reference answers (not used)
            
        Returns:
            List of rewards based on successful state transitions
        """
        rewards = []
        
        for completion in completions:
            # Extract environment messages
            env_messages = [msg.get("content", "") for msg in completion if msg.get("role") == "user"]
            
            # Track state transitions by looking for state markers
            states = []
            for msg in env_messages:
                state_marker_match = re.search(r'\[STATE: ([^\]]+)\]', msg)
                if state_marker_match:
                    state_name = state_marker_match.group(1)
                    states.append(state_name)
            
            # Count unique state transitions (changes from one state to another)
            transitions = 0
            for i in range(1, len(states)):
                if states[i] != states[i-1]:
                    transitions += 1
            
            # Reward based on number of successful transitions
            # More transitions = more progress through the game
            reward = min(1.0, transitions / 10.0)  # Cap at 1.0, normalize by expecting ~5 transitions
            
            rewards.append(reward)
            
        return rewards 