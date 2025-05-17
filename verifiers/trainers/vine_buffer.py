from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from torch.utils.data import Dataset

@dataclass
class VineBufferSample:
    states: List[Any]
    actions: List[str]
    action_token_ids: List[List[int]]
    rewards: List[float]
    values: List[float]
    old_log_probs: List[np.ndarray]
    advantages: List[float]
    returns: List[float]
    dones: List[bool]
    action_masks: List[np.ndarray]
    is_actions_valid: List[bool]

class VineBuffer(Dataset):
    """
    Buffer for storing trajectories collected using VinePPO.
    Computes returns and advantages using MC rollouts and GAE.
    """
    def __init__(
        self,
        buffer_size: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = "cuda",
        advantage_type: str = "td",
    ):
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.advantage_type = advantage_type
        if self.advantage_type not in ["gae", "td"]:
            raise ValueError(f"Invalid advantage_type: {self.advantage_type}. Must be 'gae' or 'td'.")
        self.clear()

    def clear(self):
        """Clear buffer."""
        self.states = []
        self.actions = []
        self.action_token_ids = []
        self.rewards = []
        self.values = []
        self.old_log_probs = []
        self.advantages = []
        self.returns = []
        self.dones = []
        self.action_masks = []
        self.is_actions_valid = []
        self.size = 0

    def add(
        self,
        state: Any,
        action: str,
        action_token_ids: List[int],
        reward: float,
        value: float,
        log_prob: np.ndarray,
        action_mask: np.ndarray,
        done: bool = False,
        is_action_valid: bool = False,
    ):
        """
        Add a new transition to the buffer.
        Handles buffer size limit by evicting the oldest transition if full.
        """
        # Check if buffer is full and evict the oldest transition
        if self.size >= self.buffer_size:
            self.states.pop(0)
            self.actions.pop(0)
            self.action_token_ids.pop(0)
            self.rewards.pop(0)
            self.values.pop(0)
            self.old_log_probs.pop(0)
            self.dones.pop(0)
            self.action_masks.pop(0)
            self.is_actions_valid.pop(0)
            # Size remains buffer_size
            self.size = self.buffer_size - 1 # Decrement size temporarily before incrementing later

        # Add the new transition
        self.states.append(state)
        self.actions.append(action)
        self.action_token_ids.append(action_token_ids)
        self.rewards.append(reward)
        self.values.append(value)
        self.old_log_probs.append(log_prob)
        self.dones.append(done)
        self.action_masks.append(action_mask)
        self.is_actions_valid.append(is_action_valid)
        
        # Increment size (up to buffer_size)
        self.size += 1

    def _propagate_values(self) -> List[float]:
        """
        Propagates values backward:
        - Terminal states get value 0.
        - Invalid action states get the value of the next valid state.
        - Valid action states keep their original estimated value.
        """
        if not self.rewards: # Should be caught by caller, but defensive check
            return []

        T = len(self.rewards)
        propagated_values = list(self.values)  # Start with raw V(s_t) estimates
        next_future_valid_value = 0.0

        for t in reversed(range(T)):
            if self.is_actions_valid[t]:  # Action at step t was valid
                # propagated_values[t] remains self.values[t] (already correct in the copy)
                next_future_valid_value = propagated_values[t] # This is self.values[t]
            else:  # Action at step t was invalid
                if self.dones[t]:
                    propagated_values[t] = 0.0
                    next_future_valid_value = 0.0
                else:
                    propagated_values[t] = next_future_valid_value
        
        return propagated_values

    def compute_returns_and_advantages(self):
        """
        Compute returns and advantages using Generalized Advantage Estimation (GAE).
        Properly handles episode boundaries, including those ending in truncation.
        """
        # Avoid recomputation if already done
        if len(self.advantages) == len(self.states) and len(self.returns) == len(self.states):
            if len(self.states) > 0:  # Only return if buffer is not empty
                return

        # Handle empty buffer case
        if not self.rewards:
            self.advantages = []
            self.returns = []
            return

        T = len(self.rewards)
        advantages = [0.0] * T
        returns = [0.0] * T

        # Propagate values first to get final values for GAE/TD computation
        final_values_for_computation = self._propagate_values()

        if self.advantage_type == "gae":
            gae_value_for_next_step = 0.0  # This is A_{t+1} when calculating A_t. For t=T-1, A_T = 0.
            for t in reversed(range(T)):
                v_s_t = final_values_for_computation[t]
                
                # V(s_{t+1}) is final_values_for_computation[t+1] if t+1 exists, else 0.
                # _propagate_values ensures V(terminal_state) = 0.
                v_s_t_plus_1 = final_values_for_computation[t+1] if t + 1 < T else 0.0
                
                delta = self.rewards[t] + self.gamma * v_s_t_plus_1 - v_s_t
                
                # Mask for GAE propagation: 1.0 if s_{t+1} was non-terminal, else 0.0
                mask_propagate = 0.0
                if t + 1 < T and not self.dones[t+1]: # if s_{t+1} exists and is not terminal
                    mask_propagate = 1.0
                
                current_gae_advantage = delta + self.gamma * self.gae_lambda * mask_propagate * gae_value_for_next_step
                
                advantages[t] = current_gae_advantage
                returns[t] = current_gae_advantage + v_s_t
                
                gae_value_for_next_step = current_gae_advantage # A_t becomes A_{t+1} for the next iteration (t-1)

        elif self.advantage_type == "td":
            for t in range(T):
                v_s_t = final_values_for_computation[t]
                
                # V(s_{t+1})
                v_s_t_plus_1 = 0.0
                if not self.dones[t]: # If current state s_t is not terminal
                    if t + 1 < T and not self.dones[t+1]: # And next state s_{t+1} is in buffer and not terminal
                        v_s_t_plus_1 = final_values_for_computation[t+1]
                    # else, s_{t+1} is considered terminal (value 0 for TD target calculation)

                td_target = self.rewards[t]
                if not self.dones[t]: # Only add discounted next value if current is not terminal
                    td_target += self.gamma * v_s_t_plus_1
                
                returns[t] = td_target
                advantages[t] = td_target - v_s_t
        
        else:
            # This case should be caught in __init__, but added for safety
            raise ValueError(f"Invalid advantage_type: {self.advantage_type}")

        self.advantages = advantages
        self.returns = returns

    def get_statistics(self) -> Dict[str, float]:
        """
        Get current buffer statistics.
        """
        return {
            "mean_reward": np.mean(self.rewards) if self.rewards else 0,
            "mean_value": np.mean(self.values) if self.values else 0,
            "mean_advantage": np.mean(self.advantages) if self.advantages else 0,
            "mean_return": np.mean(self.returns) if self.returns else 0,
        }

    def get(self, batch_size: Optional[int] = None) -> VineBufferSample:
        """
        Get a batch of transitions from the buffer.
        """
        if len(self.advantages) != len(self.states) or len(self.returns) != len(self.states):
            self.compute_returns_and_advantages()
            
        if batch_size is None or batch_size >= len(self.states):
            return VineBufferSample(
                states=self.states,
                actions=self.actions,
                action_token_ids=self.action_token_ids,
                rewards=self.rewards,
                values=self.values,
                old_log_probs=self.old_log_probs,
                advantages=self.advantages,
                returns=self.returns,
                dones=self.dones,
                action_masks=self.action_masks,
                is_actions_valid=self.is_actions_valid
            )
            
        indices = np.random.permutation(len(self.states))[:batch_size]
        return VineBufferSample(
            states=[self.states[i] for i in indices],
            actions=[self.actions[i] for i in indices],
            action_token_ids=[self.action_token_ids[i] for i in indices],
            rewards=[self.rewards[i] for i in indices],
            values=[self.values[i] for i in indices],
            old_log_probs=[self.old_log_probs[i] for i in indices],
            advantages=[self.advantages[i] for i in indices],
            returns=[self.returns[i] for i in indices],
            dones=[self.dones[i] for i in indices],
            action_masks=[self.action_masks[i] for i in indices],
            is_actions_valid=[self.is_actions_valid[i] for i in indices]
        )

    def __len__(self) -> int:
        return self.size 
