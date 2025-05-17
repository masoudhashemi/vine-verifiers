from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from torch.utils.data import Dataset

@dataclass
class VineBufferSample:
    states: List[Any]
    actions: List[str]
    # action_token_ids removed
    rewards: List[float]
    values: List[float]
    old_log_probs: List[np.ndarray]
    advantages: List[float]
    returns: List[float]
    dones: List[bool]
    action_masks: List[np.ndarray]

class VineBuffer(Dataset):
    """
    Buffer for storing trajectories collected using VinePPO (MCTS variant).
    Computes returns and advantages using GAE or TD, based on stored values.
    """
    def __init__(
        self,
        buffer_size: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = "cuda", # Note: device is set but buffer stores numpy/lists
        advantage_type: str = "td", # Changed default to td, consistent with some MCTS/PPO papers
    ):
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device # Store device preference, though not directly used on stored data
        self.advantage_type = advantage_type
        if self.advantage_type not in ["gae", "td"]:
            raise ValueError(f"Invalid advantage_type: {self.advantage_type}. Must be 'gae' or 'td'.")
        self.clear()

    def clear(self):
        """Clear buffer."""
        self.states = []
        self.actions = []
        # action_token_ids removed
        self.rewards = []
        self.values = [] # Stores V(s) estimates (from V-table/MC)
        self.old_log_probs = [] # Stores log probs from the policy at the time of action
        self.advantages = []
        self.returns = []
        self.dones = [] # Represents episode termination (done or truncated)
        self.action_masks = []
        self.size = 0

    def add(
        self,
        state: Any,
        action: str,
        # action_token_ids removed
        reward: float,
        value: float, # V(s) estimate
        log_prob: np.ndarray, # Log prob of action under policy
        action_mask: np.ndarray,
        done: bool = False, # True if step is terminal (done or truncated)
    ):
        """
        Add a new transition to the buffer.
        Handles buffer size limit by evicting the oldest transition if full.
        """
        # Check if buffer is full and evict the oldest transition
        if self.size >= self.buffer_size:
            self.states.pop(0)
            self.actions.pop(0)
            # action_token_ids removed
            self.rewards.pop(0)
            self.values.pop(0)
            self.old_log_probs.pop(0)
            self.dones.pop(0)
            self.action_masks.pop(0)
            # Adjust size before incrementing
            self.size = self.buffer_size - 1

        # Add the new transition
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.old_log_probs.append(log_prob)
        self.dones.append(done)
        self.action_masks.append(action_mask)
        self.size += 1

        # Invalidate old advantages/returns when new data is added
        self.advantages = []
        self.returns = []

    def compute_returns_and_advantages(self):
        """
        Compute returns and advantages (GAE or TD) based on stored rewards and values.
        Must be called before get().
        """
        # Avoid recomputation if already done for the current data
        if len(self.advantages) == self.size and len(self.returns) == self.size and self.size > 0:
            return

        if self.size == 0:
            self.advantages = []
            self.returns = []
            return

        T = self.size
        advantages = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)
        values_np = np.array(self.values, dtype=np.float32)
        rewards_np = np.array(self.rewards, dtype=np.float32)
        dones_np = np.array(self.dones, dtype=bool)

        if self.advantage_type == "gae":
            last_gae_lam = 0.0
            # Note: No special handling for negative values here; relies on V-table accuracy
            for t in reversed(range(T)):
                if t == T - 1:
                    next_non_terminal = 0.0 # No next state if last step
                    next_value = 0.0 # Assume V=0 after final state
                else:
                    next_non_terminal = 1.0 - dones_np[t+1] # 1 if next state is not terminal
                    next_value = values_np[t+1]

                delta = rewards_np[t] + self.gamma * next_value * next_non_terminal - values_np[t]
                last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
                advantages[t] = last_gae_lam
            returns = advantages + values_np

        elif self.advantage_type == "td":
            for t in range(T):
                if t == T - 1:
                    next_non_terminal = 0.0
                    next_value = 0.0
                else:
                    next_non_terminal = 1.0 - dones_np[t+1]
                    next_value = values_np[t+1]

                # Calculate TD target (used for return)
                td_target = rewards_np[t] + self.gamma * next_value * next_non_terminal
                returns[t] = td_target

                # Calculate TD advantage (TD Error)
                advantage = td_target - values_np[t]
                advantages[t] = advantage
        else:
            raise ValueError(f"Invalid advantage_type: {self.advantage_type}")

        self.advantages = advantages.tolist() # Store as list
        self.returns = returns.tolist() # Store as list

    def get_statistics(self) -> Dict[str, float]:
        """
        Get current buffer statistics (means of lists).
        """
        # Ensure advantages/returns are computed if needed
        if len(self.advantages) != self.size:
            self.compute_returns_and_advantages()

        return {
            "mean_reward": float(np.mean(self.rewards)) if self.size > 0 else 0.0,
            "mean_value": float(np.mean(self.values)) if self.size > 0 else 0.0,
            "mean_advantage": float(np.mean(self.advantages)) if self.size > 0 else 0.0,
            "mean_return": float(np.mean(self.returns)) if self.size > 0 else 0.0,
            "buffer_size": self.size
        }

    def get(self, batch_size: Optional[int] = None) -> VineBufferSample:
        """
        Get a batch or all transitions from the buffer.
        Requires compute_returns_and_advantages() to be called first.
        """
        if len(self.advantages) != self.size or len(self.returns) != self.size:
            print("Warning: compute_returns_and_advantages() should be called before get(). Computing now.")
            self.compute_returns_and_advantages()

        if batch_size is None or batch_size >= self.size:
            indices = np.arange(self.size)
        else:
            indices = np.random.permutation(self.size)[:batch_size]

        # Gather data using selected indices
        batch_states = [self.states[i] for i in indices]
        batch_actions = [self.actions[i] for i in indices]
        batch_rewards = [self.rewards[i] for i in indices]
        batch_values = [self.values[i] for i in indices]
        batch_old_log_probs = [self.old_log_probs[i] for i in indices]
        batch_advantages = [self.advantages[i] for i in indices]
        batch_returns = [self.returns[i] for i in indices]
        batch_dones = [self.dones[i] for i in indices]
        batch_action_masks = [self.action_masks[i] for i in indices]

        return VineBufferSample(
            states=batch_states,
            actions=batch_actions,
            rewards=batch_rewards,
            values=batch_values,
            old_log_probs=batch_old_log_probs,
            advantages=batch_advantages,
            returns=batch_returns,
            dones=batch_dones,
            action_masks=batch_action_masks
        )

    def __len__(self) -> int:
        return self.size 