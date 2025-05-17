import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from verifiers.envs.imprisoned.imprisoned_env import ImprisonedEnv


class QLearningAgent:
    """Q-learning agent that learns to play the Imprisoned game."""

    def __init__(
        self,
        env,
        learning_rate=0.1,
        discount_factor=0.95,
        exploration_rate=1.0,
        min_exploration_rate=0.01,
        exploration_decay_rate=0.001,
    ):
        """Initialize the Q-learning agent.

        Args:
            env: The Imprisoned environment
            learning_rate: Alpha - learning rate for Q-value updates
            discount_factor: Gamma - discount factor for future rewards
            exploration_rate: Epsilon - initial exploration rate
            min_exploration_rate: Minimum exploration rate
            exploration_decay_rate: Rate at which exploration decreases
        """
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay_rate = exploration_decay_rate

        # Initialize Q-table as a dictionary
        # Keys: (state, action) tuples
        # Values: Q-values
        self.q_table = {}

        # Map state names to indices for easier tracking
        self.state_to_idx = {}
        self.idx_to_state = {}
        for i, state in enumerate(env.states.keys()):
            self.state_to_idx[state] = i
            self.idx_to_state[i] = state

    def get_q_value(self, state, action):
        """Get Q-value for a state-action pair."""
        if (state, action) not in self.q_table:
            self.q_table[(state, action)] = 0.0
        return self.q_table[(state, action)]

    def choose_action(self, state):
        """Choose an action using epsilon-greedy policy."""
        available_actions = self.env.get_available_actions()

        if not available_actions:
            return None

        # Explore: choose a random action
        if random.random() < self.exploration_rate:
            return random.choice(available_actions)

        # Exploit: choose the best action based on Q-values
        q_values = [self.get_q_value(state, action) for action in available_actions]
        max_q = max(q_values)
        # If multiple actions have the same max Q-value, randomly select one
        best_actions = [i for i, q in enumerate(q_values) if q == max_q]
        action_idx = random.choice(best_actions)

        return available_actions[action_idx]

    def update_q_value(self, state, action, reward, next_state):
        """Update Q-value for a state-action pair."""
        # Get the best Q-value for the next state
        next_actions = self.env.get_available_actions()
        if next_actions:
            next_q_values = [self.get_q_value(next_state, next_action) for next_action in next_actions]
            max_next_q = max(next_q_values) if next_q_values else 0
        else:
            max_next_q = 0

        # Q-learning formula
        current_q = self.get_q_value(state, action)
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[(state, action)] = new_q

    def decay_exploration_rate(self, episode, total_episodes):
        """Decay exploration rate over time."""
        self.exploration_rate = self.min_exploration_rate + (1.0 - self.min_exploration_rate) * np.exp(
            -self.exploration_decay_rate * episode
        )

    def train(self, num_episodes=5000):
        """Train the agent for a specified number of episodes."""
        rewards_per_episode = []
        steps_per_episode = []
        success_rate = []
        success_window = []

        for episode in tqdm(range(num_episodes), desc="Training"):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            done = False

            while not done:
                action = self.choose_action(state)
                if action is None:
                    break

                action_idx = self.env.get_available_actions().index(action)
                next_state, reward, done, _ = self.env.step(action_idx)

                self.update_q_value(state, action, reward, next_state)

                state = next_state
                total_reward += reward
                steps += 1

                if steps > 100:  # Prevent infinite loops
                    break

            # Track metrics
            rewards_per_episode.append(total_reward)
            steps_per_episode.append(steps)
            success_window.append(1 if total_reward > 0 else 0)

            # Calculate success rate over last 100 episodes
            if episode >= 100:
                success_window.pop(0)
            success_rate.append(sum(success_window) / len(success_window))

            # Decay exploration rate
            self.decay_exploration_rate(episode, num_episodes)

        return rewards_per_episode, steps_per_episode, success_rate

    def play_game(self, max_steps=100, render=True, start_state_id: str = None):
        """Play a single game using the trained policy."""
        state = self.env.reset(start_state_id=start_state_id)
        total_reward = 0
        steps = 0
        done = False

        if render:
            self.env.render()

        while not done and steps < max_steps:
            available_actions = self.env.get_available_actions()
            if not available_actions:
                break

            # Always exploit in evaluation mode
            q_values = [self.get_q_value(state, action) for action in available_actions]
            max_q = max(q_values)
            best_actions = [i for i, q in enumerate(q_values) if q == max_q]
            action_idx = random.choice(best_actions)
            action = available_actions[action_idx]

            if render:
                print(f"\nAgent chooses: {action}")

            action_idx = available_actions.index(action)
            next_state, reward, done, _ = self.env.step(action_idx)

            state = next_state
            total_reward += reward
            steps += 1

            if render:
                print("\n" + "=" * 50)
                self.env.render()

        if render:
            print("\n=== GAME OVER ===")
            if total_reward > 0:
                print("🎉 Agent successfully escaped! Congratulations!")
            else:
                print("💀 Agent failed to escape.")
            print(f"Total reward: {total_reward}")
            print(f"Steps taken: {steps}")

        return total_reward, steps

    def save_q_table(self, filename="q_table.pkl"):
        """Save the Q-table to a file."""
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, filename="q_table.pkl"):
        """Load the Q-table from a file."""
        try:
            with open(filename, "rb") as f:
                self.q_table = pickle.load(f)
            print(f"Q-table loaded from {filename}")
        except FileNotFoundError:
            print(f"File {filename} not found. Starting with an empty Q-table.")

    def plot_training_results(self, rewards, steps, success_rate):
        """Plot training metrics."""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

        # Plot rewards
        ax1.plot(rewards)
        ax1.set_title("Rewards per Episode")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Total Reward")

        # Plot steps
        ax2.plot(steps)
        ax2.set_title("Steps per Episode")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Steps")

        # Plot success rate
        ax3.plot(success_rate)
        ax3.set_title("Success Rate (moving average over 100 episodes)")
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Success Rate")
        ax3.set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig("training_results.png")
        plt.show()

    def analyze_policy(self):
        """Analyze the learned policy to identify preferred actions for each state."""
        policy = {}

        for state in self.env.states:
            # Save the current state
            current_state = self.env.current_state
            self.env.current_state = state

            available_actions = self.env.get_available_actions()
            if not available_actions:
                policy[state] = "Terminal state or no actions available"
                continue

            q_values = [(action, self.get_q_value(state, action)) for action in available_actions]
            q_values.sort(key=lambda x: x[1], reverse=True)

            policy[state] = q_values

            # Restore the original state
            self.env.current_state = current_state

        return policy


def train_agent(episodes=5000, save_file="q_table.pkl"):
    """Train a Q-learning agent and save its Q-table."""
    env = ImprisonedEnv()
    agent = QLearningAgent(env)

    print(f"Training agent for {episodes} episodes...")
    rewards, steps, success_rate = agent.train(num_episodes=episodes)

    agent.save_q_table(save_file)
    print(f"Q-table saved to {save_file}")

    agent.plot_training_results(rewards, steps, success_rate)

    return agent


def evaluate_agent(load_file="q_table.pkl", num_games=100):
    """Evaluate a trained agent over multiple games."""
    env = ImprisonedEnv()
    agent = QLearningAgent(env)
    agent.load_q_table(load_file)

    print(f"Evaluating agent over {num_games} games...")
    rewards = []
    steps = []

    for i in range(num_games):
        reward, step = agent.play_game(render=False)
        rewards.append(reward)
        steps.append(step)

    success_rate = sum(1 for r in rewards if r > 0) / num_games
    avg_steps = sum(steps) / num_games

    print(f"Success rate: {success_rate:.2%}")
    print(f"Average steps: {avg_steps:.2f}")
    
    # Separate steps for successful and failed games
    successful_steps = [s for r, s in zip(rewards, steps) if r > 0]
    failed_steps = [s for r, s in zip(rewards, steps) if r <= 0]
    
    # Plot histograms on the same figure
    plt.figure(figsize=(12, 6))
    
    # Create bins that work for both datasets
    max_steps = max(steps) if steps else 0
    bins = range(0, max_steps + 5, 2)  # Adjust bin size as needed
    
    # Plot successful games
    plt.hist(successful_steps, bins=bins, alpha=0.7, color='green', 
             edgecolor='black', label='Successful Escapes')
    
    # Plot failed games
    plt.hist(failed_steps, bins=bins, alpha=0.7, color='red', 
             edgecolor='black', label='Failed Attempts')
    
    plt.title('Distribution of Steps per Game')
    plt.xlabel('Number of Steps')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.legend()
    plt.savefig("steps_histogram.png")
    plt.show()

    return success_rate, avg_steps


def play_interactive_game(load_file="q_table.pkl"):
    """Play a game where the agent suggests moves but the user decides."""
    env = ImprisonedEnv()
    agent = QLearningAgent(env)
    agent.load_q_table(load_file)

    state = env.reset()
    env.render()

    done = False
    while not done:
        actions = env.get_available_actions()

        if not actions:
            print("\nNo actions available. Game over!")
            break

        # Get agent's suggestion
        q_values = [(action, agent.get_q_value(state, action)) for action in actions]
        q_values.sort(key=lambda x: x[1], reverse=True)
        suggested_action = q_values[0][0]

        print("\nAvailable actions:")
        for i, action in enumerate(actions, start=1):
            action_details = env.states[env.current_state]["actions"][action]
            q_value = agent.get_q_value(state, action)
            print(f"{i}. {action}: {action_details.get('description', 'No description')} (Q-value: {q_value:.2f})")

        print(f"\nAgent suggests: {suggested_action}")

        try:
            choice = int(input("\nChoose an action (number) or 0 to follow agent's suggestion: "))
            if choice == 0:
                choice = actions.index(suggested_action) + 1

            if 1 <= choice <= len(actions):
                action_idx = choice - 1
                state, reward, done, _ = env.step(action_idx)
                print("\n" + "=" * 50)
                env.render()
            else:
                print("Invalid choice. Please pick a valid number.")
        except ValueError:
            print("Invalid input. Enter a number corresponding to an action.")

    print("\n=== GAME OVER ===")
    if state == "escape_success":
        print("🎉 You have successfully escaped! Congratulations!")
    elif reward == -1:
        state_desc = env.get_state_description()
        print(f"💀 Game Over: {state_desc}")
    else:
        print("💀 You failed to escape. Better luck next time!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or evaluate a Q-learning agent for the Imprisoned game")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "evaluate", "play", "analyze", "interactive"],
        default="train",
        help="Mode to run the script in",
    )
    parser.add_argument("--episodes", type=int, default=5000, help="Number of episodes for training")
    parser.add_argument("--games", type=int, default=100, help="Number of games for evaluation")
    parser.add_argument("--file", type=str, default="q_table.pkl", help="File to save/load Q-table")

    args = parser.parse_args()

    if args.mode == "train":
        agent = train_agent(episodes=args.episodes, save_file=args.file)
    elif args.mode == "evaluate":
        evaluate_agent(load_file=args.file, num_games=args.games)
    elif args.mode == "play":
        env = ImprisonedEnv()
        agent = QLearningAgent(env)
        agent.load_q_table(args.file)
        agent.play_game()
    elif args.mode == "analyze":
        env = ImprisonedEnv()
        agent = QLearningAgent(env)
        agent.load_q_table(args.file)
        policy = agent.analyze_policy()

        print("=== POLICY ANALYSIS ===")
        for state, actions in policy.items():
            if isinstance(actions, str):
                print(f"\n{state}: {actions}")
            else:
                print(f"\n{state}:")
                for action, q_value in actions[:3]:  # Show top 3 actions
                    print(f"  {action}: {q_value:.2f}")
    elif args.mode == "interactive":
        play_interactive_game(load_file=args.file)
