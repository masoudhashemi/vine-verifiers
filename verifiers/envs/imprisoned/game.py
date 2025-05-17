import argparse

import gymnasium as gym

from imprisoned_env import ImprisonedEnv


def play_game():
    """Runs the text-based game loop for 'Imprisoned'."""
    env = ImprisonedEnv()
    observation = env.reset()
    env.render()

    done = False
    while not done:
        actions = env.get_available_actions()

        if not actions:
            print("\nNo actions available. Game over!")
            break

        print("\nAvailable actions:")
        for i, action in enumerate(actions, start=1):
            action_details = env.states[env.current_state]["actions"][action]
            print(f"{i}. {action}: {action_details.get('description', 'No description')}")

        try:
            choice = int(input("\nChoose an action (number): ")) - 1
            if 0 <= choice < len(actions):
                observation, reward, done, _ = env.step(choice)
                print("\n" + "=" * 50)  # Add visual separator
                env.render()
            else:
                print("Invalid choice. Please pick a valid number.")
        except ValueError:
            print("Invalid input. Enter a number corresponding to an action.")

    print("\n=== GAME OVER ===")
    if observation == "escape_success":
        print("🎉 You have successfully escaped! Congratulations!")
    elif reward == -1:
        state_desc = env.get_state_description()
        print(f"💀 Game Over: {state_desc}")
    else:
        print("💀 You failed to escape. Better luck next time!")


def play_with_agent():
    """Play the game with assistance from a trained Q-learning agent."""
    try:
        from qlearning_agent import play_interactive_game

        play_interactive_game()
    except ImportError:
        print("Error: Q-learning agent module not found. Make sure qlearning_agent.py is in the same directory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play the Imprisoned text adventure game")
    parser.add_argument("--agent", action="store_true", help="Play with AI agent assistance")

    args = parser.parse_args()

    print("\n=== Welcome to *Imprisoned* ===")

    if args.agent:
        play_with_agent()
    else:
        play_game()
