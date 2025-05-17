# Imprisoned: A Text-Based Prison Escape Game

## Overview

Imprisoned is a text-based adventure game where you play as a prisoner attempting to escape from a high-security prison. Navigate through various locations, interact with other prisoners, collect useful items, and make strategic decisions to find your way to freedom.

The game features:

- Multiple starting scenarios
- Branching storylines with meaningful choices
- An inventory system for collecting and using items
- Various escape strategies (stealth, deception, force, etc.)
- A Q-learning AI agent that can learn optimal escape strategies

## Game Structure

The game is built using Python with the OpenAI Gymnasium framework, allowing for a reinforcement learning approach to both gameplay and AI training. The game state is defined in a YAML configuration file, making it easy to modify and expand.

### Key Components

- `imprisoned.yaml`: Contains all game states, actions, and transitions
- `imprisoned_env.py`: The Gymnasium environment that manages game state
- `game.py`: The main game loop for human players
- `qlearning_agent.py`: Implementation of a Q-learning agent that can play the game

## Playing the Game

### As a Human Player

To play the game normally:

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Run the game:

```bash
python game.py
```

3. Follow the on-screen instructions to navigate through the game.

### With AI Assistance

To play with suggestions from the AI agent:

```bash
python game.py --agent
```

This mode will show you the AI's recommended action based on its learned Q-values, but you still make the final decision.

## The Q-Learning Agent

The game includes a reinforcement learning agent that can learn to play the game through trial and error using the Q-learning algorithm.

### Training the Agent

To train the agent from scratch:

```bash
python qlearning_agent.py --mode train --episodes 5000
```

This will run the agent through 5000 episodes, during which it will learn which actions lead to successful escapes. The training results will be saved to `q_table.pkl` and visualized in `training_results.png`.

### Evaluating the Agent

To test how well the trained agent performs:

```bash
python qlearning_agent.py --mode evaluate --games 100
```

This will run 100 games and report the agent's success rate and average number of steps to escape.

### Watching the Agent Play

To watch the agent play a game using its learned strategy:

```bash
python qlearning_agent.py --mode play
```

### Analyzing the Agent's Strategy

To see what the agent has learned about the best actions in each state:

```bash
python qlearning_agent.py --mode analyze
```

## How the Q-Learning Works

The agent learns by:

1. Exploring the game environment through random actions
2. Receiving rewards for successful escapes
3. Updating its Q-values (expected future rewards) for each state-action pair
4. Gradually shifting from exploration to exploitation of known good strategies

The agent uses an epsilon-greedy policy, meaning it mostly chooses the best known action but occasionally tries random actions to discover new strategies.

## Game Design

The prison escape scenario offers multiple paths to freedom:

- **Social Engineering**: Befriend other prisoners, manipulate guards
- **Stealth**: Use air ducts, tunnels, or disguises
- **Deception**: Fake illness, create distractions
- **Force**: Overpower guards, join prison riots
- **Psychological**: Maintain mental clarity in solitary confinement

Each approach has its own risks and rewards, creating a complex decision space for both human players and the AI agent to navigate.

## Troubleshooting

### Missing Actions in States

If you see warnings about states having no actions, you need to update the YAML file to ensure all referenced states have action definitions. The file `transition_states_fix.yaml` contains definitions for commonly referenced transition states that may be missing actions.

To fix these warnings:

1. Copy the contents of `transition_states_fix.yaml` into your `imprisoned.yaml` file
2. Make sure all states referenced in transitions either have actions or are marked as terminal
3. Check for typos in state names that might be causing references to non-existent states

### Common Issues

- **"State not found in YAML"**: Check for typos in state names in your YAML file
- **"No actions available"**: Ensure the current state has defined actions
- **Agent getting stuck**: Make sure there are no cycles without terminal states

## Extending the Game

### Adding New States

To add new states to the game, edit the `imprisoned.yaml` file:

```yaml
states:
  new_state_name:
    description: "Description of the new state"
    actions:
      action_name:
        description: "Description of the action"
        probabilities:
          next_state_1: 0.7
          next_state_2: 0.3
```

### Adding Items

To add new items that can be collected:

```yaml
states:
  some_state:
    actions:
      find_item:
        description: "Find a new item"
        grants: item_name
        next_state: next_state
```

### Creating Requirements

To make actions require specific items:

```yaml
actions:
  use_key:
    description: "Use the key to unlock the door"
    conditions:
      requires: key
    next_state: unlocked_door
```
