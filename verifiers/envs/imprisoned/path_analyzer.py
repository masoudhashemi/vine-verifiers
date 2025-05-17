import yaml
import copy
from collections import deque, defaultdict
import random
import os
import signal

def load_game_config(file_path):
    """Load the game configuration from a YAML file."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def is_terminal_state(state_name, states):
    """Check if a state is terminal."""
    if state_name not in states:
        return False
    return states[state_name].get('terminal', False) or state_name == 'escape_success'

def get_possible_next_states(state_name, states, inventory=None):
    """Get all possible next states from the current state."""
    if inventory is None:
        inventory = set()
    
    if state_name not in states:
        return []
    
    next_states = []
    actions = states[state_name].get('actions', {})
    
    for action_name, action_data in actions.items():
        # Check if action has inventory requirements
        conditions = action_data.get('conditions', {})
        required_item = conditions.get('requires')
        
        if required_item and required_item not in inventory:
            continue
        
        # Handle different types of next state specifications
        if 'next_state' in action_data:
            next_states.append(action_data['next_state'])
        elif 'probabilities' in action_data:
            next_states.extend(action_data['probabilities'].keys())
    
    return next_states

def find_terminal_states(states):
    """Find all terminal states in the game."""
    terminal_states = []
    for state_name, state_data in states.items():
        if is_terminal_state(state_name, states):
            terminal_states.append(state_name)
    return terminal_states

def find_closest_terminal_states(state_name, states, max_depth=5):
    """Find the closest terminal states from a given state."""
    if state_name not in states:
        return []
    
    # Use BFS to find closest terminal states
    visited = set()
    queue = deque([(state_name, 0, [])])  # (state, depth, path)
    closest_terminals = []
    
    while queue:
        current, depth, path = queue.popleft()
        
        if current in visited:
            continue
        
        visited.add(current)
        
        # Check if this is a terminal state
        if is_terminal_state(current, states) and current != state_name:
            closest_terminals.append((current, depth, path + [current]))
            continue
        
        # Stop searching if we've reached max depth
        if depth >= max_depth:
            continue
        
        # Get next possible states
        next_states = get_possible_next_states(current, states)
        for next_state in next_states:
            if next_state not in visited:
                queue.append((next_state, depth + 1, path + [current]))
    
    # Sort by depth (closest first)
    closest_terminals.sort(key=lambda x: x[1])
    return closest_terminals

def analyze_paths(config):
    """Analyze all possible paths in the game and identify those exceeding 20 steps."""
    states = config['states']
    starting_states = config['starting_states']
    
    # Track the maximum path length from each starting state
    max_path_lengths = {}
    unreachable_terminals = set()
    
    # Track states that can lead to infinite loops
    potential_loops = set()
    
    # For each starting state, perform BFS to find all reachable states and path lengths
    for start in starting_states:
        if start not in states:
            continue
        
        visited = set()
        queue = deque([(start, 0, set())])  # (state, steps, inventory)
        max_length = 0
        reachable_terminals = set()
        
        while queue:
            current, steps, inventory = queue.popleft()
            
            if current in visited:
                continue
            
            visited.add(current)
            max_length = max(max_length, steps)
            
            # Check if this is a terminal state
            if is_terminal_state(current, states):
                reachable_terminals.add(current)
                continue
            
            # Get next possible states
            if current not in states:
                continue
                
            actions = states[current].get('actions', {})
            for action_name, action_data in actions.items():
                # Check inventory requirements
                conditions = action_data.get('conditions', {})
                required_item = conditions.get('requires')
                
                if required_item and required_item not in inventory:
                    continue
                
                # Update inventory if action grants an item
                new_inventory = inventory.copy()
                if 'grants' in action_data:
                    new_inventory.add(action_data['grants'])
                
                # Process next states
                if 'next_state' in action_data:
                    next_state = action_data['next_state']
                    if next_state not in visited:
                        queue.append((next_state, steps + 1, new_inventory))
                elif 'probabilities' in action_data:
                    for next_state in action_data['probabilities'].keys():
                        if next_state not in visited:
                            queue.append((next_state, steps + 1, new_inventory))
            
            # If we've exceeded 20 steps, mark this state as problematic
            if steps >= 20:
                potential_loops.add(current)
        
        max_path_lengths[start] = max_length
        
        # Find terminal states that aren't reachable from this starting point
        for state_name, state_data in states.items():
            if is_terminal_state(state_name, states) and state_name not in reachable_terminals:
                unreachable_terminals.add(state_name)
    
    return {
        'max_path_lengths': max_path_lengths,
        'potential_loops': potential_loops,
        'unreachable_terminals': unreachable_terminals
    }

def find_all_paths_from_start(config, start_state, max_depth=30):
    """Find all paths from a starting state using BFS, with inventory tracking."""
    states = config['states']
    
    if start_state not in states:
        return []
    
    # Use BFS to find all paths
    visited_states = set()  # Track visited states to avoid cycles
    paths = []
    
    # Queue entries: (current_state, path_so_far, depth, inventory)
    queue = deque([(start_state, [start_state], 0, set())])
    
    while queue:
        current, path, depth, inventory = queue.popleft()
        
        # If we've reached a terminal state, record the path
        if is_terminal_state(current, states):
            path_type = "success" if current == "escape_success" else "failure"
            paths.append((path, depth, path_type))
            continue
        
        # Skip if we've reached max depth - mark as incomplete
        if depth >= max_depth:
            paths.append((path, depth, "incomplete"))
            continue
        
        # Get next possible states with inventory consideration
        if current not in states:
            # If state doesn't exist, mark as error
            paths.append((path, depth, "error_state"))
            continue
            
        # Mark this state as visited in this path
        state_inventory_key = (current, frozenset(inventory))
        if state_inventory_key in visited_states:
            # Detected a cycle - mark as cyclic
            paths.append((path, depth, "cyclic"))
            continue
        
        visited_states.add(state_inventory_key)
        
        # Check if there are any actions
        actions = states[current].get('actions', {})
        if not actions:
            # Dead end - no actions available
            paths.append((path, depth, "dead_end"))
            continue
        
        has_next_state = False
        for action_name, action_data in actions.items():
            # Check inventory requirements
            conditions = action_data.get('conditions', {})
            required_item = conditions.get('requires')
            
            if required_item and required_item not in inventory:
                continue
            
            # Update inventory if action grants an item
            new_inventory = inventory.copy()
            if 'grants' in action_data:
                new_inventory.add(action_data['grants'])
            
            # Process next states
            if 'next_state' in action_data:
                has_next_state = True
                next_state = action_data['next_state']
                new_path = path + [next_state]
                queue.append((next_state, new_path, depth + 1, new_inventory))
            elif 'probabilities' in action_data:
                has_next_state = True
                for next_state in action_data['probabilities'].keys():
                    new_path = path + [next_state]
                    queue.append((next_state, new_path, depth + 1, new_inventory))
        
        # If no valid next states were found due to inventory constraints
        if not has_next_state:
            paths.append((path, depth, "inventory_blocked"))
    
    return paths

def find_states_exceeding_steps(config, max_steps=20):
    """Find states that can lead to paths exceeding the maximum steps."""
    states = config['states']
    starting_states = config['starting_states']
    
    problematic_states = set()
    long_paths = []
    
    # For each starting state, find paths that exceed max_steps
    for start in starting_states:
        if start not in states:
            continue
        
        print(f"Checking for long paths from {start}...")
        paths = find_all_paths_from_start(config, start, max_depth=max_steps+10)
        
        # Filter for paths that exceed max_steps and don't end in terminal states
        for path, length, path_type in paths:
            if length > max_steps and path_type not in ["success", "failure"]:
                # Add all states in this path to problematic states
                problematic_states.update(path)
                long_paths.append((path, length, path_type))
                
                # Print details about this long path
                print(f"Found long path ({length} steps, {path_type}): {' -> '.join(path[:3])}...{' -> '.join(path[-3:])}")
    
    print(f"Found {len(long_paths)} paths exceeding {max_steps} steps")
    return problematic_states

def fix_long_paths_recursively(config, max_steps=20):
    """Recursively modify the configuration to ensure all paths complete within max_steps."""
    # Create a deep copy of the config to modify
    new_config = copy.deepcopy(config)
    states = new_config['states']
    
    # Find problematic states
    problematic_states = find_states_exceeding_steps(config, max_steps)
    print(f"Found {len(problematic_states)} problematic states that could lead to paths > {max_steps} steps")
    if problematic_states:
        print(f"Some examples: {list(problematic_states)[:5]}")
    
    # Find all terminal states
    terminal_states = find_terminal_states(states)
    success_terminals = [s for s in terminal_states if s == 'escape_success']
    failure_terminals = [s for s in terminal_states if s != 'escape_success']
    
    print(f"Terminal states: {terminal_states}")
    print(f"Success terminals: {success_terminals}")
    print(f"Failure terminals: {failure_terminals}")
    
    # If no terminal states exist, create them
    if not success_terminals:
        states['escape_success'] = {
            'description': 'You successfully escape the prison!',
            'terminal': True
        }
        success_terminals = ['escape_success']
    
    if not failure_terminals:
        states['escape_failure'] = {
            'description': 'Your escape attempt has failed.',
            'terminal': True
        }
        failure_terminals = ['escape_failure']
    
    # For each problematic state, modify actions to lead to terminal states
    modified_states = 0
    
    # First, check for states with no actions and add actions to them
    for state_name, state_data in states.items():
        if is_terminal_state(state_name, states):
            continue
            
        actions = state_data.get('actions', {})
        if not actions:
            # Add a basic action if none exist
            closest_terminals = find_closest_terminal_states(state_name, states)
            if closest_terminals:
                closest_terminal = closest_terminals[0][0]
                actions['make_choice'] = {
                    'description': 'You must make a choice.',
                    'next_state': closest_terminal
                }
            else:
                # If no close terminals, use random ones
                actions['make_choice'] = {
                    'description': 'You must make a choice.',
                    'probabilities': {
                        random.choice(failure_terminals): 0.8,
                        random.choice(success_terminals): 0.2
                    }
                }
            states[state_name]['actions'] = actions
            modified_states += 1
            print(f"Added actions to state with no actions: {state_name}")
    
    # Now fix problematic states
    for state_name in problematic_states:
        if state_name not in states or is_terminal_state(state_name, states):
            continue
        
        # Find closest terminal states
        closest_terminals = find_closest_terminal_states(state_name, states)
        
        # If no close terminals found, use the global terminal states
        if not closest_terminals:
            closest_success = random.choice(success_terminals)
            closest_failure = random.choice(failure_terminals)
        else:
            # Try to find both success and failure terminals
            success_paths = [t for t in closest_terminals if t[0] in success_terminals]
            failure_paths = [t for t in closest_terminals if t[0] in failure_terminals]
            
            closest_success = success_paths[0][0] if success_paths else random.choice(success_terminals)
            closest_failure = failure_paths[0][0] if failure_paths else random.choice(failure_terminals)
        
        # Get existing actions
        actions = states[state_name].get('actions', {})
        
        # Modify an existing action if possible, otherwise add a new one
        if actions:
            # Choose a random action to modify
            action_name = random.choice(list(actions.keys()))
            action_data = actions[action_name]
            
            # Modify the action to have a chance of ending the game
            if 'next_state' in action_data:
                # Replace direct next_state with probabilities
                next_state = action_data['next_state']
                action_data.pop('next_state')
                action_data['probabilities'] = {
                    next_state: 0.5,
                    closest_failure: 0.4,
                    closest_success: 0.1  # Small chance of success
                }
                modified_states += 1
                print(f"Modified action {action_name} in state {state_name} to include terminal states")
            elif 'probabilities' in action_data:
                # Add terminal states to existing probabilities
                probs = action_data['probabilities']
                # Normalize existing probabilities to sum to 0.5
                total = sum(probs.values())
                if total > 0:  # Avoid division by zero
                    for state in list(probs.keys()):
                        probs[state] = probs[state] / total * 0.5
                else:
                    # If there are no probabilities, just set them to 0
                    for state in list(probs.keys()):
                        probs[state] = 0
                
                # Add terminal states with remaining probability
                probs[closest_failure] = 0.4
                probs[closest_success] = 0.1
                modified_states += 1
                print(f"Added terminal states to probabilistic action {action_name} in state {state_name}")
        else:
            # Add a new action if no actions exist
            actions['desperate_attempt'] = {
                'description': 'Make a desperate attempt to progress.',
                'probabilities': {
                    closest_failure: 0.8,
                    closest_success: 0.2
                }
            }
            modified_states += 1
            print(f"Added new action 'desperate_attempt' to state {state_name}")
        
        # Update the state's actions
        states[state_name]['actions'] = actions
    
    # Check for cycles and fix them
    print("Checking for cycles in the game...")
    cycle_states = find_cycle_states(new_config)
    if cycle_states:
        print(f"Found {len(cycle_states)} states involved in cycles")
        for state_name in cycle_states:
            if state_name not in states or is_terminal_state(state_name, states):
                continue
                
            # Add a chance to break out of the cycle
            actions = states[state_name].get('actions', {})
            if actions:
                # Add a new escape action
                actions['break_cycle'] = {
                    'description': 'Try to break free from this repetitive situation.',
                    'probabilities': {
                        random.choice(failure_terminals): 0.7,
                        random.choice(success_terminals): 0.3
                    }
                }
                states[state_name]['actions'] = actions
                modified_states += 1
                print(f"Added cycle-breaking action to state {state_name}")
    
    print(f"Modified {modified_states} states to ensure paths complete within {max_steps} steps")
    
    return new_config

def find_cycle_states(config):
    """Find states that are part of cycles in the game."""
    states = config['states']
    cycle_states = set()
    
    # Build a directed graph of state transitions
    graph = {}
    for state_name, state_data in states.items():
        if is_terminal_state(state_name, states):
            continue
            
        graph[state_name] = set()
        actions = state_data.get('actions', {})
        for action_name, action_data in actions.items():
            if 'next_state' in action_data:
                graph[state_name].add(action_data['next_state'])
            elif 'probabilities' in action_data:
                for next_state in action_data['probabilities'].keys():
                    graph[state_name].add(next_state)
    
    # Use DFS to find cycles
    def find_cycles_dfs(node, visited, path):
        if node in path:
            # Found a cycle
            cycle_index = path.index(node)
            cycle_states.update(path[cycle_index:])
            return
            
        if node in visited:
            return
            
        visited.add(node)
        path.append(node)
        
        if node in graph:
            for neighbor in graph[node]:
                find_cycles_dfs(neighbor, visited, path)
                
        path.pop()
    
    # Check each state for cycles
    for state_name in graph:
        find_cycles_dfs(state_name, set(), [])
    
    return cycle_states

def check_path_lengths(config, max_steps=20):
    """Check if all paths in the game complete within max_steps."""
    states = config['states']
    starting_states = config['starting_states']
    
    all_paths_valid = True
    
    # For each starting state, perform BFS to find all path lengths
    for start in starting_states:
        if start not in states:
            continue
        
        print(f"Checking paths from {start}...")
        
        paths = find_all_paths_from_start(config, start, max_depth=max_steps+5)
        long_paths = []
        non_terminal_paths = []
        
        for path, length, path_type in paths:
            if length > max_steps:
                # Record this long path
                long_paths.append((path, length, path_type))
                all_paths_valid = False
            
            # Check if path ends in a terminal state
            if path_type not in ["success", "failure"] and length >= max_steps:
                non_terminal_paths.append((path, length, path_type))
                all_paths_valid = False
        
        if long_paths:
            print(f"Found {len(long_paths)} paths exceeding {max_steps} steps from {start}")
            for path, length, path_type in sorted(long_paths, key=lambda x: x[1], reverse=True)[:3]:
                print(f"  Path of length {length} ({path_type}): {' -> '.join(path[:3])}...{' -> '.join(path[-3:])}")
        
        if non_terminal_paths:
            print(f"Found {len(non_terminal_paths)} paths that don't end in terminal states from {start}")
            for path, length, path_type in sorted(non_terminal_paths, key=lambda x: x[1], reverse=True)[:3]:
                print(f"  Non-terminal path ({path_type}): {' -> '.join(path[:3])}...{' -> '.join(path[-3:])}")
        
        if not long_paths and not non_terminal_paths:
            print(f"All paths from {start} complete within {max_steps} steps and end in terminal states")
    
    return all_paths_valid

def save_game_config(config, file_path):
    """Save the game configuration to a YAML file."""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

def analyze_all_paths(config, max_steps=20):
    """Analyze all paths from all starting states."""
    starting_states = config['states']
    
    path_stats = {
        'total_paths': 0,
        'success_paths': 0,
        'failure_paths': 0,
        'long_paths': 0,
        'max_path_length': 0,
        'paths_by_length': defaultdict(int),
        'long_path_examples': []
    }
    
    print(f"Analyzing paths from {len(starting_states)} starting states...")
    
    for start in config['starting_states']:
        print(f"Finding paths from {start}...")
        
        paths = find_all_paths_from_start(config, start, max_depth=max_steps+5)
        
        for path, length, path_type in paths:
            path_stats['total_paths'] += 1
            path_stats['max_path_length'] = max(path_stats['max_path_length'], length)
            path_stats['paths_by_length'][length] += 1
            
            if length > max_steps:
                path_stats['long_paths'] += 1
                # Store some examples of long paths
                if len(path_stats['long_path_examples']) < 5:
                    path_stats['long_path_examples'].append((path, length))
            
            if path_type == "success":
                path_stats['success_paths'] += 1
            elif path_type == "failure":
                path_stats['failure_paths'] += 1
    
    return path_stats

def find_longest_paths(config, max_paths=3, max_depth=30):
    """Find the longest paths from each starting state."""
    states = config['states']
    starting_states = config['starting_states']
    
    longest_paths = {}
    
    for start in starting_states:
        print(f"Finding longest paths from {start}...")
        
        # Get all paths from this starting state
        all_paths = find_all_paths_from_start(config, start, max_depth=max_depth)
        
        # Sort paths by length (descending)
        sorted_paths = sorted(all_paths, key=lambda x: x[1], reverse=True)
        
        # Store the longest paths
        longest_paths[start] = sorted_paths[:max_paths]
        
        # Print the longest paths
        print(f"Longest paths from {start}:")
        for path, length, path_type in sorted_paths[:max_paths]:
            print(f"  Path of length {length} ({path_type}):")
            # Print the full path
            for i, state in enumerate(path):
                if i > 0:
                    # Try to find the action that led to this state
                    prev_state = path[i-1]
                    if prev_state in states:
                        actions = states[prev_state].get('actions', {})
                        action_desc = "unknown action"
                        for action_name, action_data in actions.items():
                            if 'next_state' in action_data and action_data['next_state'] == state:
                                action_desc = f"{action_name}: {action_data.get('description', 'No description')}"
                                break
                            elif 'probabilities' in action_data and state in action_data['probabilities']:
                                prob = action_data['probabilities'][state]
                                action_desc = f"{action_name} ({prob:.2f}): {action_data.get('description', 'No description')}"
                                break
                        print(f"    {i}. {prev_state} --[{action_desc}]--> {state}")
                    else:
                        print(f"    {i}. {prev_state} --> {state}")
                else:
                    print(f"    {i}. {state} (starting state)")
            print()
    
    return longest_paths

def main():
    # Try different possible paths for the config file
    possible_paths = [
        '/mnt/core_llm/masoud/verifiers/verifiers/envs/imprisoned/imprisoned.yaml',
        'verifiers/verifiers/envs/imprisoned/imprisoned.yaml',
        'imprisoned.yaml'
    ]
    
    config_path = None
    for path in possible_paths:
        if os.path.exists(path):
            config_path = path
            break
    
    if not config_path:
        print("Error: Could not find the configuration file. Please specify the correct path.")
        return
    
    print(f"Using configuration file: {config_path}")
    
    # Load the original configuration
    try:
        original_config = load_game_config(config_path)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return
    
    # Analyze paths using the original method
    analysis = analyze_paths(original_config)
    
    print("Path Analysis:")
    print(f"Maximum path lengths from starting states: {analysis['max_path_lengths']}")
    print(f"Potential loop states: {analysis['potential_loops']}")
    print(f"Unreachable terminal states: {analysis['unreachable_terminals']}")
    
    # Analyze all paths with timeout
    print("\nPerforming detailed path analysis...")
    try:
        # Set a timeout for the analysis to prevent hanging
        def timeout_handler(signum, frame):
            raise TimeoutError("Path analysis timed out")
        
        # Set a 60-second timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)
        
        path_stats = analyze_all_paths(original_config)
        
        # Cancel the alarm
        signal.alarm(0)
        
        print("\nDetailed Path Analysis:")
        print(f"Total paths found: {path_stats['total_paths']}")
        print(f"Success paths: {path_stats['success_paths']}")
        print(f"Failure paths: {path_stats['failure_paths']}")
        print(f"Paths exceeding 20 steps: {path_stats['long_paths']}")
        print(f"Maximum path length: {path_stats['max_path_length']}")
        
        print("\nPath length distribution:")
        for length in sorted(path_stats['paths_by_length'].keys()):
            count = path_stats['paths_by_length'][length]
            print(f"  Length {length}: {count} paths")
        
        if path_stats['long_path_examples']:
            print("\nExamples of long paths:")
            for path, length in path_stats['long_path_examples']:
                print(f"  Path of length {length}: {' -> '.join(path[:3])}...{' -> '.join(path[-3:])}")
    
    except TimeoutError as e:
        print(f"Warning: {e}")
        print("Skipping detailed path analysis due to timeout")
    
    except Exception as e:
        print(f"Error during path analysis: {e}")
    
    # After the initial path analysis
    print("\nFinding and displaying the longest paths...")
    try:
        # Set a timeout for the analysis to prevent hanging
        def timeout_handler(signum, frame):
            raise TimeoutError("Longest path analysis timed out")
        
        # Set a 60-second timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(120)  # Increase timeout to 2 minutes
        
        longest_paths = find_longest_paths(original_config, max_depth=40)  # Increase max_depth
        
        # Cancel the alarm
        signal.alarm(0)
        
    except TimeoutError as e:
        print(f"Warning: {e}")
        print("Skipping longest path analysis due to timeout")
    
    except Exception as e:
        print(f"Error during longest path analysis: {e}")
    
    # Fix long paths recursively
    fixed_config = fix_long_paths_recursively(original_config)
    
    # Verify the fix worked
    post_analysis = analyze_paths(fixed_config)
    print("\nAfter fixes:")
    print(f"Maximum path lengths from starting states: {post_analysis['max_path_lengths']}")
    print(f"Potential loop states: {post_analysis['potential_loops']}")
    
    # Additional verification
    print("\nPerforming detailed path length check...")
    check_path_lengths(fixed_config)
    
    # Save the modified configuration
    output_path = os.path.join(os.path.dirname(config_path), 'imprisoned_fixed.yaml')
    save_game_config(fixed_config, output_path)
    print(f"\nFixed configuration saved to {output_path}")

if __name__ == "__main__":
    main() 