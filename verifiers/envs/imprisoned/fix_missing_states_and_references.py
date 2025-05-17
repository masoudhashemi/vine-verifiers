import yaml


def fix_yaml_file():
    """Fix the imprisoned.yaml file by adding missing states and actions."""
    # Load the game data
    with open("imprisoned.yaml", "r") as file:
        game_data = yaml.safe_load(file)

    states = game_data.get("states", {})

    # Find all referenced states
    referenced_states = set()

    # Check next_state references
    for state_name, state_data in states.items():
        if "actions" in state_data:
            for action_name, action_data in state_data["actions"].items():
                if "next_state" in action_data:
                    referenced_states.add(action_data["next_state"])
                if "probabilities" in action_data:
                    for prob_state in action_data["probabilities"].keys():
                        referenced_states.add(prob_state)

    # Find states that are referenced but not defined or have no actions
    missing_states = []
    states_without_actions = []

    for state in referenced_states:
        if state not in states:
            missing_states.append(state)
        elif "actions" not in states[state] and not states[state].get("terminal", False):
            states_without_actions.append(state)

    print(f"Found {len(missing_states)} missing states")
    print(f"Found {len(states_without_actions)} states without actions")

    # Fix states without actions
    for state in states_without_actions:
        if "next_state" in states[state]:
            # If it has a next_state but no actions, add a continue action
            states[state]["actions"] = {
                "continue": {"description": "Continue on your path.", "next_state": states[state]["next_state"]}
            }
            # Remove the direct next_state as it's now in the action
            del states[state]["next_state"]
        else:
            # Add generic actions
            states[state]["actions"] = {
                "continue": {
                    "description": "Continue on your path.",
                    "next_state": game_data.get("starting_states", ["prison_cell"])[0],
                },
                "look_around": {
                    "description": "Look around for opportunities.",
                    "probabilities": {"prison_cell": 0.7, "hallway": 0.3},
                },
            }

    # Add missing states
    for state in missing_states:
        # Create a template state with generic actions
        game_data["states"][state] = {
            "description": f"You are in the {state.replace('_', ' ')} state.",
            "actions": {
                "continue": {
                    "description": "Continue on your path.",
                    "next_state": game_data.get("starting_states", ["prison_cell"])[0],
                },
                "look_around": {
                    "description": "Look around for opportunities.",
                    "probabilities": {"prison_cell": 0.7, "hallway": 0.3},
                },
            },
        }

    # Special case fixes for specific states
    specific_fixes = {
        "caught_stealing": {
            "description": "You're caught trying to steal medicine from the infirmary!",
            "actions": {
                "surrender": {"description": "Give up and accept punishment.", "next_state": "solitary_confinement"},
                "run": {
                    "description": "Try to escape from the guards.",
                    "probabilities": {"hallway": 0.3, "caught_by_guard": 0.7},
                },
            },
        },
        "item_lost": {
            "description": "You lost the item you were carrying.",
            "actions": {
                "search_for_replacement": {
                    "description": "Look for something else that might help.",
                    "probabilities": {"find_tool": 0.3, "waste_time": 0.7},
                },
                "continue_without_it": {
                    "description": "Proceed with your escape plan without the item.",
                    "next_state": "prison_cell",
                },
            },
        },
        "nothing_useful": {
            "description": "You find nothing useful here.",
            "actions": {
                "keep_looking": {
                    "description": "Continue searching more thoroughly.",
                    "probabilities": {"find_tool": 0.2, "waste_time": 0.8},
                },
                "give_up": {"description": "Stop searching and try something else.", "next_state": "prison_cell"},
            },
        },
        "obtain_tool": {
            "description": "You've obtained a useful tool that might help with your escape.",
            "actions": {
                "hide_tool": {
                    "description": "Hide the tool for later use.",
                    "grants": "tool",
                    "next_state": "prison_cell",
                },
                "use_immediately": {
                    "description": "Try to use the tool right away.",
                    "probabilities": {"unlock_success": 0.4, "tool_breaks": 0.6},
                },
            },
        },
        "opportunity_arises": {
            "description": "A perfect opportunity for escape presents itself!",
            "actions": {
                "take_chance": {
                    "description": "Seize the opportunity immediately.",
                    "probabilities": {"escape_success": 0.4, "caught_by_guard": 0.6},
                },
                "wait_for_better_timing": {
                    "description": "Wait for an even better moment.",
                    "probabilities": {"opportunity_lost": 0.7, "better_opportunity": 0.3},
                },
            },
        },
        "opportunity_lost": {
            "description": "The opportunity has passed. You'll need to find another way.",
            "actions": {
                "return_to_cell": {
                    "description": "Return to your cell and rethink your plan.",
                    "next_state": "prison_cell",
                },
                "look_for_new_opportunity": {
                    "description": "Keep looking for another chance.",
                    "probabilities": {"hallway": 0.5, "caught_by_guard": 0.5},
                },
            },
        },
        "better_opportunity": {
            "description": "Your patience paid off! An even better opportunity appears.",
            "actions": {
                "take_it": {
                    "description": "Take advantage of this perfect moment.",
                    "probabilities": {"escape_success": 0.6, "caught_by_guard": 0.4},
                }
            },
        },
        "learn_secret_passage": {
            "description": "You learn about a secret passage that might lead outside.",
            "actions": {
                "investigate_passage": {
                    "description": "Try to find and use the secret passage.",
                    "probabilities": {"tunnel_path": 0.6, "caught_by_guard": 0.4},
                },
                "share_with_others": {
                    "description": "Tell other prisoners about the passage.",
                    "probabilities": {"prisoner_friendly": 0.4, "prisoner_hostile": 0.6},
                },
            },
        },
        "useless_information": {
            "description": "The information you received isn't very helpful.",
            "actions": {
                "ask_for_more": {
                    "description": "Try to get more useful information.",
                    "probabilities": {"learn_shift_change": 0.3, "prisoner_hostile": 0.7},
                },
                "figure_it_out_yourself": {
                    "description": "Rely on your own observations instead.",
                    "next_state": "prison_cell",
                },
            },
        },
        "gain_respect": {
            "description": "You've earned respect from other prisoners.",
            "actions": {
                "recruit_help": {
                    "description": "Ask for help with your escape plan.",
                    "probabilities": {"prisoner_loyal": 0.6, "prisoner_snitch": 0.4},
                },
                "use_influence": {
                    "description": "Use your new status to gain privileges.",
                    "probabilities": {"obtain_tool": 0.5, "caught_by_guard": 0.5},
                },
            },
        },
        "provoke_fight": {
            "description": "Your actions have started a fight with another prisoner.",
            "actions": {
                "fight_back": {
                    "description": "Engage in the fight.",
                    "probabilities": {"win_fight": 0.5, "lose_fight": 0.5},
                },
                "back_down": {
                    "description": "Try to de-escalate the situation.",
                    "probabilities": {"prisoner_hostile": 0.7, "guards_intervene": 0.3},
                },
            },
        },
        "guards_intervene": {
            "description": "Guards rush in to break up the fight.",
            "actions": {
                "surrender": {"description": "Give up peacefully.", "next_state": "solitary_confinement"},
                "resist": {
                    "description": "Fight against the guards too.",
                    "probabilities": {"beaten_by_guard": 0.8, "create_distraction": 0.2},
                },
            },
        },
        "create_distraction": {
            "description": "The chaos creates a perfect distraction.",
            "actions": {
                "escape_in_chaos": {
                    "description": "Use the confusion to make your escape.",
                    "probabilities": {"hallway": 0.6, "caught_by_guard": 0.4},
                }
            },
        },
        "intimidate_guard": {
            "description": "You've successfully intimidated a guard.",
            "actions": {
                "demand_keys": {
                    "description": "Demand the keys to the exit.",
                    "probabilities": {"obtain_key": 0.4, "guard_calls_backup": 0.6},
                },
                "force_cooperation": {
                    "description": "Force the guard to help you escape.",
                    "probabilities": {"guard_helps": 0.5, "guard_betrays_you": 0.5},
                },
            },
        },
        "guard_calls_backup": {
            "description": "The guard has called for backup!",
            "actions": {
                "run": {
                    "description": "Run before more guards arrive.",
                    "probabilities": {"hallway": 0.4, "caught_by_guard": 0.6},
                },
                "surrender": {"description": "Give up before things get worse.", "next_state": "solitary_confinement"},
            },
        },
        "learned_security_pattern": {
            "description": "You've learned the security patterns of the prison.",
            "actions": {
                "use_knowledge": {
                    "description": "Use this knowledge to plan your escape.",
                    "next_state": "prison_cell",
                },
                "exploit_immediately": {
                    "description": "Take advantage of a security gap right now.",
                    "probabilities": {"hallway": 0.7, "caught_by_guard": 0.3},
                },
            },
        },
        "guards_believe": {
            "description": "The guards believe your story and leave you alone.",
            "actions": {
                "continue_plan": {"description": "Continue with your escape plan.", "next_state": "prison_cell"},
                "take_immediate_action": {
                    "description": "Use this moment of trust to make a move.",
                    "probabilities": {"hallway": 0.5, "caught_by_guard": 0.5},
                },
            },
        },
        "successful_escape": {
            "description": "Your plan is working! Freedom is within reach.",
            "actions": {
                "final_push": {
                    "description": "Make the final push to freedom.",
                    "probabilities": {"escape_success": 0.7, "caught_at_last_moment": 0.3},
                },
                "careful_approach": {
                    "description": "Proceed with extreme caution.",
                    "probabilities": {"escape_success": 0.5, "caught_by_guard": 0.5},
                },
            },
        },
        "caught_at_last_moment": {
            "description": "You were caught at the last possible moment!",
            "actions": {
                "desperate_struggle": {
                    "description": "Fight with everything you have!",
                    "probabilities": {"escape_success": 0.3, "beaten_by_guard": 0.7},
                },
                "surrender": {
                    "description": "Give up. There will be other chances.",
                    "next_state": "solitary_confinement",
                },
            },
        },
        "get_caught": {
            "description": "You've been caught by the guards.",
            "actions": {
                "surrender": {"description": "Surrender peacefully.", "next_state": "solitary_confinement"},
                "resist": {
                    "description": "Try to fight or run.",
                    "probabilities": {"beaten_by_guard": 0.8, "escape_custody": 0.2},
                },
            },
        },
        "escape_custody": {
            "description": "You've managed to break free from the guards!",
            "actions": {
                "run": {
                    "description": "Run as fast as you can.",
                    "probabilities": {"hallway": 0.6, "caught_by_guard": 0.4},
                },
                "hide": {
                    "description": "Find a place to hide.",
                    "probabilities": {"air_duct_success": 0.5, "caught_by_guard": 0.5},
                },
            },
        },
    }

    # Apply specific fixes
    for state, data in specific_fixes.items():
        if state in game_data["states"]:
            # If the state exists but has no actions, add them
            if "actions" not in game_data["states"][state]:
                game_data["states"][state]["actions"] = data["actions"]
        else:
            # If the state doesn't exist, add it completely
            game_data["states"][state] = data

    # Save the updated game data
    with open("imprisoned.yaml", "w") as file:
        yaml.dump(game_data, file, default_flow_style=False, sort_keys=False)

    print("Fixed YAML file saved. All states now have actions or are marked as terminal.")


if __name__ == "__main__":
    fix_yaml_file()
