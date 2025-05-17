import gymnasium as gym
from gymnasium import spaces

from verifiers.envs.knight_and_knaves.kk_verification import extract_answer, parse_statements, verify_kk_puzzle
from verifiers.envs.knight_and_knaves.knights_and_knaves import generate_knights_knaves_propositional_logic


def verify_knights_knaves_solution(statements, solution):
    """Verify that the solution is consistent with all statements."""
    inconsistencies = []

    def evaluate_statement(statement):
        """Evaluate if a statement matches reality given the solution."""
        # Handle complex logical operators
        if "AND" in statement:
            parts = statement.split(" AND ")
            return evaluate_statement(parts[0].strip()) and evaluate_statement(parts[1].strip())
        elif "OR" in statement:
            parts = statement.split(" OR ")
            return evaluate_statement(parts[0].strip()) or evaluate_statement(parts[1].strip())
        elif "If" in statement and "then" in statement:
            # Handle conditional statements
            parts = statement.split("If ")[1].split(", then ")
            antecedent = parts[0].strip()
            consequent = parts[1].strip()
            antecedent_value = evaluate_statement(antecedent)
            consequent_value = evaluate_statement(consequent)
            # If P then Q is false only when P is true and Q is false
            return not (antecedent_value and not consequent_value)
        elif "if and only if" in statement:
            # Handle bi-conditional statements
            parts = statement.split(" is a Knight if and only if ")
            left_person = parts[0].strip()
            right_statement = parts[1].strip()
            left_value = solution.get(left_person) == "Knight"
            right_value = evaluate_statement(right_statement)
            return left_value == right_value
        elif "would say" in statement:
            # Handle nested statements
            parts = statement.split(" would say '")
            speaker = parts[0].strip()
            nested_statement = parts[1].strip("'")
            speaker_is_knight = solution.get(speaker) == "Knight"
            nested_value = evaluate_statement(nested_statement)
            # If speaker is Knight, they tell truth about nested statement
            # If speaker is Knave, they lie about nested statement
            return nested_value if speaker_is_knight else not nested_value
        elif "not the case that" in statement:
            # Handle double negation
            negated_part = statement.split("It is not the case that ")[1]
            return not evaluate_statement(negated_part)
        elif "not a" in statement:
            # Handle simple negation
            if "is not a Knight" in statement:
                person = statement.split(" is")[0].strip()
                return solution.get(person) != "Knight"
            elif "is not a Knave" in statement:
                person = statement.split(" is")[0].strip()
                return solution.get(person) != "Knave"
        elif "At least" in statement and "people here are Knights" in statement:
            # Handle quantifier statements
            count = int(statement.split("At least ")[1].split(" people")[0])
            actual_count = sum(1 for p, type_ in solution.items() if type_ == "Knight")
            return actual_count >= count
        elif "Exactly" in statement and "people here are Knights" in statement:
            # Handle exact quantifier statements
            count = int(statement.split("Exactly ")[1].split(" people")[0])
            actual_count = sum(1 for p, type_ in solution.items() if type_ == "Knight")
            return actual_count == count
        elif "Everyone here is a Knight" in statement:
            # Handle universal quantifier
            return all(type_ == "Knight" for p, type_ in solution.items())
        elif "telling the truth" in statement:
            # Handle meta-statements
            person = statement.split(" is")[0].strip()
            return solution.get(person) == "Knight"
        elif "telling the lie" in statement:
            # Handle meta-statements
            person = statement.split(" is")[0].strip()
            return solution.get(person) == "Knave"
        elif "I am a Knight" in statement:
            # Handle self-reference
            # This is handled specially in the main verification loop
            return True  # Placeholder, actual logic is in main loop

        # Now handle atomic statements
        if "is a Knight" in statement:
            person = statement.split(" is")[0].strip()
            return solution.get(person) == "Knight"
        elif "is a Knave" in statement:
            person = statement.split(" is")[0].strip()
            return solution.get(person) == "Knave"

        # If we can't parse the statement, return None
        return None

    for speaker, statement in statements:
        is_knight = solution.get(speaker) == "Knight"

        # Special handling for self-referential statements
        if statement == "I am a Knight":
            # For Knights: this is true
            # For Knaves: this is false
            if is_knight and not is_knight:
                inconsistencies.append(f"{speaker} is a Knight but said 'I am a Knight' which doesn't match reality")
            elif not is_knight and is_knight:
                inconsistencies.append(f"{speaker} is a Knave but said 'I am a Knight' which matches reality")
            continue

        # For all other statements
        statement_value = evaluate_statement(statement)

        if statement_value is None:
            inconsistencies.append(f"Could not evaluate statement: '{statement}'")
            continue

        # For Knights: what they say must match reality
        # For Knaves: what they say must be opposite to reality
        if is_knight and not statement_value:
            inconsistencies.append(f"{speaker} is a Knight but said '{statement}' which doesn't match reality")
        elif not is_knight and statement_value:
            inconsistencies.append(f"{speaker} is a Knave but said '{statement}' which matches reality")

    return len(inconsistencies) == 0, inconsistencies


class KnightsKnavesEnv(gym.Env):
    """Knights and Knaves Puzzle Environment."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, size=2, complexity=1):
        super().__init__()
        self.size = size
        self.complexity = complexity
        self.observation_space = spaces.Text(
            max_length=500, charset="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,' :?!\n"
        )  # Adjusted max_length and charset
        self.action_space = spaces.Text(
            max_length=500, charset="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ,"
        )  # Adjusted max_length and charset
        self._puzzle = None
        self._statements = None
        self._correct_solution_dict = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        is_valid = False
        iters = 0
        while not is_valid and iters < 20:
            (
                clauses,
                puzzle_text,
                solution_text,
                reasoning_steps,
                pre_assignments,
            ) = generate_knights_knaves_propositional_logic(size=self.size, complexity=self.complexity)
            statements = parse_statements(puzzle_text)
            is_valid, _ = verify_kk_puzzle(statements, {p[2:]: is_knight for p, is_knight in pre_assignments.items()})
            iters += 1

        # Parse the generated puzzle into statements format
        generated_statements = []
        for line in puzzle_text.split("\n"):
            if line.strip():
                speaker = line.split(" says: ")[0]
                statement = line.split("'")[1]
                generated_statements.append((speaker, statement))

        correct_solution_dict = {p[2:]: "Knight" if is_knight else "Knave" for p, is_knight in pre_assignments.items()}

        self._puzzle = puzzle_text
        self._statements = generated_statements
        self._correct_solution_dict = correct_solution_dict

        observation = self._puzzle
        info = {"correct_solution": self._correct_solution_dict}
        return observation, info

    def step(self, action_text):
        """Step the environment with an action."""
        action_text = action_text.strip()
        try:
            proposed_solution = {}
            for part in action_text.split(","):
                part = part.strip()
                if " is a " in part:
                    person_type_parts = part.split(" is a ")
                    person = person_type_parts[0].strip()
                    type_ = person_type_parts[1].strip().capitalize()  # Ensure Knight/Knave capitalization
                    if type_ not in ["Knight", "Knave"]:
                        raise ValueError("Invalid type in solution")
                    proposed_solution[person] = type_
                elif " is " in part:  # Handle potential "is" instead of "is a"
                    person_type_parts = part.split(" is ")
                    person = person_type_parts[0].strip()
                    type_ = person_type_parts[1].strip().capitalize()
                    if type_ not in ["Knight", "Knave"]:
                        raise ValueError("Invalid type in solution")
                    proposed_solution[person] = type_

            is_valid_solution, inconsistencies = verify_knights_knaves_solution(self._statements, proposed_solution)

            if is_valid_solution and proposed_solution == self._correct_solution_dict:
                reward = 1.0
            else:
                reward = -1.0

        except Exception as e:  # Catch parsing errors in action
            reward = -1.0  # Penalize invalid action format
            is_valid_solution = False  # Treat as invalid solution

        terminated = True  # Single step environment
        truncated = False
        observation = self._puzzle  # Return the puzzle as observation again for single step env
        info = {
            "is_valid_solution": is_valid_solution,
            "correct_solution": self._correct_solution_dict,
            "proposed_solution": proposed_solution,
        }

        return observation, reward, terminated, truncated, info

    def render(self):
        """Render the environment (text-based, so just prints the puzzle)."""
        print("\nKnights and Knaves Puzzle:")
        print(self._puzzle)
        print("\nProvide your solution in text format, e.g., 'P1 is a Knight, P2 is a Knave'")

    def close(self):
        """Close the environment."""
        pass
