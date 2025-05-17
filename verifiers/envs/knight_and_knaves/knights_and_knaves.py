import random

from z3 import And, Bool, Not, Or, Solver, sat, unknown, unsat


def generate_knights_knaves_propositional_logic(size=2, complexity=1):
    """Generates Knights/Knaves logic, puzzle, solution, and reasoning.
    Args:
        size (int): Number of people in the puzzle (2-6)
        complexity (int): Complexity level of the puzzle (1-5)
            1: Basic statements with AND/OR
            2: Medium complexity (negation, self-reference)
            3: Advanced complexity (conditionals, nested statements)
            4: Expert complexity (quantifiers, biconditionals, meta-statements)

    The puzzles are set on a fictional island where all inhabitants are either knights, who always tell the truth, or knaves, who always lie.
    The puzzles involve a visitor to the island who meets small groups of inhabitants.
    Usually the aim is for the visitor to deduce the inhabitants' type from their statements, but some puzzles of this type ask for other facts to be deduced.
    The puzzle may also be to determine a yes–no question which the visitor can ask in order to discover a particular piece of information.
    """
    # Maximum attempts to generate a solvable puzzle
    max_attempts = 10
    size = max(2, min(10, size))
    complexity = max(1, min(4, complexity))
    people = [f"P{i}" for i in range(1, size + 1)]

    for attempt in range(max_attempts):
        # Pre-assign some people as Knights or Knaves
        pre_assignments = {}
        num_knights = random.randint(size // 3, 2 * size // 3)
        knight_people = random.sample(people, num_knights)
        for p in people:
            pre_assignments[f"K_{p}"] = p in knight_people

        statements = []
        clauses = []
        reasoning_steps = []  # List of strings for reasoning steps

        # Add initial reasoning about pre-assignments
        reasoning_steps.append("Initial assignments:")
        for p in sorted(people):
            is_knight = pre_assignments[f"K_{p}"]
            reasoning_steps.append(f"  {p} is assigned as a {'Knight' if is_knight else 'Knave'}")

        # Define statement types based on complexity
        basic_with_logic_types = ["other", "other_and", "other_or"]  # Level 1: Basic statements with AND/OR
        medium_types = [
            "other",
            "other_and",
            "other_or",
            "self",
            "negation",
            "double_negation",
        ]  # Level 2: Medium complexity
        advanced_types = [
            "other",
            "other_and",
            "other_or",
            "self",
            "negation",
            "double_negation",
            "conditional",
            "nested",
        ]  # Level 3: Advanced complexity
        expert_types = [
            "other",
            "other_and",
            "other_or",
            "self",
            "negation",
            "double_negation",
            "conditional",
            "nested",
            "quantifier",
            "biconditional",
            "meta",
        ]  # Level 4: Expert complexity

        # Select available statement types based on complexity level
        if complexity == 1:
            available_types = basic_with_logic_types
        elif complexity == 2:
            available_types = medium_types
        elif complexity == 3:
            available_types = advanced_types
        elif complexity == 4:
            available_types = expert_types
        else:  # complexity > 4
            available_types = expert_types

        # Ensure each person makes at least one statement
        for speaker in people:
            is_knight = pre_assignments[f"K_{speaker}"]
            # Pick a random statement type based on complexity
            statement_type = random.choice(available_types)

            # Keep trying until we successfully generate a statement for this person
            while True:
                try:
                    # Basic statement types (from original implementation)
                    if statement_type == "other":
                        # Simple statement about another person
                        targets = [p for p in people if p != speaker]
                        target = random.choice(targets)
                        target_is_knight = pre_assignments[f"K_{target}"]

                        if is_knight:
                            # Knight tells truth
                            statements.append((speaker, f"{target} is a {'Knight' if target_is_knight else 'Knave'}"))
                            reasoning_steps.append(f"{speaker} (Knight) makes true statement about {target}")
                        else:
                            # Knave lies
                            statements.append((speaker, f"{target} is a {'Knave' if target_is_knight else 'Knight'}"))
                            reasoning_steps.append(f"{speaker} (Knave) makes false statement about {target}")
                        clauses.append(f"K_{speaker} IFF {'K_' if target_is_knight else 'NOT K_'}{target}")

                    elif statement_type == "other_and":
                        # AND statement about two others
                        targets = [p for p in people if p != speaker]
                        if len(targets) >= 2:
                            target, target2 = random.sample(targets, 2)
                            t1_knight = pre_assignments[f"K_{target}"]
                            t2_knight = pre_assignments[f"K_{target2}"]

                            if is_knight:
                                # Knight tells truth
                                s1 = f"{target} is a {'Knight' if t1_knight else 'Knave'}"
                                s2 = f"{target2} is a {'Knight' if t2_knight else 'Knave'}"
                                reasoning_steps.append(
                                    f"{speaker} (Knight) makes true AND statement about {target} and {target2}"
                                )
                            else:
                                # Knave lies
                                s1 = f"{target} is a {'Knave' if t1_knight else 'Knight'}"
                                s2 = f"{target2} is a {'Knave' if t2_knight else 'Knight'}"
                                reasoning_steps.append(
                                    f"{speaker} (Knave) makes false AND statement about {target} and {target2}"
                                )
                            statements.append((speaker, f"{s1} AND {s2}"))
                            clauses.append(
                                f"K_{speaker} IFF ({'K_' if t1_knight else 'NOT K_'}{target} AND {'K_' if t2_knight else 'NOT K_'}{target2})"
                            )
                        else:
                            # Fallback if not enough targets
                            statement_type = "other"
                            continue

                    elif statement_type == "other_or":
                        # OR statement about two others
                        targets = [p for p in people if p != speaker]
                        if len(targets) >= 2:
                            target, target2 = random.sample(targets, 2)
                            t1_knight = pre_assignments[f"K_{target}"]
                            t2_knight = pre_assignments[f"K_{target2}"]

                            if is_knight:
                                # Knight tells truth
                                s1 = f"{target} is a {'Knight' if t1_knight else 'Knave'}"
                                s2 = f"{target2} is a {'Knight' if t2_knight else 'Knave'}"
                                reasoning_steps.append(
                                    f"{speaker} (Knight) makes true OR statement about {target} and {target2}"
                                )
                            else:
                                # Knave lies
                                s1 = f"{target} is a {'Knave' if t1_knight else 'Knight'}"
                                s2 = f"{target2} is a {'Knave' if t2_knight else 'Knight'}"
                                reasoning_steps.append(
                                    f"{speaker} (Knave) makes false OR statement about {target} and {target2}"
                                )
                            statements.append((speaker, f"{s1} OR {s2}"))
                            clauses.append(
                                f"K_{speaker} IFF ({'K_' if t1_knight else 'NOT K_'}{target} OR {'K_' if t2_knight else 'NOT K_'}{target2})"
                            )
                        else:
                            # Fallback if not enough targets
                            statement_type = "other"
                            continue

                    # Medium complexity statement types
                    elif statement_type == "self":
                        # Self-referential statement
                        if is_knight:
                            # Knight tells truth about self
                            statements.append((speaker, f"I am a Knight"))
                            reasoning_steps.append(f"{speaker} (Knight) makes true statement about self")
                            clauses.append(f"K_{speaker} IFF K_{speaker}")  # Tautology for Knights
                        else:
                            # Knave lies about self
                            statements.append((speaker, f"I am a Knight"))  # Knave claims to be Knight
                            reasoning_steps.append(f"{speaker} (Knave) makes false statement about self")
                            clauses.append(f"K_{speaker} IFF K_{speaker}")  # Contradiction for Knaves

                    elif statement_type == "negation":
                        # Negation statement
                        targets = [p for p in people if p != speaker]
                        target = random.choice(targets)
                        target_is_knight = pre_assignments[f"K_{target}"]

                        if is_knight:
                            # Knight tells truth
                            statements.append(
                                (speaker, f"{target} is not a {'Knave' if target_is_knight else 'Knight'}")
                            )
                            reasoning_steps.append(f"{speaker} (Knight) makes true negation about {target}")
                        else:
                            # Knave lies
                            statements.append(
                                (speaker, f"{target} is not a {'Knight' if target_is_knight else 'Knave'}")
                            )
                            reasoning_steps.append(f"{speaker} (Knave) makes false negation about {target}")
                        clauses.append(f"K_{speaker} IFF {'K_' if target_is_knight else 'NOT K_'}{target}")

                    elif statement_type == "double_negation":
                        # Double negation for added complexity
                        targets = [p for p in people if p != speaker]
                        target = random.choice(targets)
                        target_is_knight = pre_assignments[f"K_{target}"]

                        if is_knight:
                            # Knight tells truth
                            statements.append(
                                (
                                    speaker,
                                    f"It is not the case that {target} is not a {'Knight' if target_is_knight else 'Knave'}",
                                )
                            )
                            reasoning_steps.append(f"{speaker} (Knight) makes true double negation about {target}")
                        else:
                            # Knave lies
                            statements.append(
                                (
                                    speaker,
                                    f"It is not the case that {target} is not a {'Knave' if target_is_knight else 'Knight'}",
                                )
                            )
                            reasoning_steps.append(f"{speaker} (Knave) makes false double negation about {target}")
                        clauses.append(f"K_{speaker} IFF {'K_' if target_is_knight else 'NOT K_'}{target}")

                    # Advanced complexity statement types
                    elif statement_type == "conditional":
                        # Conditional statement (if-then)
                        targets = [p for p in people if p != speaker]
                        if len(targets) >= 2:
                            target, target2 = random.sample(targets, 2)
                            t1_knight = pre_assignments[f"K_{target}"]
                            t2_knight = pre_assignments[f"K_{target2}"]

                            # Determine if conditional is true based on truth table:
                            # If P then Q is false only when P is true and Q is false
                            conditional_true = not (t1_knight and not t2_knight)

                            if is_knight:
                                # Knight tells truth
                                statements.append(
                                    (
                                        speaker,
                                        f"If {target} is a Knight, then {target2} is a {'Knight' if t2_knight else 'Knave'}",
                                    )
                                )
                                reasoning_steps.append(f"{speaker} (Knight) makes true conditional statement")
                            else:
                                # Knave lies (conditional must be false)
                                # To make conditional false, we need antecedent true and consequent false
                                statements.append(
                                    (
                                        speaker,
                                        f"If {target} is a {'Knight' if t1_knight else 'Knave'}, then {target2} is a {'Knave' if t2_knight else 'Knight'}",
                                    )
                                )
                                reasoning_steps.append(f"{speaker} (Knave) makes false conditional statement")

                            # The logical representation is complex but can be simplified to: K_speaker IFF (NOT K_target OR K_target2)
                            clauses.append(
                                f"K_{speaker} IFF (NOT {'K_' if t1_knight else 'NOT K_'}{target} OR {'K_' if t2_knight else 'NOT K_'}{target2})"
                            )
                        else:
                            # Fallback if not enough targets
                            statement_type = "other"
                            continue

                    elif statement_type == "nested":
                        # Nested statement (what someone else said)
                        targets = [p for p in people if p != speaker]
                        if len(targets) >= 2:
                            target, target2 = random.sample(targets, 2)
                            t1_knight = pre_assignments[f"K_{target}"]
                            t2_knight = pre_assignments[f"K_{target2}"]

                            # What would target say about target2?
                            if t1_knight:  # If target is Knight
                                target_would_say = f"{target2} is a {'Knight' if t2_knight else 'Knave'}"
                            else:  # If target is Knave
                                target_would_say = f"{target2} is a {'Knave' if t2_knight else 'Knight'}"

                            if is_knight:
                                # Knight tells truth about what target would say
                                statements.append((speaker, f"{target} would say '{target_would_say}'"))
                                reasoning_steps.append(
                                    f"{speaker} (Knight) makes true nested statement about what {target} would say"
                                )
                            else:
                                # Knave lies about what target would say
                                if t1_knight:  # If target is Knight
                                    false_statement = f"{target2} is a {'Knave' if t2_knight else 'Knight'}"
                                else:  # If target is Knave
                                    false_statement = f"{target2} is a {'Knight' if t2_knight else 'Knave'}"
                                statements.append((speaker, f"{target} would say '{false_statement}'"))
                                reasoning_steps.append(
                                    f"{speaker} (Knave) makes false nested statement about what {target} would say"
                                )

                            # This is complex to represent in propositional logic
                            clauses.append(
                                f"K_{speaker} IFF (K_{target} IFF {'K_' if t2_knight else 'NOT K_'}{target2})"
                            )
                        else:
                            # Fallback if not enough targets
                            statement_type = "other"
                            continue

                    elif statement_type == "quantifier":
                        # Quantifier statement (at least, exactly, etc.)
                        if is_knight:
                            # Knight tells truth
                            # Count actual knights
                            actual_knight_count = sum(1 for p in people if pre_assignments[f"K_{p}"])
                            if random.choice([True, False]):
                                # "At least" statement
                                threshold = random.randint(1, actual_knight_count)
                                statements.append((speaker, f"At least {threshold} people here are Knights"))
                            else:
                                # "Exactly" statement
                                statements.append((speaker, f"Exactly {actual_knight_count} people here are Knights"))
                            reasoning_steps.append(f"{speaker} (Knight) makes true quantifier statement")
                        else:
                            # Knave lies
                            # Count actual knights
                            actual_knight_count = sum(1 for p in people if pre_assignments[f"K_{p}"])
                            if random.choice([True, False]):
                                # False "at least" statement
                                threshold = random.randint(actual_knight_count + 1, size)
                                if threshold <= size:  # Ensure valid threshold
                                    statements.append((speaker, f"At least {threshold} people here are Knights"))
                                else:
                                    statements.append((speaker, f"Everyone here is a Knight"))
                            else:
                                # False "exactly" statement
                                false_count = random.choice([i for i in range(size + 1) if i != actual_knight_count])
                                statements.append((speaker, f"Exactly {false_count} people here are Knights"))
                            reasoning_steps.append(f"{speaker} (Knave) makes false quantifier statement")

                        # This is difficult to represent in simple propositional logic
                        clauses.append(f"K_{speaker} IFF (complex quantifier logic)")

                    elif statement_type == "biconditional":
                        # Bi-conditional statement (if and only if)
                        targets = [p for p in people if p != speaker]
                        if len(targets) >= 2:
                            target, target2 = random.sample(targets, 2)
                            t1_knight = pre_assignments[f"K_{target}"]
                            t2_knight = pre_assignments[f"K_{target2}"]

                            # Bi-conditional is true when both sides are same (both true or both false)
                            biconditional_true = t1_knight == t2_knight

                            if is_knight:
                                # Knight tells truth
                                statements.append(
                                    (
                                        speaker,
                                        f"{target} is a Knight if and only if {target2} is a {'Knight' if t2_knight else 'Knave'}",
                                    )
                                )
                                reasoning_steps.append(f"{speaker} (Knight) makes true bi-conditional statement")
                            else:
                                # Knave lies (bi-conditional must be false)
                                statements.append(
                                    (
                                        speaker,
                                        f"{target} is a Knight if and only if {target2} is a {'Knave' if t2_knight else 'Knight'}",
                                    )
                                )
                                reasoning_steps.append(f"{speaker} (Knave) makes false bi-conditional statement")

                            clauses.append(
                                f"K_{speaker} IFF ({'K_' if t1_knight else 'NOT K_'}{target} IFF {'K_' if t2_knight else 'NOT K_'}{target2})"
                            )
                        else:
                            # Fallback if not enough targets
                            statement_type = "other"
                            continue

                    elif statement_type == "meta":
                        # Meta-statement about statements
                        targets = [p for p in people if p != speaker]
                        target = random.choice(targets)
                        target_is_knight = pre_assignments[f"K_{target}"]

                        if is_knight:
                            # Knight tells truth
                            statements.append(
                                (speaker, f"{target} is telling the {'truth' if target_is_knight else 'lie'} right now")
                            )
                            reasoning_steps.append(f"{speaker} (Knight) makes true meta-statement about {target}")
                        else:
                            # Knave lies
                            statements.append(
                                (speaker, f"{target} is telling the {'lie' if target_is_knight else 'truth'} right now")
                            )
                            reasoning_steps.append(f"{speaker} (Knave) makes false meta-statement about {target}")
                        clauses.append(f"K_{speaker} IFF {'K_' if target_is_knight else 'NOT K_'}{target}")

                    # If we successfully generate a statement, break the loop
                    break
                except Exception as e:
                    print(f"Error in generating statement: {e}")
                    # If statement generation fails, try a different statement type
                    if complexity == 1:
                        statement_type = "other"  # For level 1, only use "other"
                    else:
                        statement_type = random.choice(available_types)
                    continue

        puzzle_text = "\n".join(f"{speaker} says: '{stmt}'" for speaker, stmt in statements)
        solution_text = ", ".join(f"{p} is a {'Knight' if pre_assignments[f'K_{p}'] else 'Knave'}" for p in people)

        # Add final reasoning step
        reasoning_steps.append("\nFinal solution:")
        for p in sorted(people):
            is_knight = pre_assignments[f"K_{p}"]
            reasoning_steps.append(f"  {p} is a {'Knight' if is_knight else 'Knave'}")

        # Verify that the puzzle is solvable and has a unique solution
        if verify_puzzle_solvability(statements, pre_assignments, people):
            return clauses, puzzle_text, solution_text, reasoning_steps, pre_assignments

    # If we couldn't generate a solvable puzzle after max attempts, return the last one generated
    return clauses, puzzle_text, solution_text, reasoning_steps, pre_assignments


def verify_puzzle_solvability(statements, pre_assignments, people):
    """
    Verify that the puzzle has a unique solution that can be deduced from the statements.

    Args:
        statements: List of (speaker, statement) tuples
        pre_assignments: Dictionary mapping person identifiers to boolean values (True for Knight, False for Knave)
        people: List of person identifiers

    Returns:
        Boolean indicating whether the puzzle is solvable with a unique solution
    """
    try:
        solver = Solver()

        # Create boolean variables for each person
        variables = {}
        for p in people:
            variables[p] = Bool(p)

        # Convert statements list to dictionary format for compatibility with parse_statement
        statements_dict = {speaker: stmt for speaker, stmt in statements}

        # Add constraints for each statement
        def parse_statement(statement):
            """Convert statement to Z3 boolean expression"""
            # Handle complex logical operators
            if "AND" in statement:
                parts = statement.split("AND")
                return And(parse_statement(parts[0].strip()), parse_statement(parts[1].strip()))
            elif "OR" in statement:
                parts = statement.split("OR")
                return Or(parse_statement(parts[0].strip()), parse_statement(parts[1].strip()))
            elif "If" in statement and "then" in statement:
                # Handle conditional statements
                parts = statement.split("If ")[1].split(", then ")
                antecedent = parts[0].strip()
                consequent = parts[1].strip()
                # If P then Q is equivalent to (not P) or Q
                return Or(Not(parse_statement(antecedent)), parse_statement(consequent))
            elif "if and only if" in statement:
                # Handle bi-conditional statements
                parts = statement.split(" is a Knight if and only if ")
                left_person = parts[0].strip()
                right_statement = parts[1].strip()
                # P iff Q is equivalent to (P and Q) or (not P and not Q)
                return variables[left_person] == parse_statement(right_statement)
            elif "would say" in statement:
                # Handle nested statements
                parts = statement.split(" would say '")
                speaker = parts[0].strip()
                nested_statement = parts[1].strip("'")
                # If speaker is Knight, they tell truth about nested statement
                # If speaker is Knave, they lie about nested statement
                return variables[speaker] == (parse_statement(nested_statement) == variables[speaker])
            elif "not the case that" in statement:
                # Handle double negation
                negated_part = statement.split("It is not the case that ")[1]
                return Not(parse_statement(negated_part))
            elif "not a" in statement:
                # Handle simple negation
                if "is not a Knight" in statement:
                    person = statement.split(" is")[0].strip()
                    return Not(variables[person])
                elif "is not a Knave" in statement:
                    person = statement.split(" is")[0].strip()
                    return variables[person]
            elif "At least" in statement and "people here are Knights" in statement:
                # Handle quantifier statements
                count = int(statement.split("At least ")[1].split(" people")[0])
                # Create a sum of all knights
                knight_count = sum([variables[p] for p in variables])
                return knight_count >= count
            elif "Exactly" in statement and "people here are Knights" in statement:
                # Handle exact quantifier statements
                count = int(statement.split("Exactly ")[1].split(" people")[0])
                # Create a sum of all knights
                knight_count = sum([variables[p] for p in variables])
                return knight_count == count
            elif "Everyone here is a Knight" in statement:
                # Handle universal quantifier
                return And([variables[p] for p in variables])
            elif "telling the truth" in statement:
                # Handle meta-statements
                person = statement.split(" is")[0].strip()
                return variables[person]
            elif "telling the lie" in statement:
                # Handle meta-statements
                person = statement.split(" is")[0].strip()
                return Not(variables[person])
            elif "I am a Knight" in statement:
                # Handle self-reference - this is a special case
                # This statement is always consistent with both Knight and Knave assignments
                # so we don't add any constraint
                return True

            # Now handle atomic statements
            if "is a Knight" in statement:
                person = statement.split(" is")[0].strip()
                return variables[person]
            elif "is a Knave" in statement:
                person = statement.split(" is")[0].strip()
                return Not(variables[person])

            # If we can't parse the statement, return None
            return None

        # For each person, their statement must be equivalent to them being a Knight
        for person, statement in statements_dict.items():
            # Special handling for self-referential statements
            if statement == "I am a Knight":
                # This creates a circular reference that doesn't constrain anything
                continue

            # For all other statements:
            # If person is Knight (variables[person] is True), statement must be True
            # If person is Knave (variables[person] is False), statement must be False
            parsed = parse_statement(statement)
            if parsed is not None:
                solver.add(variables[person] == parsed)

        # Check if the puzzle has a solution
        if solver.check() == unsat:
            return False  # No solution

        # Get the solution
        model = solver.model()
        solution = {p: bool(model.evaluate(variables[p])) for p in people}

        # Check if the solution matches the pre-assignments
        for p in people:
            if solution.get(p) != pre_assignments.get(f"K_{p}"):
                return False  # Solution doesn't match pre-assignments

        # Check for uniqueness by adding a constraint that excludes the current solution
        exclusion_constraint = Or([variables[p] != solution[p] for p in people])
        solver.add(exclusion_constraint)

        # If there's no other solution, the puzzle has a unique solution
        return solver.check() == unsat

    except Exception as e:
        # print(f"Error in verify_puzzle_solvability: {e}")
        return False  # If there's an error, assume the puzzle is not solvable


def generate_prompts_and_responses(num_examples=5, size=2, complexity=1):
    """Generates prompts, logic, puzzles, solutions, and reasoning."""
    prompts_responses = []
    for _ in range(num_examples):
        clauses, puzzle, solution, reasoning_steps, assignment = generate_knights_knaves_propositional_logic(
            size, complexity
        )
        prompt = (
            "Determine whether each person is a Knight (always tells the truth) or a Knave (always lies).  "
            "Show your reasoning steps using the provided propositional logic encoding.\n\n"
            f"Puzzle:\n{puzzle}\n"
            "Propositional Logic (using 'K_p' for person p being a Knight):\n" + "\n".join(clauses) + "\n\n"
            "Reasoning Steps:"
        )
        response = f"Solution:\n{solution}\nReasoning:\n" + "\n".join(reasoning_steps)
        prompts_responses.append(
            {
                "type": "Knights and Knaves",
                "prompt": prompt,
                "response": response,
                "logic": clauses,
                "puzzle": puzzle,
                "solution": solution,
                "reasoning": reasoning_steps,
                "assignment": assignment,
            }
        )
    return prompts_responses
