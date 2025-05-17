import re
from typing import Any, Dict, List, Tuple


def parse_statements(text):
    """
    Convert text format of statements to dictionary format

    Args:
        text: String containing statements in format:
              P1 says: '...'
              P2 says: '...'
              P3 says: '...'

    Returns:
        Dictionary mapping person to their statement
    """
    statements = {}

    # Split text into lines and process each line
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    for line in lines:
        # Check if line matches pattern "PX says: '...'"
        if "says:" not in line:
            continue

        # Split into person and statement
        person, statement = line.split("says:", 1)

        # Clean up person (remove whitespace)
        person = person.strip()

        # Clean up statement (remove quotes and whitespace)
        statement = statement.strip().strip("'\"")

        statements[person] = statement

    return statements


def extract_answer(completion):
    """
    Extract the answer from the completion text.
    Returns a dictionary mapping person identifiers to boolean values (True for Knight, False for Knave).
    """
    # First try to extract from XML format
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, completion, re.DOTALL)
    
    if match:
        answer_text = match.group(1).strip()
    else:
        # If no XML tags, try to find lines that look like answers
        lines = completion.split('\n')
        answer_lines = []
        for line in lines:
            if re.search(r'P\d+\s+is\s+a\s+(Knight|Knave)', line, re.IGNORECASE):
                answer_lines.append(line)
        answer_text = ' '.join(answer_lines)
    
    # Extract person-type pairs
    result = {}
    # Look for patterns like "P1 is a Knight" or "P2 is a Knave"
    pairs = re.findall(r'(P\d+)\s+is\s+a\s+(Knight|Knave)', answer_text, re.IGNORECASE)
    
    for person, type_str in pairs:
        result[person] = (type_str.lower() == 'knight')
    
    return result


def verify_kk_puzzle(statements, answer_dict):
    """
    Verify if the answer is consistent with the statements in the puzzle.
    
    Args:
        statements: Dictionary mapping person identifiers to their statements
        answer_dict: Dictionary mapping person identifiers to boolean values 
                    (True for Knight, False for Knave)
    
    Returns:
        Tuple of (is_valid, sat_model)
    """
    try:
        # Filter answer_dict to only include people mentioned in the statements
        valid_people = set(statements.keys())
        filtered_answer = {p: v for p, v in answer_dict.items() if p in valid_people}
        
        # If we filtered out entries, log a warning
        if len(filtered_answer) < len(answer_dict):
            removed = set(answer_dict.keys()) - valid_people
            print(f"Warning: Filtered out unknown people from answer: {removed}")
        
        # If we don't have answers for all people in the puzzle, it's invalid
        if set(filtered_answer.keys()) != valid_people:
            missing = valid_people - set(filtered_answer.keys())
            print(f"Warning: Missing answers for people: {missing}")
            return False, None
            
        from z3 import And, Bool, Not, Or, Solver, sat

        solver = Solver()

        # Create boolean variables for each person
        # True means Knight (always tells truth), False means Knave (always lies)
        variables = {}
        for person in filtered_answer.keys():
            variables[person] = Bool(person)

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
                # If person is Knight, statement is true
                # If person is Knave, statement is false
                # This creates a circular reference that needs special handling
                return True  # Placeholder, handled separately

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
        for person, statement in statements.items():
            # Special handling for self-referential statements
            if statement == "I am a Knight":
                # This creates a circular reference:
                # If Knight: statement is true, which means they are a Knight (consistent)
                # If Knave: statement is false, which means they are not a Knight (consistent)
                # So this statement doesn't actually constrain anything
                continue

            # For all other statements:
            # If person is Knight (variables[person] is True), statement must be True
            # If person is Knave (variables[person] is False), statement must be False
            parsed = parse_statement(statement)
            if parsed is not None:
                solver.add(variables[person] == parsed)

        # Add constraints from the answer
        for person, is_knight in filtered_answer.items():
            solver.add(variables[person] == is_knight)

        # Check if solution exists
        if solver.check() == sat:
            model = solver.model()
            return True, {str(d): bool(model[d]) for d in model.decls()}

        return False, None
    except Exception as e:
        print(f"Error in verify_kk_puzzle: {e}")
        return False, None


def count_valid_statements(statements, answer_dict):
    """
    Count how many statements are valid given the answer.
    
    Args:
        statements: Dictionary mapping person identifiers to their statements
        answer_dict: Dictionary mapping person identifiers to boolean values
    
    Returns:
        Tuple of (valid_count, total_count, invalid_statements)
    """
    try:
        # Filter answer_dict to only include people mentioned in the statements
        valid_people = set(statements.keys())
        filtered_answer = {p: v for p, v in answer_dict.items() if p in valid_people}
        
        # If we don't have answers for all people in the puzzle, we can only check what we have
        if set(filtered_answer.keys()) != valid_people:
            missing = valid_people - set(filtered_answer.keys())
            print(f"Warning in count_valid_statements: Missing answers for people: {missing}")
        
        from z3 import And, Bool, Not, Or, Solver, is_false, is_true, sat

        valid_count = 0
        invalid_statements = []

        # Create boolean variables for each person
        variables = {}
        for person in filtered_answer.keys():
            variables[person] = Bool(person)

        def parse_statement(statement):
            """Convert statement to Z3 boolean expression - same as in verify_kk_puzzle"""
            # Handle complex logical operators
            if "AND" in statement:
                parts = statement.split("AND")
                return And(parse_statement(parts[0].strip()), parse_statement(parts[1].strip()))
            elif "OR" in statement:
                parts = statement.split("OR")
                return Or(parse_statement(parts[0].strip()), parse_statement(parts[1].strip()))
            elif "If" in statement and "then" in statement:
                parts = statement.split("If ")[1].split(", then ")
                antecedent = parts[0].strip()
                consequent = parts[1].strip()
                return Or(Not(parse_statement(antecedent)), parse_statement(consequent))
            elif "if and only if" in statement:
                parts = statement.split(" is a Knight if and only if ")
                left_person = parts[0].strip()
                right_statement = parts[1].strip()
                return variables[left_person] == parse_statement(right_statement)
            elif "would say" in statement:
                parts = statement.split(" would say '")
                speaker = parts[0].strip()
                nested_statement = parts[1].strip("'")
                return variables[speaker] == (parse_statement(nested_statement) == variables[speaker])
            elif "not the case that" in statement:
                negated_part = statement.split("It is not the case that ")[1]
                return Not(parse_statement(negated_part))
            elif "not a" in statement:
                if "is not a Knight" in statement:
                    person = statement.split(" is")[0].strip()
                    return Not(variables[person])
                elif "is not a Knave" in statement:
                    person = statement.split(" is")[0].strip()
                    return variables[person]
            elif "At least" in statement and "people here are Knights" in statement:
                count = int(statement.split("At least ")[1].split(" people")[0])
                knight_count = sum([variables[p] for p in variables])
                return knight_count >= count
            elif "Exactly" in statement and "people here are Knights" in statement:
                count = int(statement.split("Exactly ")[1].split(" people")[0])
                knight_count = sum([variables[p] for p in variables])
                return knight_count == count
            elif "Everyone here is a Knight" in statement:
                return And([variables[p] for p in variables])
            elif "telling the truth" in statement:
                person = statement.split(" is")[0].strip()
                return variables[person]
            elif "telling the lie" in statement:
                person = statement.split(" is")[0].strip()
                return Not(variables[person])
            elif "I am a Knight" in statement:
                return True  # Special case, handled separately

            # Atomic statements
            if "is a Knight" in statement:
                person = statement.split(" is")[0].strip()
                return variables[person]
            elif "is a Knave" in statement:
                person = statement.split(" is")[0].strip()
                return Not(variables[person])

            return None

        # Check each statement individually
        for person, statement in statements.items():
            # Special handling for self-referential statements
            if statement == "I am a Knight":
                # This is always consistent, so count as valid
                valid_count += 1
                continue

            # Create a new solver for each statement
            solver = Solver()

            # Add the model constraints
            for p, is_knight in filtered_answer.items():
                if is_knight:
                    solver.add(variables[p])
                else:
                    solver.add(Not(variables[p]))

            # Parse the statement
            parsed_statement = parse_statement(statement)
            if parsed_statement is None:
                # If we can't parse, we can't validate
                continue

            # Check if statement is consistent with the person's type
            # Knights tell truth, Knaves lie
            is_knight = filtered_answer.get(person, False)

            if is_knight:
                # Knight's statement must be true
                solver.add(parsed_statement)
            else:
                # Knave's statement must be false
                solver.add(Not(parsed_statement))

            # Check if this is satisfiable
            if solver.check() == sat:
                valid_count += 1
            else:
                invalid_statements.append(f"{person}: '{statement}'")

        return valid_count, len(statements), invalid_statements

    except Exception as e:
        print(f"Error in count_valid_statements: {e}")
        return 0, len(statements), list(statements.keys())
