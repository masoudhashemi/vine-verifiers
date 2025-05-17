import os
import re
import random
import pickle
import pandas as pd
from typing import Any, Dict, List, Tuple, Union, Optional
from datasets import Dataset
from trl.trainer.grpo_trainer import RewardFunc

from verifiers.envs.simple_env import SimpleEnv
from verifiers.parsers import XMLParser
from verifiers.envs.knight_and_knaves.kk_env import KnightsKnavesEnv
from verifiers.envs.knight_and_knaves.kk_verification import (
    parse_statements, 
    extract_answer, 
    verify_kk_puzzle, 
    count_valid_statements
)

class KnightsKnavesSimpleEnv(SimpleEnv):
    """Simple environment for Knights and Knaves puzzle solving"""

    def __init__(self, 
                 size: Optional[Union[int, List[int]]] = None, 
                 complexity: Optional[Union[int, List[int]]] = None, 
                 dataset: str = "knights_knaves", 
                 **kwargs):
        self.dataset = dataset
        self.size = size
        self.complexity = complexity
        system_prompt = (
            "You are an expert logic puzzle solver. Solve Knights and Knaves puzzles by showing your reasoning and final answer. "
            "A Knight always tells the truth, and a Knave always lies.\n\n"
            "Format your solution like this:\n"
            "<thinking>\n"
            "1. First analyze what we know...\n"
            "2. Then consider the implications...\n"
            "3. Finally deduce who is who...\n"
            "</thinking>\n"
            "<answer>\n"
            "P1 is a Knight, P2 is a Knave, etc.\n"
            "</answer>"
        )
        
        # Initialize the base KnightsKnaves environment
        self.env = KnightsKnavesEnv(size=size if isinstance(size, int) else 3, 
                                    complexity=complexity if isinstance(complexity, int) else 2)
        self.current_problem = None
        self.current_solution = None
        self._cached_datasets = None
        self.parser = XMLParser(fields=["thinking", "answer"])

        super().__init__(system_prompt=system_prompt, **kwargs)

    def _dict_to_str(self, dict_data: Dict[str, str]) -> str:
        """Convert dictionary to string"""
        return ", ".join([f"{k} is a {v}" for k, v in dict_data.items()])

    def _str_to_dict(self, str_data: Union[str, List[str]]) -> Dict[str, str]:
        """Convert string or list of strings to dictionary"""
        if isinstance(str_data, list):
            str_data = ", ".join(str_data)
            
        result = {}
        for part in str_data.split(","):
            part = part.strip()
            if " is a " in part:
                person_type_parts = part.split(" is a ")
                person = person_type_parts[0].strip()
                type_ = person_type_parts[1].strip().capitalize()
                result[person] = type_
            elif " is " in part:
                person_type_parts = part.split(" is ")
                person = person_type_parts[0].strip()
                type_ = person_type_parts[1].strip().capitalize()
                result[person] = type_
                
        return result

    def generate_combined_dataset(self, train_size, eval_size, test_size):
        """
        Generate a combined dataset of unique puzzles and then split into train/val/test.
        This ensures diversity across all splits with balanced distribution of puzzle types.
        """        
        # Define size and complexity ranges
        if self.size is None:
            sizes = [3, 4, 5]
        elif isinstance(self.size, int):
            sizes = [self.size]
        else:
            sizes = self.size
        
        if self.complexity is None:
            complexities = [2, 3, 4]
        elif isinstance(self.complexity, int):
            complexities = [self.complexity]
        else:
            complexities = self.complexity
        
        # Calculate total samples needed
        total_samples = train_size + eval_size + test_size
        print(f"Generating {total_samples} total samples")
        
        # Generate all samples in a single loop
        all_samples = []
        samples_by_config = {}  # Dictionary to track samples by size/complexity
        
        # Initialize the dictionary
        for s in sizes:
            for c in complexities:
                samples_by_config[(s, c)] = []
        
        # Generate samples until we have enough for all configurations
        attempts = 0
        max_attempts = total_samples * 5
        
        while len(all_samples) < total_samples and attempts < max_attempts:
            # Randomly select size and complexity
            s = random.choice(sizes)
            c = random.choice(complexities)
            
            env = KnightsKnavesEnv(size=s, complexity=c)
            observation, info = env.reset()
            statements = parse_statements(observation)
            is_valid, _ = verify_kk_puzzle(
                statements, {k: v == "Knight" for k, v in info["correct_solution"].items()}
            )
            
            if is_valid:
                # Check similarity with existing samples
                is_diverse = True
                for existing_sample in all_samples:
                    similarity = self._calculate_puzzle_similarity(observation, existing_sample["prompt"])
                    if similarity > 0.7:
                        is_diverse = False
                        break
                
                if is_diverse:
                    answer = self._dict_to_str(info["correct_solution"])
                    sample = {
                        "prompt": observation, 
                        "answer": answer,
                        "size": s,
                        "complexity": c
                    }
                    all_samples.append(sample)
                    samples_by_config[(s, c)].append(sample)
                    
                    if len(all_samples) % 10 == 0:
                        print(f"Generated {len(all_samples)}/{total_samples} samples")
            
            if attempts % 100 == 0:
                print(f"attempts: {attempts} and all_samples: {len(all_samples)}")
                
            attempts += 1
        
        print(f"Generated {len(all_samples)} total samples after {attempts} attempts")
        
        # Print distribution of samples
        for (s, c), samples in samples_by_config.items():
            print(f"Size {s}, Complexity {c}: {len(samples)} samples")
        
        # Now split into train, val, test ensuring balanced distribution
        test_samples = []
        eval_samples = []
        train_samples = []
        
        # First, allocate samples to test set with balanced distribution
        samples_per_config_test = max(1, test_size // (len(sizes) * len(complexities)))
        for (s, c), samples in samples_by_config.items():
            # Take up to samples_per_config_test samples for test
            config_test_samples = samples[:samples_per_config_test]
            test_samples.extend(config_test_samples)
            # Remove these samples from the pool
            samples_by_config[(s, c)] = samples[samples_per_config_test:]
        
        # Next, allocate samples to eval set with balanced distribution
        samples_per_config_eval = max(1, eval_size // (len(sizes) * len(complexities)))
        for (s, c), samples in samples_by_config.items():
            # Take up to samples_per_config_eval samples for eval
            config_eval_samples = samples[:samples_per_config_eval]
            eval_samples.extend(config_eval_samples)
            # Remove these samples from the pool
            samples_by_config[(s, c)] = samples[samples_per_config_eval:]
        
        # Finally, allocate remaining samples to train set
        for (s, c), samples in samples_by_config.items():
            train_samples.extend(samples)
        
        # If we don't have enough training samples, take some from test and eval
        remaining_samples = []
        if len(train_samples) < train_size:
            # Take extra samples from test if available
            if len(test_samples) > test_size:
                extra_test = test_samples[test_size:]
                test_samples = test_samples[:test_size]
                remaining_samples.extend(extra_test)
            
            # Take extra samples from eval if available
            if len(eval_samples) > eval_size:
                extra_eval = eval_samples[eval_size:]
                eval_samples = eval_samples[:eval_size]
                remaining_samples.extend(extra_eval)
            
            # Add remaining samples to train
            train_samples.extend(remaining_samples)
        
        # Trim to requested sizes
        train_samples = train_samples[:train_size]
        eval_samples = eval_samples[:eval_size]
        test_samples = test_samples[:test_size]
        
        print(f"Final split: {len(train_samples)} train, {len(eval_samples)} validation, {len(test_samples)} test")
        
        # Convert to DataFrames and then to HuggingFace datasets
        train_df = pd.DataFrame(train_samples)
        eval_df = pd.DataFrame(eval_samples)
        test_df = pd.DataFrame(test_samples)
        
        # Convert to HuggingFace datasets
        train_dataset = Dataset.from_pandas(train_df)
        eval_dataset = Dataset.from_pandas(eval_df)
        test_dataset = Dataset.from_pandas(test_df)
        
        return {
            'train': train_dataset,
            'val': eval_dataset,
            'test': test_dataset
        }

    def get_dataset(self, split="train", **kwargs: Any) -> Dataset | None:
        """Generate diverse puzzles on the fly instead of using a fixed dataset"""
        # Get dataset sizes
        train_size = kwargs.get("train_size", 400 * 9)
        if isinstance(train_size, float) and train_size <= 1:
            train_size = int(train_size * 400 * 9)
        
        eval_size = kwargs.get("eval_size", 10 * 9)
        test_size = kwargs.get("test_size", 10 * 9)
        
        # Create a unique identifier for this dataset configuration
        # Handle None values for size and complexity
        size_str = "default" if self.size is None else str(self.size)
        complexity_str = "default" if self.complexity is None else str(self.complexity)
        config_id = f"s{size_str}_c{complexity_str}_tr{train_size}_ev{eval_size}_te{test_size}"
        
        cache_dir = kwargs.get("cache_dir", "kk_dataset_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"kk_dataset_{config_id}.pkl")
        
        # Check if we already have a cached dataset file
        if os.path.exists(cache_file):
            print(f"Loading cached dataset from {cache_file}")
            with open(cache_file, 'rb') as f:
                self._cached_datasets = pickle.load(f)
        # Check if we already have a cached dataset in memory
        elif not hasattr(self, '_cached_datasets') or self._cached_datasets is None:
            # Generate combined dataset
            print(f"Generating new dataset with config: {config_id}")
            self._cached_datasets = self.generate_combined_dataset(
                train_size, eval_size, test_size
            )
            # Save the dataset to cache
            print(f"Saving dataset to {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump(self._cached_datasets, f)
        
        # Get the appropriate dataset based on split
        if split == "train":
            dataset = self._cached_datasets['train']
        elif split == "val":
            dataset = self._cached_datasets['val']
        elif split == "test":
            dataset = self._cached_datasets['test']
        else:
            raise ValueError(f"Unknown split: {split}")
        
        print(f"Returning {len(dataset)} samples for {split} split")
        
        return dataset
    
    def get_eval_dataset(self, **kwargs: Any) -> Dataset | None:
        """Get evaluation dataset"""
        return self.get_dataset(split="val", **kwargs)

    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        """Return reward functions for evaluating model responses"""
        return [self.check_format, self.check_solution]

    def check_format(self, completion: str, **kwargs) -> float:
        """Check if the solution follows the required XML format"""
        if not completion:
            return 0.0

        try:
            parsed = self.parser.parse(completion)
            has_thinking = hasattr(parsed, 'thinking') and parsed.thinking is not None
            has_answer = hasattr(parsed, 'answer') and parsed.answer is not None

            # Check if answer section contains the required format
            if has_answer:
                answer_section = parsed.answer
                has_solution_format = any(x in answer_section for x in [" is a Knight", " is a Knave"])
            else:
                has_solution_format = False

            # Give partial credit for partial formatting
            score = 0.0
            if has_thinking:
                score += 0.3
            if has_answer:
                score += 0.3
            if has_solution_format:
                score += 0.4

            return score

        except Exception:
            return 0.0

    def check_solution(self, completion: str, reference: str, **kwargs) -> float:
        """Check if the solution is correct using logical verification"""
        try:
            answer_response = extract_answer(completion)
            statements = parse_statements(reference)
            is_valid, sat_model = verify_kk_puzzle(statements, answer_response)
            
            print(f"answer_response: {answer_response}")
            print(f"statements: {statements}")
            print(f"is_valid: {is_valid}")

            if is_valid:
                return 1.0

            # if at least half of the clauses are valid, give partial reward
            valid_count, total_count, invalid_statements = count_valid_statements(statements, answer_response)
            print(f"valid_count / total_count: {valid_count} / {total_count}")
            if total_count > 0:
                ratio = valid_count / total_count
                if ratio >= 0.5:
                    return 0.5 * ratio  # Max 0.5 for partial solutions
            return 0.0

        except Exception as e:
            print(f"Error in verifying solution: {e}")
            return 0.0

    def _calculate_puzzle_similarity(self, puzzle1: str, puzzle2: str) -> float:
        """
        Calculate similarity between two Knights and Knaves puzzles.
        Returns a value between 0 (completely different) and 1 (identical).
        """
        # Extract statements from both puzzles
        statements1 = parse_statements(puzzle1)
        statements2 = parse_statements(puzzle2)
        statements1 = [f"{k}: {v}" for k, v in statements1.items()]
        statements2 = [f"{k}: {v}" for k, v in statements2.items()]
        # replace numbers with "" 
        statements1 = [re.sub(r'\d+', '', s) for s in statements1]
        statements2 = [re.sub(r'\d+', '', s) for s in statements2]
        
        # If either puzzle has no statements, they can't be compared properly
        if not statements1 or not statements2:
            return 0.0
        
        if len(statements1) != len(statements2):
            return 0.0
        
        matches = 0
        for s1 in statements1:
            for s2 in statements2:
                if s1.strip() == s2.strip():
                    matches += 1
                    break

        if matches == len(statements1):
            return 1.0
        else:
            return 0.0 