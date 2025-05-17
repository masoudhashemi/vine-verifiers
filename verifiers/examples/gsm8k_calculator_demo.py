import argparse
import math
import os
import re
import sys
from typing import List, Dict, Tuple, Optional

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import verifiers as vf
from verifiers.utils.imports import LLM, SamplingParams  # type: ignore
from verifiers.tools import calculator
from verifiers.prompts import CALCULATOR_FEW_SHOT, DEFAULT_TOOL_PROMPT_TEMPLATE

def extract_number(text: str) -> Optional[float]:
    """
    Extract the first number from text, handling various formats.
    
    Args:
        text: Text to extract number from
        
    Returns:
        First number found as float, or None if no number found
    """
    if not text:
        return None
        
    # Remove commas in numbers and find all numbers in the text
    text = text.replace(',', '')
    numbers = re.findall(r'-?\d+\.?\d*', text)
    
    if not numbers:
        return None
        
    # Return the first number found as a float
    return float(numbers[0])

def compare_answers(model_answer: str, reference_answer: str) -> Tuple[bool, str]:
    """
    Compare model's answer with reference answer by extracting and comparing numbers.
    
    Args:
        model_answer: Model's generated answer
        reference_answer: Reference answer from dataset
        
    Returns:
        Tuple of (is_correct, explanation)
    """
    model_number = extract_number(model_answer)
    reference_number = extract_number(reference_answer)
    
    if model_number is None:
        return False, "No number found in model's answer"
    
    if reference_number is None:
        return False, "No number found in reference answer"
    
    # Compare with a small tolerance (0.001 or 0.1% relative difference)
    abs_diff = abs(model_number - reference_number)
    rel_diff = abs_diff / max(abs(reference_number), 1e-10)
    
    if abs_diff < 1e-10 or rel_diff < 0.001:
        return True, f"Correct! Model: {model_number}, Reference: {reference_number}"
    else:
        return False, f"Incorrect. Model: {model_number}, Reference: {reference_number}, Difference: {abs_diff}"

def run_demo(model_name: str, example_idx: int = 0, max_steps: int = 5, 
             vllm_device="cuda", gpu_memory_utilization=0.9, 
             vllm_dtype="auto", enable_prefix_caching=True, max_model_len=None):
    """
    Run an interactive demo of the GSM8K calculator tool environment.
    
    Args:
        model_name: Name of the model to use
        example_idx: Index of the example to use from the GSM8K test set
        max_steps: Maximum number of steps to run
        vllm_device: Device to run VLLM on ("cuda", "cpu")
        gpu_memory_utilization: Fraction of GPU memory to use
        vllm_dtype: Data type for model weights
        enable_prefix_caching: Whether to enable prefix caching
        max_model_len: Maximum model sequence length
    """
    print(f"Loading model: {model_name}")
    # Initialize VLLM model with detailed configuration
    model = LLM(
        model=model_name,
        device=vllm_device,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=vllm_dtype,
        enable_prefix_caching=enable_prefix_caching,
        max_model_len=max_model_len,
    )
    
    # Initialize tool environment for GSM8K
    print("Initializing environment...")
    vf_env = vf.ToolEnv(
        dataset="gsm8k",
        few_shot=CALCULATOR_FEW_SHOT[0],
        tools=[calculator],
        sleep_time=0.0,
        max_steps=max_steps
    )
    
    # Get the evaluation dataset
    eval_dataset = vf_env.get_eval_dataset(n=100)
    
    current_example_idx = example_idx
    continue_demo = True
    
    while continue_demo:
        # Get the example
        example = eval_dataset[current_example_idx]
        print("\n" + "="*80)
        print(f"EXAMPLE {current_example_idx}:")
        print(f"QUESTION: {example['question']}")
        print(f"EXPECTED ANSWER: {example['answer']}")
        print("="*80 + "\n")
        
        # Set up the initial messages
        messages = [
            {"role": "system", "content": vf_env.system_prompt}
        ]
        
        # Add few-shot examples if available
        if vf_env.few_shot:
            for msg in vf_env.few_shot:
                messages.append(msg)
        
        # Add the user query
        messages.append({"role": "user", "content": example['question']})
        
        # Print the initial state
        print_messages(messages, skip_system=True)
        
        # Run the conversation loop
        step = 0
        completed = False
        
        while not completed and step < max_steps:
            step += 1
            print(f"\n--- STEP {step} ---")
            
            # Get model response
            print("Model thinking...")
            sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.7,
                max_tokens=600,
                stop=["</tool>", "</answer>"],
                include_stop_str_in_output=True
            )
            
            response = model.chat(
                messages=messages,
                sampling_params=sampling_params
            )

            # Add model response to messages
            assistant_msg = {"role": "assistant", "content": response[0].outputs[0].text}
            messages.append(assistant_msg)
            
            # Print the model's response
            print("\nMODEL RESPONSE:")
            print(assistant_msg["content"])
            
            # Check if we're done
            if vf_env.is_completed(messages):
                completed = True
                print("\nConversation completed!")
            else:
                # Get environment response
                env_msg = vf_env.env_response(messages)
                messages.append(env_msg)
                
                # Print the environment response
                print("\nENVIRONMENT RESPONSE:")
                print(env_msg["content"])
        
        # Print final result
        print("\n" + "="*80)
        print("FINAL CONVERSATION:")
        print_messages(messages, skip_system=True, skip_few_shot=True)
        print("="*80)
        
        # Evaluate the answer
        from verifiers.parsers import XMLParser
        parser = XMLParser(fields=["reasoning", ("tool", "answer")])
        
        try:
            # Find the last assistant message
            assistant_msgs = [msg for msg in messages if msg['role'] == 'assistant']
            if assistant_msgs:
                last_msg = assistant_msgs[-1]
                parsed = parser.parse(last_msg['content'])
                
                if hasattr(parsed, 'answer') and parsed.answer is not None:
                    print(f"\nMODEL'S FINAL ANSWER: {parsed.answer}")
                    print(f"EXPECTED ANSWER: {example['answer']}")
                    
                    # Evaluate accuracy using number matching
                    is_correct, explanation = compare_answers(parsed.answer, example['answer'])
                    print("\nEVALUATION:")
                    print(f"Result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
                    print(f"Details: {explanation}")
                else:
                    print("\nNo final answer found in the model's response.")
        except Exception as e:
            print(f"\nError parsing model's answer: {str(e)}")
        
        # Ask if the user wants to continue with another example
        while True:
            user_input = input("\nTry another example? Enter a number (0-99) for a specific example, 'n' for next, or 'q' to quit: ").strip().lower()
            
            if user_input == 'q':
                continue_demo = False
                break
            elif user_input == 'n':
                current_example_idx = (current_example_idx + 1) % len(eval_dataset)
                break
            elif user_input.isdigit() and 0 <= int(user_input) < len(eval_dataset):
                current_example_idx = int(user_input)
                break
            else:
                print("Invalid input. Please try again.")

def print_messages(messages: List[Dict[str, str]], skip_system: bool = False, skip_few_shot: bool = False):
    """Print the conversation messages in a readable format."""
    start_idx = 0
    if skip_system:
        start_idx = 1
    
    if skip_few_shot and len(messages) > 2:
        # Find where the actual conversation starts (after system and few-shot)
        for i, msg in enumerate(messages):
            if msg["role"] == "user" and i > start_idx:
                start_idx = i
                break
    
    for i, msg in enumerate(messages[start_idx:], start=start_idx):
        role = msg["role"].upper()
        print(f"\n[{role}]:")
        print(msg["content"])

def run_batch_evaluation(model_name: str, num_examples: int = 10, max_steps: int = 5,
                         vllm_device="cuda", gpu_memory_utilization=0.9, 
                         vllm_dtype="auto", enable_prefix_caching=True, max_model_len=None):
    """
    Run batch evaluation on multiple examples and report overall accuracy.
    
    Args:
        model_name: Name of the model to use
        num_examples: Number of examples to evaluate
        max_steps: Maximum number of steps per example
        vllm_device: Device to run VLLM on ("cuda", "cpu")
        gpu_memory_utilization: Fraction of GPU memory to use
        vllm_dtype: Data type for model weights
        enable_prefix_caching: Whether to enable prefix caching
        max_model_len: Maximum model sequence length
    """
    print(f"Loading model: {model_name}")
    # Initialize VLLM model with detailed configuration
    model = LLM(
        model=model_name,
        device=vllm_device,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=vllm_dtype,
        enable_prefix_caching=enable_prefix_caching,
        max_model_len=max_model_len,
    )
    
    # Initialize tool environment for GSM8K
    print("Initializing environment...")
    vf_env = vf.ToolEnv(
        dataset="gsm8k",
        few_shot=CALCULATOR_FEW_SHOT[0],
        tools=[calculator],
        sleep_time=0.0,
        max_steps=max_steps
    )
    
    # Get the evaluation dataset
    eval_dataset = vf_env.get_eval_dataset(n=num_examples)
    
    # Parser for extracting answers
    from verifiers.parsers import XMLParser
    parser = XMLParser(fields=["reasoning", ("tool", "answer")])
    
    # Track results
    correct_count = 0
    total_count = 0
    results = []
    
    print(f"\nEvaluating {num_examples} examples...")
    
    for idx, example in enumerate(eval_dataset):
        print(f"\nProcessing example {idx+1}/{num_examples}...")
        
        # Set up the initial messages
        messages = [
            {"role": "system", "content": vf_env.system_prompt}
        ]
        
        # Add few-shot examples if available
        if vf_env.few_shot:
            for msg in vf_env.few_shot:
                messages.append(msg)
        
        # Add the user query
        messages.append({"role": "user", "content": example['question']})
        
        # Run the conversation loop
        step = 0
        completed = False
        
        while not completed and step < max_steps:
            step += 1
            
            # Get model response
            sampling_params = SamplingParams(
                temperature=0.1,
                max_tokens=600,
                stop=["</tool>", "</answer>"],
                include_stop_str_in_output=True
            )
            
            response = model.chat(
                messages=messages,
                sampling_params=sampling_params
            )
            
            # Add model response to messages
            assistant_msg = {"role": "assistant", "content": response[0].outputs[0].text}
            messages.append(assistant_msg)
            
            # Check if we're done
            if vf_env.is_completed(messages):
                completed = True
            else:
                # Get environment response
                env_msg = vf_env.env_response(messages)
                messages.append(env_msg)
        
        # Evaluate the answer
        try:
            # Find the last assistant message
            assistant_msgs = [msg for msg in messages if msg['role'] == 'assistant']
            if assistant_msgs:
                last_msg = assistant_msgs[-1]
                parsed = parser.parse(last_msg['content'])
                
                if hasattr(parsed, 'answer') and parsed.answer is not None:
                    is_correct, explanation = compare_answers(parsed.answer, example['answer'])
                    
                    if is_correct:
                        correct_count += 1
                    
                    results.append({
                        "example_idx": idx,
                        "question": example['question'],
                        "reference_answer": example['answer'],
                        "model_answer": parsed.answer,
                        "is_correct": is_correct,
                        "explanation": explanation
                    })
                    
                    print(f"  Result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
                else:
                    results.append({
                        "example_idx": idx,
                        "question": example['question'],
                        "reference_answer": example['answer'],
                        "model_answer": "No answer provided",
                        "is_correct": False,
                        "explanation": "No answer field found in response"
                    })
                    print("  Result: ✗ INCORRECT (No answer provided)")
        except Exception as e:
            results.append({
                "example_idx": idx,
                "question": example['question'],
                "reference_answer": example['answer'],
                "model_answer": "Error parsing response",
                "is_correct": False,
                "explanation": f"Error: {str(e)}"
            })
            print(f"  Result: ✗ INCORRECT (Error: {str(e)})")
        
        total_count += 1
    
    # Print summary
    accuracy = correct_count / total_count if total_count > 0 else 0
    print("\n" + "="*80)
    print(f"EVALUATION SUMMARY:")
    print(f"Model: {model_name}")
    print(f"Examples evaluated: {total_count}")
    print(f"Correct answers: {correct_count}")
    print(f"Accuracy: {accuracy:.2%}")
    print("="*80)
    
    return results, accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a demo of the GSM8K calculator tool environment")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", 
                        help="Model to use for the demo")
    parser.add_argument("--example", type=int, default=0, 
                        help="Index of the example to use from the GSM8K test set")
    parser.add_argument("--max_steps", type=int, default=5,
                        help="Maximum number of steps to run")
    parser.add_argument("--batch", action="store_true",
                        help="Run batch evaluation instead of single example")
    parser.add_argument("--num_examples", type=int, default=10,
                        help="Number of examples to evaluate in batch mode")
    # Add VLLM-specific arguments
    parser.add_argument("--vllm_device", type=str, default="cuda",
                        help="Device to run VLLM on (cuda, cpu)")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9,
                        help="Fraction of GPU memory to use")
    parser.add_argument("--vllm_dtype", type=str, default="auto",
                        help="Data type for model weights (auto, float16, bfloat16)")
    parser.add_argument("--vllm_enable_prefix_caching", action="store_true", default=True,
                        help="Enable prefix caching for faster generation")
    parser.add_argument("--vllm_max_model_len", type=int, default=None,
                        help="Maximum model sequence length")
    
    args = parser.parse_args()
    
    if args.batch:
        run_batch_evaluation(
            args.model, 
            args.num_examples, 
            args.max_steps,
            args.vllm_device,
            args.vllm_gpu_memory_utilization,
            args.vllm_dtype,
            args.vllm_enable_prefix_caching,
            args.vllm_max_model_len
        )
    else:
        run_demo(
            args.model, 
            args.example, 
            args.max_steps,
            args.vllm_device,
            args.vllm_gpu_memory_utilization,
            args.vllm_dtype,
            args.vllm_enable_prefix_caching,
            args.vllm_max_model_len
        )
