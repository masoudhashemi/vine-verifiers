import argparse
import math
import re
import verifiers as vf
from verifiers.tools import calculator
from verifiers.prompts import CALCULATOR_FEW_SHOT

# model_name = "Qwen/Qwen2.5-1.5B-Instruct"
model_name = "Qwen/Qwen2.5-3B-Instruct"
model, tokenizer = vf.get_model_and_tokenizer(model_name)

# Define a custom numeric answer reward function
def extract_number(text: str) -> float:
    """Extract the first number from text, handling various formats."""
    if not text:
        return float('nan')
        
    # Remove commas in numbers and find all numbers in the text
    text = text.replace(',', '')
    numbers = re.findall(r'-?\d+\.?\d*', text)
    
    if not numbers:
        return float('nan')
        
    # Return the first number found as a float
    return float(numbers[0])

def numeric_answer_reward_func(completions, answer, **kwargs) -> list:
    """
    Reward function that compares numeric values in answers.
    
    Extracts numbers from both the model's answer and the reference answer,
    then compares their values with a small tolerance for floating-point differences.
    """
    rewards = []
    
    for completion, ref_answer in zip(completions, answer):
        # Default reward is 0
        reward = 0.0
        
        # Find the last assistant message
        assistant_msgs = [msg for msg in completion if msg['role'] == 'assistant']
        if not assistant_msgs:
            rewards.append(reward)
            continue
            
        last_msg = assistant_msgs[-1]
        
        try:
            # Parse the message to extract the answer field
            from verifiers.parsers import XMLParser
            parser = XMLParser(fields=["reasoning", ("tool", "answer")])
            parsed = parser.parse(last_msg['content'])
            if hasattr(parsed, 'answer') and parsed.answer is not None:
                # Extract numbers from both answers
                model_number = extract_number(parsed.answer)
                reference_number = extract_number(ref_answer)
                
                # Check if both are valid numbers
                if not (math.isnan(model_number) or math.isnan(reference_number)):
                    # Compare with a small tolerance (0.001 or 0.1% relative difference)
                    abs_diff = abs(model_number - reference_number)
                    rel_diff = abs_diff / max(abs(reference_number), 1e-10)
                    
                    if abs_diff < 1e-10 or rel_diff < 0.001:
                        reward = 1.0
        except Exception as e:
            # If parsing fails, reward remains 0
            pass
            
        rewards.append(reward)
        
    return rewards

# Initialize tool environment for GSM8K
vf_env = vf.ToolEnv(
    dataset="gsm8k",
    few_shot=CALCULATOR_FEW_SHOT[0],
    tools=[calculator],
    sleep_time=0.0,
    max_steps=5
)
dataset = vf_env.get_dataset()
eval_dataset = vf_env.get_eval_dataset(n=100)

# Get the default rubric and replace the first reward function with our numeric one
rubric = vf_env.get_rubric()
# Replace the first reward function (which should be the answer reward)
rubric[0] = numeric_answer_reward_func

# notable defaults: lr = 1e-6, max_grad_norm = 0.01, constant lr 10 warmup steps, 1024 tokens in+out
run_name = "gsm8k-calc_" + model_name.split("/")[-1].lower()
training_args = vf.get_default_grpo_config(
    run_name=run_name,
    num_gpus=8
)

parser = argparse.ArgumentParser(description="Train a model on Knights and Knaves puzzles")
parser.add_argument("--use_reward_filtering", action="store_true", help="Enable reward filtering")
parser.add_argument("--reward_min_threshold", type=float, default=0.1, help="Minimum reward threshold")
parser.add_argument("--reward_max_threshold", type=float, default=0.9, help="Maximum reward threshold")
parser.add_argument("--max_reward", type=float, default=1.0, help="Maximum reward")
parser.add_argument("--use_sequential_sampler", action="store_true", help="Use sequential sampling instead of random")
parser.add_argument("--advantage_type", type=str, default="standardized", choices=["standardized", "reinforcepp", "loo"], 
                    help="Type of advantage calculation to use")
args = parser.parse_args()

training_args.learning_rate = 1e-6
training_args.max_completion_length = 600
# rollouts per prompt
training_args.num_generations = 14
# minibatch size per GPU ( bs 6 * 7 gpus / 7 rollouts -> 6 prompts per batch)
training_args.per_device_train_batch_size = 6
# batches to accumulate (6 prompts * 4 -> 32 prompts per global batch)
training_args.gradient_accumulation_steps = 4
# steps per global batch (1 on-policy, 1 off-policy)
training_args.num_iterations = 1
# no ref model
training_args.beta = 0.04
# evals
#training_args.eval_strategy = "steps"
##training_args.eval_on_start = True
#training_args.eval_steps = 100
# training_args.per_device_eval_batch_size = 8
# training_args.eval_accumulation_steps = 1

trainer = vf.GRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=rubric,
    env=vf_env,
    args=training_args,
    train_dataset=dataset,
    use_reward_filtering=args.use_reward_filtering,
    reward_min_threshold=args.reward_min_threshold * args.max_reward,
    reward_max_threshold=args.reward_max_threshold * args.max_reward,
    advantage_type=args.advantage_type,
    use_sequential_sampler=args.use_sequential_sampler,
    #eval_dataset=eval_dataset,
)

trainer.train() 