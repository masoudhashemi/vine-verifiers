import verifiers as vf
from datasets import Dataset
from functools import partial
import argparse
import torch
from verifiers.utils.data_utils import format_prompt
from verifiers.envs.knight_and_knaves.kk_verification import extract_answer, parse_statements, verify_kk_puzzle

model_name = "Qwen/Qwen2.5-3B-Instruct"
model, tokenizer = vf.get_model_and_tokenizer(model_name)

vf_env = vf.KnightsKnavesSimpleEnv(
    size=None,
    complexity=None
)

# Get the dataset and rubric
raw_dataset = vf_env.get_dataset()
#shuffle the dataset
raw_dataset = raw_dataset.shuffle()
eval_dataset = vf_env.get_eval_dataset()

# Format dataset for TRL's expected format
def format_for_trl(example):
    return {
        "prompt": format_prompt(
            prompt=example["prompt"],
            system_prompt=vf_env.system_prompt
        ),
        "answer": example["answer"],
        "reference": example["prompt"]
    }

# Convert to the format expected by TRL
dataset = Dataset.from_dict({
    "prompt": [format_for_trl(example)["prompt"] for example in raw_dataset],
    "answer": [example["answer"] for example in raw_dataset],
    "reference": [example["prompt"] for example in raw_dataset]
})

# Convert eval dataset to the format expected by TRL
eval_trl_dataset = Dataset.from_dict({
    "prompt": [format_for_trl(example)["prompt"] for example in eval_dataset],
    "answer": [example["answer"] for example in eval_dataset],
    "reference": [example["prompt"] for example in eval_dataset]
})

# Wrap reward functions to match expected signature
def wrap_reward_func(func):
    def wrapped(prompts, completions, **kwargs):
        results = []
        for i, completion_msgs in enumerate(completions):
            # Extract the completion text from the message
            completion_text = completion_msgs[0]["content"] if completion_msgs else ""
            # Call the original reward function with the right arguments
            if func.__name__ == "check_format":
                score = func(completion=completion_text)
            else:  # check_solution
                reference = kwargs.get("reference", [""])[i]
                score = func(completion=completion_text, reference=reference)
            results.append(score)
        return results
    
    # Preserve the function name
    wrapped.__name__ = func.__name__
    return wrapped

# Wrap the reward functions
raw_rubric = vf_env.get_rubric()
rubric = [wrap_reward_func(func) for func in raw_rubric]

def compute_kk_metrics(eval_pred):
    predictions, labels, inputs = eval_pred
    references = [inp["reference"] for inp in inputs]
    total_puzzles = len(predictions)
    correct_puzzles = 0
    
    for i, (prediction, reference) in enumerate(zip(predictions, references)):
        try:
            completion_text = prediction[0]["content"] if prediction else ""
            answer_dict = extract_answer(completion_text)
            statements = parse_statements(reference)
            is_valid, _ = verify_kk_puzzle(statements, answer_dict)
            if is_valid:
                correct_puzzles += 1
        except Exception as e:
            print(f"Error evaluating puzzle {i}: {e}")
            continue
    
    accuracy = correct_puzzles / total_puzzles if total_puzzles > 0 else 0
    return {
        "kk_accuracy": accuracy,
    }

# Configure training
run_name = "knights_knaves_" + model_name.split("/")[-1].lower()
training_args = vf.get_default_grpo_config(run_name=run_name, num_gpus=8, reward_weights=[0.2, 0.8])

parser = argparse.ArgumentParser(description="Train a model on Knights and Knaves puzzles")
parser.add_argument("--use_reward_filtering", action="store_true", help="Enable reward filtering")
parser.add_argument("--reward_min_threshold", type=float, default=0.1, help="Minimum reward threshold")
parser.add_argument("--reward_max_threshold", type=float, default=0.9, help="Maximum reward threshold")
parser.add_argument("--max_reward", type=float, default=1.5, help="Maximum reward")
parser.add_argument("--use_sequential_sampler", action="store_true", help="Use sequential sampling instead of random")
parser.add_argument("--advantage_type", type=str, default="standardized", choices=["standardized", "reinforcepp", "loo"], 
                    help="Type of advantage calculation to use")
args = parser.parse_args()

training_args.learning_rate = 1e-6
training_args.max_completion_length = 2048
# rollouts per prompt
training_args.num_generations = 14
# minibatch size per GPU ( bs 2 * 7 gpus / 14 rollouts -> 1 prompts per batch)
training_args.per_device_train_batch_size = 6
# batches to accumulate (1 prompts * 2 -> 2 prompts per global batch)
training_args.gradient_accumulation_steps = 4
# steps per global batch (1 on-policy, 2 off-policy)
training_args.num_iterations = 1
# ref model and kl penalty
training_args.beta = 0.04
training_args.epsilon = 0.2
# max_grad_norm
training_args.max_grad_norm = 1.0
# seed
training_args.seed = 1234

# number of epochs
training_args.num_train_epochs = 1

# evals
# training_args.eval_on_start = True
# training_args.eval_strategy = "steps"
# training_args.per_device_eval_batch_size = 8
# training_args.eval_accumulation_steps = 1
# training_args.eval_steps = 100


# Initialize trainer with reward filtering options and sequential sampler option
trainer = vf.GRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=rubric, 
    env=vf_env,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=eval_trl_dataset,
    use_reward_filtering=args.use_reward_filtering,
    reward_min_threshold=args.reward_min_threshold * args.max_reward,
    reward_max_threshold=args.reward_max_threshold * args.max_reward,
    advantage_type=args.advantage_type,
    use_sequential_sampler=args.use_sequential_sampler,
    # compute_metrics=compute_kk_metrics,
)

# Start training
trainer.train()
print("Saving model...")
trainer.save_model("/mnt/core_llm/masoud/model_cache/knights_knaves_simple")

# Run a final evaluation
print("Running final evaluation...")
eval_results = trainer.evaluate()
print(f"Final evaluation results: {eval_results}")
