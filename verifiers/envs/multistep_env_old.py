from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
import random
import time
from typing import List, Dict, Sequence, Any, Union, Tuple

from datasets import Dataset
from trl.trainer.grpo_trainer import RewardFunc
from ..utils.imports import LLM, SamplingParams  # type: ignore
from rich.console import Console
from rich.text import Text
import re

from verifiers.envs.environment import Environment

console = Console()

class MultiStepEnv(Environment):
    def __init__(self,
                 system_prompt: str = "",
                 few_shot: List[Dict[str, str]] = [],
                 sampling_args: Dict[str, Any] = {},
                 mask_env_response: bool = True,
                 max_workers: int = 10,
                 max_steps: int = 10,
                 sleep_time: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.system_prompt = system_prompt
        self.few_shot = few_shot
        self.sampling_args = {
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
            "n": 1
        }
        self.sampling_args.update(sampling_args)
        self.env_mask = 0 if mask_env_response else 1
        self.max_workers = max_workers
        self.sleep_time = sleep_time
        self.max_steps = max_steps
    def get_dataset(self, **kwargs: Any) -> Dataset | None:
        pass

    def get_eval_dataset(self, **kwargs: Any) -> Dataset | None:
        pass

    @abstractmethod
    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        pass

    @abstractmethod
    def is_completed(self, messages: List[Dict[str, str]], **kwargs: Any) -> bool:
        pass

    @abstractmethod
    def env_response(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, str]:
        pass

    def step(self,
             states: List[Dict[str, Any]],
             llm: LLM,
             sampling_params: SamplingParams) -> List[Dict[str, Any]]:
        
        live_indices = [i for i, s in enumerate(states) if not s["completed"]]
        messages_to_step = [states[i]["messages"] for i in live_indices]
        llm_responses = llm.chat(messages_to_step, sampling_params=sampling_params, use_tqdm=False) # type: ignore

        for i, j in enumerate(live_indices):
            # Get the current state
            if len(states[j]["prompt_ids"]) == 0:
                states[j]["prompt_ids"] = llm_responses[i].prompt_token_ids
            states[j]["messages"].append({"role": "assistant", "content": llm_responses[i].outputs[0].text})
        
            # Calculate what's new in this step
            prev_completion_len = len(states[j]["completion_ids"])
            
            # Update completion ids with new tokens
            new_completion_ids = list(llm_responses[i].prompt_token_ids)[len(states[j]["prompt_ids"]) + prev_completion_len:] # type: ignore
            new_completion_ids.extend(list(llm_responses[i].outputs[0].token_ids))
            
            # Extend the existing completion_ids with the new tokens
            states[j]["completion_ids"].extend(new_completion_ids)
            
            # Update the mask - env response tokens get env_mask, LLM response tokens get 1
            env_response_len = len(new_completion_ids) - len(llm_responses[i].outputs[0].token_ids)
            # Ensure env_response_len is not negative
            env_response_len = max(0, env_response_len)
            states[j]["completion_mask"].extend([self.env_mask] * env_response_len)
            states[j]["completion_mask"].extend([1] * len(llm_responses[i].outputs[0].token_ids))
            
            # ==== local debugging ====
            DEBUG = False
            if DEBUG:
                debug_text = Text()                
                prompt_text = "\n".join([f"{m['role']}: {m['content']}" for m in states[j]["messages"][:states[j]["prompt_messages"]]])
                debug_text.append(f"PROMPT:\n{prompt_text}\n\n", style="dim")                
                debug_text.append("COMPLETIONS:\n", style="bold")                
                tokenizer = llm.get_tokenizer()                
                for idx, (token_id, mask) in enumerate(zip(states[j]["completion_ids"], states[j]["completion_mask"])):
                    token_text = tokenizer.decode([token_id])
                    if mask == self.env_mask:
                        # Environment tokens in red
                        debug_text.append(token_text, style="red")
                    else:
                        # LLM tokens in green
                        debug_text.append(token_text, style="green")
                console.print("\n" + "="*80)
                console.print(f"STATE {j} - STEP {len(states[j]['messages']) - states[j]['prompt_messages']}")
                console.print("="*80)
                console.print(debug_text)
                console.print("="*80 + "\n")
            # ==== end of local debugging ====

            if (
                self.is_completed(states[j]["messages"])
                or len(states[j]["completion_ids"]) > sampling_params.max_tokens
            ):
                states[j]["completed"] = True
                states[j]["completion_ids"] = states[j]["completion_ids"][: sampling_params.max_tokens]

            else:
                states[j]["messages"].append(self.env_response(states[j]["messages"]))

        return states

    def generate(self, prompts: List[List[Dict[str, Any]]],
                 llm: LLM,
                 sampling_params: SamplingParams,
                 **kwargs: Any) -> Dict[str, List[Sequence[int]] | List[str] |  List[List[Dict[str, Any]]]]:
        custom_sp = sampling_params.clone()
        for k, v in self.sampling_args.items():
            setattr(custom_sp, k, v)

        # initialize state variables
        all_completed = False
        states = [{
            "messages": m,
            "prompt_messages": len(m),
            "prompt_ids": [],
            "completed": False,
            "completion_ids": [],
            "completion_mask": []
        } for m in prompts]

        # main loop
        while not all_completed:
            states = self.step(states, llm, custom_sp)
            all_completed = all(state["completed"] for state in states)

        completion_messages = [s["messages"][s["prompt_messages"]:] for s in states]
        completion_ids = [s["completion_ids"] for s in states]
        completion_mask = [s["completion_mask"] for s in states]
        output = {
            "ids": completion_ids,
            "messages": completion_messages,
            "mask": completion_mask
        }
        return output

    def step_api(self, 
             client: Any,
             model: str,
             messages: List[Dict[str, str]],
             **kwargs: Any) -> Tuple[List[Dict[str, str]], bool]:
        """
        Execute a single step using OpenAI API, including environment response if needed.
        
        Args:
            client: OpenAI client instance
            messages: Conversation history
            model: Model name to use
            **kwargs: Additional arguments for the chat completion API
        
        Returns:
            Updated messages list with assistant response and possibly environment response
        """
        messages_copy = messages.copy()
        
        try:            
            # Get assistant response
            response = client.chat.completions.create(
                model=model,
                messages=messages_copy,
            )
            
            # Add assistant response to messages
            assistant_msg = {
                "role": "assistant", 
                "content": response.choices[0].message.content
            }
            messages_copy.append(assistant_msg)
            
            # Check if we're done
            if self.is_completed(messages_copy):
                rollout_is_completed = True
            else:
                rollout_is_completed = False
                # If not done, get and add environment response
                env_msg = self.env_response(messages_copy)
                messages_copy.append(env_msg)
            
            return messages_copy, rollout_is_completed
            
        except Exception as e:
            # Handle errors by adding error message and returning
            error_msg = {"role": "assistant", "content": f"Error in API call: {str(e)}"}
            messages_copy.append(error_msg)
            return messages_copy, True
    
    def eval_api(self, 
                client: Any,
                model: str,
                max_concurrent: int = 32,
                timeout: int = 60,
                sampling_args: Dict[str, Any] = {},
                **kwargs: Any):
        """
        Evaluate model using OpenAI API with proper concurrency.
        
        Args:
            client: OpenAI client instance
            model: Model name as string
            max_concurrent: Maximum number of concurrent API calls
            timeout: Maximum seconds to wait for each example
            sampling_args: Arguments specific to sampling (separate from env sampling_args)
            **kwargs: Additional arguments for evaluation
        
        Returns:
            Tuple of (eval_dataset, rewards)
        """
        def run_evaluation():
            # Import libraries here to avoid requiring them for normal operation
            import asyncio
            from asyncio import Semaphore
            # Get the evaluation dataset
            if self.eval_dataset is None:
                self.eval_dataset = self.get_eval_dataset(**kwargs)
                
            if self.eval_dataset is None:
                raise ValueError("Failed to load evaluation dataset")
            
            eval_dataset = self.eval_dataset
            
            async def process_example(example, semaphore):
                async with semaphore:
                    # Initialize conversation with system prompt and few-shot examples
                    prompt = example["prompt"]
                    messages = example["prompt"].copy()
                    answer = example["answer"]
                    
                    # Save the length of initial messages to extract just the interaction part later
                    initial_length = len(messages)

                    # Run the conversation loop until completion or max steps
                    for _ in range(self.max_steps):  # Safety limit on conversation turns
                        try:
                            # Run step_api to get model and environment response
                            # Note: step_api now returns a tuple (messages, is_completed)
                            step_result = await asyncio.get_event_loop().run_in_executor(
                                None,
                                lambda: self.step_api(
                                    client=client,
                                    model=model,
                                    messages=messages,
                                    **sampling_args
                                )
                            )
                            
                            # Unpack the step_api result
                            messages, is_completed = step_result
                            
                            # If the rollout is completed, break the loop
                            if is_completed:
                                break
                            
                        except Exception as e:
                            print(f"Error processing example {example.get('id', 'unknown')}: {str(e)}")
                            break
                    
                    # Extract only the interaction part (not system/few-shot)
                    completions = messages[initial_length:]
                    
                    return {
                        "prompt": prompt,
                        "completions": completions,
                        "answer": answer
                    }
            
            async def run_all_examples():
                # Create semaphore for concurrency control
                from tqdm.asyncio import tqdm_asyncio

                semaphore = Semaphore(max_concurrent)
                
                # Process all examples concurrently
                tasks = [process_example(example, semaphore) for example in eval_dataset]
                results = await tqdm_asyncio.gather(
                    *tasks,
                    total=len(eval_dataset),
                    desc=f"Evaluating {len(eval_dataset)} examples"
                )
                
                return results
            
            # Run the async evaluation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(run_all_examples())
            finally:
                loop.close()
            
            # Calculate rewards
            results_prompt = [result["prompt"] for result in results]
            results_answer = [result["answer"] for result in results]
            results_completions = [result["completions"] for result in results]
            results = {"prompt": results_prompt, "answer": results_answer, "completions": results_completions}
            
            reward_funcs = self.get_rubric()
            rewards = {}
            
            for reward_func in reward_funcs:
                func_rewards = reward_func(**results) # type: ignore
                func_reward_avg = sum(func_rewards) / len(func_rewards)
                func_name = reward_func.__name__ # type: ignore
                print(f"{func_name}: {func_reward_avg}")
                rewards[func_name] = func_reward_avg
            
            return rewards
            
        # Run the evaluation function
        return run_evaluation()
    

    