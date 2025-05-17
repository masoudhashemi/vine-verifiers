from contextlib import contextmanager
from typing import Union

import torch
import transformers
from accelerate import Accelerator
from accelerate.utils import is_deepspeed_available
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from jinja2 import Template
except ImportError:
    print("jinja2 not installed, chat template functionality may be limited")

if is_deepspeed_available():
    import deepspeed

    DeepSpeedEngine = deepspeed.DeepSpeedEngine
else:
    DeepSpeedEngine = None


def load_model_tokenizer(model_name_or_path, tokenizer_name=None, peft_r=None, peft_alpha=None, args=None):
    """Load model and tokenizer with proper chat template"""
    # Load tokenizer
    tokenizer_name = tokenizer_name if tokenizer_name else model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True, padding_side="left")

    # Set padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add default chat template if not supported
    if not hasattr(tokenizer, "apply_chat_template"):
        DEFAULT_CHAT_TEMPLATE = (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}System: {{ message['content'] }}\n{% endif %}"
            "{% if message['role'] == 'user' %}Human: {{ message['content'] }}\n{% endif %}"
            "{% if message['role'] == 'assistant' %}Assistant: {{ message['content'] }}\n{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}Assistant: {% endif %}"
        )
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

        def apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **kwargs):
            from jinja2 import Template

            template = Template(tokenizer.chat_template)
            chat_text = template.render(messages=messages, add_generation_prompt=add_generation_prompt)
            if tokenize:
                return tokenizer(chat_text, **kwargs).input_ids
            return chat_text

        tokenizer.apply_chat_template = apply_chat_template

    # Prepare model kwargs
    model_kwargs = {
        "pretrained_model_name_or_path": model_name_or_path,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    # For multi-GPU, either leave device_map unset or set it to auto.
    if not args.multi_gpu:
        model_kwargs["device_map"] = "cuda:0"
    else:
        model_kwargs["device_map"] = "auto"

    def class_to_use(model_name):
        if "falcon" in model_name.lower():
            return transformers.FalconForCausalLM
        return transformers.AutoModelForCausalLM

    if args.flash_attention:
        if torch.cuda.get_device_capability(0)[0] < 8:
            raise ValueError("Flash attention only works for CUDA compatibility >= 8")
        model = class_to_use(model_name_or_path).from_pretrained(
            torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", **model_kwargs
        )
        print("LOADED THE MODEL WITH FLASH ATTENTION")
        tokenizer.padding_side = "left"
    else:
        model = class_to_use(model_name_or_path).from_pretrained(torch_dtype=torch.bfloat16, **model_kwargs)
        print("Loaded the model without flash attention")

    # Apply LoRA if specified
    if peft_r is not None:
        from peft import LoraConfig, get_peft_model

        alpha = max(peft_r * 2, 128) if peft_alpha is None else peft_alpha
        config = LoraConfig(
            r=peft_r, lora_alpha=alpha, lora_dropout=0.0, target_modules=["embed_tokens", "lm_head", "q_proj", "v_proj"]
        )
        model = get_peft_model(model, config)

    print(f"Model dtype: {model.dtype}")
    print("Warning... Setting pad token to EOS")
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        print("Resizing the embeddings to match the tokenizer size.")
        model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


@contextmanager
def unwrap_model_for_generation(
    model: Union[DistributedDataParallel, DeepSpeedEngine] if DeepSpeedEngine else DistributedDataParallel,
    accelerator: "Accelerator",
    is_peft_model: bool = False,
    gather_deepspeed3_params: bool = True,
):
    """Context manager to unwrap a model for generation."""
    unwrapped_model = accelerator.unwrap_model(model)
    if is_peft_model:
        unwrapped_model.pretrained_model.disable_adapter()
    if accelerator.state.deepspeed_plugin is not None and accelerator.state.deepspeed_plugin.zero_stage == 3:
        if not gather_deepspeed3_params:
            yield accelerator.unwrap_model(model)
        else:
            with deepspeed.zero.GatheredParameters(model.parameters()):
                yield accelerator.unwrap_model(model)
    else:
        yield unwrapped_model


def prepare_deepspeed(model):
    """Prepare model for DeepSpeed."""
    import deepspeed

    ds_config = {
        "train_batch_size": 1,
        "fp16": {"enabled": True},
        "zero_optimization": {"stage": 3},
    }
    model, *_ = deepspeed.initialize(model=model, config=ds_config)
    return model
