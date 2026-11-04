"""Local Hugging Face model loading and greedy inference, without autograd."""

from __future__ import annotations

from typing import Any

import torch


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def select_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    try:
        target = torch.device(requested)
    except (RuntimeError, ValueError, TypeError) as exc:
        raise ValueError("device must be auto, cpu, mps, cuda or cuda:N") from exc
    if target.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; install a CUDA-enabled PyTorch build or select cpu/auto")
        index = int(requested.split(":", 1)[1]) if ":" in requested else torch.cuda.current_device()
        if not 0 <= index < torch.cuda.device_count():
            raise ValueError(f"CUDA device index {index} exceeds the available device count")
        resolved = torch.device("cuda", index)
        if resolved.index != index:
            raise ValueError(f"CUDA device index {index} cannot be represented by this PyTorch build")
        return resolved
    if target.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable; use cpu or auto")
    if target.type not in {"cpu", "mps"}:
        raise ValueError("device must be auto, cpu, mps, cuda or cuda:N")
    return target


@torch.no_grad()
def load_model(
    model_id: str = DEFAULT_MODEL,
    device: str = "auto",
    dtype: str = "float32",
    local_files_only: bool = False,
) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    target = select_device(device)
    types = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if dtype not in types:
        raise ValueError(f"unsupported model dtype: {dtype}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=types[dtype], local_files_only=local_files_only,
        attn_implementation="eager",
    )
    model.eval().requires_grad_(False)
    return model.to(target), tokenizer


def encode_prompt(tokenizer: Any, prompt: str, device: torch.device) -> Any:
    """Use the same chat formatting for profiling, triggers and generation."""
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
        )
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    else:
        encoded = tokenizer(prompt, return_tensors="pt")
    return encoded.to(device)


@torch.no_grad()
def generate(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int = 64) -> str:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    encoded = encode_prompt(tokenizer, prompt, next(model.parameters()).device)
    output = model.generate(
        **encoded, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id,
    )
    return tokenizer.decode(output[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True)