"""Generic inference runner with prompt instructions loaded from JSON files."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


LOGGER = logging.getLogger(__name__)

DEFAULT_SYSTEM_INSTRUCTION = (
    "You are a reasoning assistant that reasons step by step, and puts your final "
    "answer within \\boxed{}."
)
DEFAULT_USER_INSTRUCTION = (
    "Solve the given logical deductive reasoning problem using only the information "
    "explicitly provided in the question. Do not use or rely on any external "
    'knowledge, assumptions, or common sense. If the problem cannot be definitively '
    'solved from the provided information, return "Uncertain".'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", type=str, required=True, help="Path or name of the base model.")
    parser.add_argument("--dataset_path", type=Path, required=True, help="Path to the input JSON/JSONL dataset.")
    parser.add_argument("--save_dir", type=Path, required=True, help="Directory to save the output.")
    parser.add_argument("--save_name", type=str, required=True, help="Output filename.")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling.")
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--question_field", type=str, default="question", help="Dataset field used as the user input.")
    parser.add_argument(
        "--prompt_file",
        type=Path,
        default=None,
        help="Prompt file containing system and user instructions in JSON format.",
    )
    parser.add_argument(
        "--prompt_phase",
        type=str,
        default=None,
        help="Phase name to extract from a nested prompt JSON file, for example 'translate' or 'solve'.",
    )
    parser.add_argument(
        "--system_instruction",
        type=str,
        default=None,
        help="Optional explicit system instruction override.",
    )
    parser.add_argument(
        "--user_instruction",
        type=str,
        default=None,
        help="Optional explicit user instruction override.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except AttributeError:
        pass
    return "cpu"


def load_prompt_instructions(prompt_file: Path, phase: str | None = None) -> tuple[str, str]:
    prompt_data = json.loads(prompt_file.read_text(encoding="utf-8"))
    if not isinstance(prompt_data, dict):
        raise ValueError(f"Prompt file {prompt_file} must contain a JSON object.")

    if "system_instruction" in prompt_data or "user_instruction" in prompt_data:
        system_instruction = prompt_data.get("system_instruction")
        user_instruction = prompt_data.get("user_instruction")
    else:
        if phase is None:
            raise ValueError(
                f"Prompt file {prompt_file} contains multiple sections. Pass --prompt_phase."
            )
        section = prompt_data.get(phase)
        if not isinstance(section, dict):
            raise ValueError(f"Prompt phase {phase!r} not found in {prompt_file}.")
        system_instruction = section.get("system_instruction")
        user_instruction = section.get("user_instruction")

    if not isinstance(system_instruction, str) or not system_instruction.strip():
        raise ValueError(f"Prompt file {prompt_file} is missing a non-empty system_instruction.")
    if not isinstance(user_instruction, str) or not user_instruction.strip():
        raise ValueError(f"Prompt file {prompt_file} is missing a non-empty user_instruction.")
    return system_instruction.strip(), user_instruction.strip()


def resolve_instructions(args: argparse.Namespace) -> tuple[str, str]:
    system_instruction = DEFAULT_SYSTEM_INSTRUCTION
    user_instruction = DEFAULT_USER_INSTRUCTION

    if args.prompt_file is not None:
        system_instruction, user_instruction = load_prompt_instructions(
            args.prompt_file.resolve(),
            args.prompt_phase,
        )

    if args.system_instruction is not None:
        system_instruction = args.system_instruction.strip()
    if args.user_instruction is not None:
        user_instruction = args.user_instruction.strip()

    return system_instruction, user_instruction


def build_model_and_tokenizer(args: argparse.Namespace) -> tuple[str, Any, Any]:
    device = detect_device()
    LOGGER.info("Running on device: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype="auto",
        cache_dir=args.cache_dir,
    )
    model.eval()
    return device, tokenizer, model


def evaluate(
    model: Any,
    tokenizer: Any,
    device: str,
    system_instruction: str,
    prompt: str,
    temperature: float,
    do_sample: bool,
    num_return_sequences: int,
    max_new_tokens: int,
) -> str | list[str]:
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_return_sequences,
        do_sample=do_sample,
        temperature=temperature,
    )
    input_length = model_inputs.input_ids.shape[1]
    generated_ids = [output_ids[input_length:] for output_ids in generated_ids]
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    if num_return_sequences == 1:
        return responses[0]
    return responses


def load_samples(dataset_path: Path) -> list[dict[str, Any]]:
    data = load_dataset("json", data_files=str(dataset_path))
    return list(data["train"])


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    system_instruction, user_instruction = resolve_instructions(args)
    device, tokenizer, model = build_model_and_tokenizer(args)
    samples = load_samples(args.dataset_path.resolve())

    args.save_dir.mkdir(parents=True, exist_ok=True)
    save_path = args.save_dir / args.save_name

    with save_path.open("a", encoding="utf-8") as handle:
        for sample in tqdm(samples):
            question = str(sample[args.question_field])
            input_prompt = f"{user_instruction}\n{question}"

            reason_answer = evaluate(
                model=model,
                tokenizer=tokenizer,
                device=device,
                system_instruction=system_instruction,
                prompt=input_prompt,
                max_new_tokens=args.max_length,
                temperature=args.temperature,
                num_return_sequences=args.num_return_sequences,
                do_sample=args.do_sample,
            )

            output = {
                "id": sample.get("id"),
                "num_edits_preserving_label": sample.get("num_edits_preserving_label"),
                "question": question,
                "label": sample.get("label"),
                "output": reason_answer,
            }
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
