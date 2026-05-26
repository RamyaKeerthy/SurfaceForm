"""Symbolic CoT inference pipeline with dataset-specific prompt loading."""

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import re

import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


LOGGER = logging.getLogger(__name__)
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
DATA_ALIASES = {
    "arlsat": "arlsat",
    "folio": "folio",
    "logicaldeduction": "deduction",
    "deduction": "deduction",
}


def normalize_data_name(value: str) -> str:
    normalized = value.strip().lower()
    try:
        return DATA_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported dataset {value!r}. Expected one of: {', '.join(sorted(DATA_ALIASES))}."
        ) from exc


def infer_data_name(args: argparse.Namespace) -> str:
    if args.data:
        return normalize_data_name(args.data)

    candidates = (
        str(args.dataset_path),
        str(args.save_name),
        str(args.save_dir),
    )
    for candidate in candidates:
        lowered = candidate.lower()
        for needle, normalized in (
            ("logicaldeduction", "deduction"),
            ("deduction", "deduction"),
            ("arlsat", "arlsat"),
            ("folio", "folio"),
        ):
            if needle in lowered:
                return normalized

    raise ValueError(
        "Unable to infer dataset name for prompt loading. Pass --data explicitly."
    )


def load_prompt_bundle(data_name: str, prompt_dir: Path) -> dict[str, dict[str, str]]:
    prompt_path = prompt_dir / f"{data_name}-symboliccot_reasoning.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    prompt_values = json.loads(prompt_path.read_text(encoding="utf-8"))
    if not isinstance(prompt_values, dict):
        raise ValueError(f"Prompt file {prompt_path} must contain a JSON object.")

    required_phases = ("translate", "plan", "solve")
    for phase in required_phases:
        phase_prompts = prompt_values.get(phase)
        if not isinstance(phase_prompts, dict):
            raise ValueError(f"Prompt file {prompt_path} is missing object for phase {phase!r}.")

        for key in ("system_instruction", "user_instruction"):
            value = phase_prompts.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Prompt file {prompt_path} phase {phase!r} is missing non-empty {key!r}."
                )

    return prompt_values


def repair_json(s: str) -> str:
    out = []
    in_str = False
    esc = False
    i = 0
    valid_esc = {'"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u'}

    while i < len(s):
        ch = s[i]

        if not in_str:
            out.append(ch)
            if ch == '"':
                in_str = True
                esc = False
            i += 1
            continue

        # inside string
        if esc:
            # invalid escape -> escape the backslash
            if ch not in valid_esc:
                out.insert(-1, '\\')
            out.append(ch)
            esc = False
            i += 1
            continue

        if ch == '\\':
            # backslash + newline -> replace with \\n
            if i + 1 < len(s) and s[i + 1] in '\r\n':
                out.append('\\n')
                i += 2
                continue
            out.append('\\')
            esc = True
            i += 1
            continue

        if ch == '"':
            out.append(ch)
            in_str = False
            i += 1
            continue

        if ch == '\n':
            out.append('\\n')
            i += 1
            continue

        if ch == '\r':
            out.append('\\n')
            i += 1
            continue

        if ch == '\t':
            out.append('\\t')
            i += 1
            continue

        out.append(ch)
        i += 1

    return ''.join(out)

def parse_json(df, json_column):
    def safe_load(val):
        if isinstance(val, dict):
            return val
        if pd.isna(val):
            return {}

        s = str(val).strip()
        if not s:
            return {}

        if s.startswith("```"):
            s = s.strip("`").strip()
            if s.lower().startswith("json"):
                s = s[4:].strip()

        # keep from first '{'
        if "{" in s and not s.lstrip().startswith("{"):
            s = s[s.index("{"):]
        if "}" in s:
            s = s[:s.rindex("}")+1]

        s = repair_json(s)

        try:
            return json.loads(s)
        except json.JSONDecodeError as e:
            LOGGER.warning("Bad JSON: %s", e)
            LOGGER.warning("Preview: %r", s[:300])
            return {}
        # return json.loads(s)

    # Parse each row into a dict
    parsed = df[json_column].apply(safe_load)

    # Turn series of dicts into a dataframe
    expanded = pd.json_normalize(parsed)

    # Merge back to original dataframe
    for col in expanded.columns:
        df[col] = expanded[col]

    columns = [
        'id',
        'question',
        'output',
        'num_edits_preserving_label',
        'label',
        'predicates',
        'premises',
        'conclusion',
    ]
    existing_columns = [col for col in columns if col in df.columns]
    return df[existing_columns]

def extract_fol(text):
    """Extract the FOL part from 'NL ::: FOL' format."""
    if pd.isna(text):
        return None

    text = str(text)
    if " ::: " not in text:
        return None
    parts = text.split(" ::: ", 1)
    return parts[1].strip()

def format_question_symbolic(row):
    premises_raw = row["premises"]
    conclusion_raw = row["conclusion"]

    # Check if both contain NL ::: FOL format somewhere
    if (
        pd.isna(premises_raw) or
        pd.isna(conclusion_raw) or
        " ::: " not in str(premises_raw) or
        " ::: " not in str(conclusion_raw)
    ):
        return row["output"]

    premises_lines = str(premises_raw).split("\n")

    fol_premises = []
    for line in premises_lines:
        line = line.strip()
        if not line:
            continue

        fol = extract_fol(line)
        if fol is None:
            fol = line

        fol_premises.append(fol)

    conclusion_fol = extract_fol(conclusion_raw)
    if conclusion_fol is None:
        conclusion_fol = conclusion_raw

    premises_text = "\n".join(
        f"{i+1}. {p}" for i, p in enumerate(fol_premises)
    )

    return f"""Premises:
{premises_text}

Question:
{conclusion_fol}"""


def detect_device():
    if torch.cuda.is_available():
        return "cuda"
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except AttributeError:
        pass
    return "cpu"


def get_label(sample):
    if 'label' in sample:
        return sample['label']
    if 'answer' in sample:
        return sample['answer']
    return None


def get_question_text(sample, column_name):
    if column_name in sample:
        return sample[column_name]
    if 'question' in sample:
        return sample['question']
    raise KeyError(f"Column '{column_name}' not found in sample and no fallback 'question' field exists.")


def load_samples(dataset_path):
    data = load_dataset("json", data_files=dataset_path)
    return list(data['train'])


def normalize_plan_text(value):
    if isinstance(value, str):
        return re.sub(r'^Plan:\s*\n?', '', value, count=1)
    return value


def prepare_samples_for_phase(samples, phase):
    if phase == "translate":
        return samples

    df = pd.DataFrame(samples).copy()
    if df.empty:
        return samples

    if phase in {"plan", "solve"}:
        if 'question_symbolic' not in df.columns or df['question_symbolic'].isna().all() or (df['question_symbolic'].astype(str).str.strip() == '').all():
            parse_source_column = 'output' if 'output' in df.columns else None

            if parse_source_column:
                df = parse_json(df, parse_source_column)

            if {'premises', 'conclusion'}.issubset(df.columns):
                df['question_symbolic'] = df.apply(format_question_symbolic, axis=1)

    if phase == "solve":
        if 'plan' not in df.columns or df['plan'].isna().all() or (df['plan'].astype(str).str.strip() == '').all():
            if 'output' in df.columns:
                df['plan'] = df['output'].apply(normalize_plan_text)

        elif 'plan' in df.columns:
            df['plan'] = df['plan'].apply(normalize_plan_text)

    return df.to_dict(orient='records')


def build_model_and_tokenizer(args):
    device = detect_device()
    LOGGER.info("Running on device: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        attn_implementation="eager",
        torch_dtype='auto',
        cache_dir=args.cache_dir
    )
    model.eval()
    return device, tokenizer, model


def get_instructions(phase, prompts):
    try:
        phase_prompts = prompts[phase]
    except KeyError as exc:
        raise ValueError(f"Unsupported phase: {phase}") from exc
    return phase_prompts["system_instruction"], phase_prompts["user_instruction"]


def evaluate(
    model,
    tokenizer,
    device,
    system_instruction,
    prompt,
    temperature=0.1,
    do_sample=True,
    num_return_sequences=8,
    max_new_tokens=128,
):
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
    think_end_id = 151668

    def split_think_and_response(ids, end_id=think_end_id):
        if torch.is_tensor(ids):
            ids = ids.tolist()

        try:
            last_pos_from_end = ids[::-1].index(end_id)
            idx = len(ids) - 1 - last_pos_from_end
            resp_ids = ids[idx + 1:]
        except ValueError:
            resp_ids = ids

        response = tokenizer.decode(resp_ids, skip_special_tokens=True).strip("\n")
        return response

    responses = [split_think_and_response(ids) for ids in generated_ids]
    if num_return_sequences == 1:
        return responses[0]
    return responses


def build_prompt(sample, phase, column_name, user_instruction):
    if phase == "translate":
        question = get_question_text(sample, column_name)
        return f"{user_instruction}\n{str(question)}"
    if phase == "plan":
        question_symbolic = sample.get('question_symbolic', '')
        return f"{user_instruction}\n{str(question_symbolic)}"
    if phase == "solve":
        question_symbolic = str(sample.get('question_symbolic', ''))
        plan = str(sample.get('plan', ''))
        return user_instruction.replace('[[CONTEXT]]', question_symbolic).replace('[[PLAN]]', plan)
    raise ValueError(f"Unsupported phase: {phase}")


def stage_output_name(save_name, phase):
    root, ext = os.path.splitext(save_name)
    ext = ext or ".jsonl"
    return f"{root}_{phase}{ext}"


def stage_output_path(save_dir, save_name, phase):
    return os.path.join(save_dir, stage_output_name(save_name, phase))


def make_record_key(record_id, question, num_edits_preserving_label):
    payload = json.dumps(
        {
            "id": record_id,
            "question": question,
            "num_edits_preserving_label": num_edits_preserving_label,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_existing_outputs(save_path):
    records = []
    existing_keys = set()

    if not os.path.exists(save_path):
        return records, existing_keys

    with open(save_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            record_key = record.get('key')
            if not record_key:
                record_key = make_record_key(
                    record.get('id'),
                    record.get('question', ''),
                    record.get('num_edits_preserving_label', 0),
                )
                record['key'] = record_key

            existing_keys.add(record_key)
            records.append(record)

    return records, existing_keys


def predecessor_phase(phase):
    if phase == "plan":
        return "translate"
    if phase == "solve":
        return "plan"
    return None


def load_predecessor_samples(args, phase):
    prev_phase = predecessor_phase(phase)
    if prev_phase is None:
        return load_samples(args.dataset_path)

    prev_path = stage_output_path(args.save_dir, args.save_name, prev_phase)
    if not os.path.exists(prev_path):
        raise FileNotFoundError(
            f"Expected {prev_phase} output at {prev_path} before running phase '{phase}'."
        )
    return load_samples(prev_path)


def run_phase(samples, args, device, tokenizer, model, prompts, phase, save_name=None):
    system_instruction, user_instruction = get_instructions(phase, prompts)

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = stage_output_path(args.save_dir, save_name or args.save_name, phase)
    outputs, existing_keys = load_existing_outputs(save_path)

    with open(save_path, 'w', encoding='utf-8') as f:
        for record in outputs:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

        for sample in tqdm(samples):
            question_text = sample.get('question', get_question_text(sample, args.column_name))
            record_key = make_record_key(
                sample.get('id'),
                question_text,
                sample.get('num_edits_preserving_label', 0),
            )
            if record_key in existing_keys:
                continue

            prompt = build_prompt(sample, phase, args.column_name, user_instruction)
            reason_answer = evaluate(
                model,
                tokenizer,
                device,
                system_instruction,
                prompt,
                max_new_tokens=args.max_length,
                temperature=args.temperature,
                num_return_sequences=args.num_return_sequences,
                do_sample=args.do_sample,
            )
            primary_output = reason_answer[0] if isinstance(reason_answer, list) else reason_answer

            output = {
                'key': record_key,
                'id': sample.get('id'),
                'num_edits_preserving_label': sample.get('num_edits_preserving_label', 0),
                'question': question_text,
                'question_symbolic': sample.get('question_symbolic', ''),
                'label': get_label(sample),
                'output': primary_output,
            }
            if isinstance(reason_answer, list):
                output['all_outputs'] = reason_answer

            f.write(json.dumps(output, ensure_ascii=False) + '\n')
            existing_keys.add(record_key)
            outputs.append(output)

    return outputs


def run_all_stages(samples, args, device, tokenizer, model, prompts):
    translate_outputs = run_phase(
        samples,
        args,
        device,
        tokenizer,
        model,
        prompts,
        phase="translate",
    )

    df_translate = pd.DataFrame(translate_outputs)
    df_symbolic = parse_json(df_translate.copy(), "output")
    df_symbolic["question_symbolic"] = df_symbolic.apply(format_question_symbolic, axis=1)

    plan_outputs = run_phase(
        df_symbolic.to_dict(orient='records'),
        args,
        device,
        tokenizer,
        model,
        prompts,
        phase="plan",
    )

    df_plan = pd.DataFrame(plan_outputs)
    df_plan['plan'] = df_plan['output'].apply(
        lambda x: x.replace('Plan:\n', '', 1) if isinstance(x, str) else x
    )

    solve_inputs = df_symbolic.merge(
        df_plan[['id', 'plan']],
        on='id',
        how='left',
    )

    return run_phase(
        solve_inputs.to_dict(orient='records'),
        args,
        device,
        tokenizer,
        model,
        prompts,
        phase="solve",
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, required=True, help="Path or name of the base model")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to dataset")
    parser.add_argument('--save_dir', type=str, required=True, help="Directory to save the output")
    parser.add_argument('--save_name', type=str, required=True, help="File to save the output")
    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help="Dataset name used to select the symbolic CoT prompt bundle JSON.",
    )
    parser.add_argument(
        '--prompt_dir',
        type=Path,
        default=PROMPT_DIR,
        help="Directory containing symbolic CoT prompt bundle files.",
    )
    parser.add_argument('--load_8bit', action='store_true', help="Use 8-bit loading")
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--column_name', type=str, default='question', help="Column used as the question input")
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--cache_dir', type=str, default="")
    parser.add_argument('--do_sample', action='store_true', help="Inference sampling")
    parser.add_argument('--num_return_sequences', type=int, default=1)
    parser.add_argument('--phase', type=str, choices=['translate', 'plan', 'solve'], default='solve')
    parser.add_argument('--run_all_stages', action='store_true', help="Run translate, plan, and solve sequentially")
    parser.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    args.prompt_dir = args.prompt_dir.resolve()
    data_name = infer_data_name(args)
    prompts = load_prompt_bundle(data_name, args.prompt_dir)
    LOGGER.info("Loaded prompts from %s", args.prompt_dir / f"{data_name}-symboliccot_reasoning.txt")
    device, tokenizer, model = build_model_and_tokenizer(args)

    if args.run_all_stages:
        raw_samples = load_samples(args.dataset_path)
        run_all_stages(raw_samples, args, device, tokenizer, model, prompts)
    else:
        raw_samples = load_predecessor_samples(args, args.phase)
        samples = prepare_samples_for_phase(raw_samples, args.phase)
        run_phase(samples, args, device, tokenizer, model, prompts, phase=args.phase)


if __name__ == "__main__":
    main()
