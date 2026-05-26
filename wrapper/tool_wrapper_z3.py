#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from z3solver.sat_problem_solver import LSAT_Z3_Program


REPLACEMENTS = [
    ("AND", "OR"),
    ("XOR", "OR"),
    ("OR", "XOR"),
    ("NOT", ""),
    ("And", "Or"),
    ("Xor", "Or"),
    ("Or", "Xor"),
    ("Not", ""),
    ("and", "or"),
    ("xor", "or"),
    ("or", "xor"),
    ("not", ""),
]


def run_solver(logic_program: str) -> Tuple[str, str]:
    try:
        z3_program = LSAT_Z3_Program(logic_program, "AR-LSAT")
        ans, error_message = z3_program.execute_program()
        if ans and len(ans) == 1:
            prediction = z3_program.answer_mapping(ans)
        else:
            prediction = ""
        return prediction, error_message or ""
    except Exception:
        return "", traceback.format_exc()


def read_input_table(input_path: Path) -> List[Dict]:
    suffix = input_path.suffix.lower()

    if suffix == ".jsonl":
        rows = []
        with input_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSONL at line {line_no}: {e}") from e
        return rows

    if suffix == ".json":
        with input_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON input must be a list of objects.")
        return data

    if suffix == ".csv":
        with input_path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    raise ValueError(f"Unsupported input format: {suffix}. Use .jsonl, .json, or .csv")


def write_output_table(rows: List[Dict], output_path: Path) -> None:
    suffix = output_path.suffix.lower()

    if suffix == ".jsonl":
        with output_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return

    if suffix == ".csv":
        fieldnames = sorted({k for row in rows for k in row.keys()}) if rows else []
        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return

    raise ValueError(f"Unsupported output format: {suffix}. Use .jsonl or .csv")


def split_logic_program_sections(logic_program: str) -> Tuple[str, str, str]:
    pattern = re.compile(
        r"(?s)^\s*#\s*Declarations\s*\n(?P<decl>.*?)\n\s*#\s*Constraints\s*\n(?P<constraints>.*?)\n\s*#\s*Options\s*\n(?P<options>.*)\s*$"
    )
    m = pattern.match(logic_program)
    if not m:
        raise ValueError(
            "logic_program does not match expected '# Declarations / # Constraints / # Options' structure."
        )
    return m.group("decl"), m.group("constraints"), m.group("options")


def rebuild_logic_program(declarations: str, constraints: str, options: str) -> str:
    return f"# Declarations\n{declarations}\n\n# Constraints\n{constraints}\n\n# Options\n{options}"


def split_constraint_line(line: str) -> Optional[Tuple[str, str]]:
    if ":::" not in line:
        return None
    lhs, rhs = line.split(":::", 1)
    return lhs.rstrip(), rhs.lstrip()


def make_single_occurrence_replacement_candidates(expr: str) -> List[Dict]:
    candidates = []

    for old_op, new_op in REPLACEMENTS:
        pattern = re.compile(rf"\b{re.escape(old_op)}\b")
        matches = list(pattern.finditer(expr))

        for occ_idx, match in enumerate(matches, start=1):
            start, end = match.span()
            edited_expr = expr[:start] + new_op + expr[end:]

            candidates.append(
                {
                    "original_operator": old_op,
                    "new_operator": new_op,
                    "operator_occurrence_in_constraint": occ_idx,
                    "edited_expr": edited_expr,
                }
            )

    return candidates


def validate_required_columns(row: Dict) -> None:
    required = {"id", "question", "label", "logic_program", "state"}
    missing = required - set(row.keys())
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def normalize_label(label) -> str:
    if label is None:
        return ""
    label = str(label).strip()
    label = label.strip("()[]{}").strip()
    return label.upper()


def process_record_cumulative(row: Dict) -> List[Dict]:
    validate_required_columns(row)
    if str(row["state"]).strip() != "Perfect":
        return []

    declarations, constraints_text, options = split_logic_program_sections(row["logic_program"])
    current_constraint_lines = constraints_text.splitlines()
    successful_rows: List[Dict] = []
    num_edits = 0

    for constraint_idx in range(len(current_constraint_lines)):
        raw_line = current_constraint_lines[constraint_idx]
        parsed = split_constraint_line(raw_line)
        if not parsed:
            continue

        lhs_text, rhs_expr = parsed
        candidates = make_single_occurrence_replacement_candidates(rhs_expr)

        for cand in candidates:
            if cand["edited_expr"] == rhs_expr:
                continue

            trial_constraint_lines = deepcopy(current_constraint_lines)
            trial_constraint_lines[constraint_idx] = f"{lhs_text} ::: {cand['edited_expr']}"
            trial_constraints_text = "\n".join(trial_constraint_lines)
            trial_logic_program = rebuild_logic_program(declarations, trial_constraints_text, options)

            prediction, error_message = run_solver(trial_logic_program)

            if prediction == row["label"]:
                num_edits += 1
                current_constraint_lines = trial_constraint_lines

                successful_rows.append(
                    {
                        "edited_id": f"{row['id']}__edit_{num_edits}",
                        "source_id": row["id"],
                        "question": row["question"],
                        "label": row["label"],
                        "state": row["state"],
                        "num_edits_no_alteration": num_edits,
                        "edited_constraint_index": constraint_idx,
                        "edited_constraint_text": lhs_text,
                        "edited_constraint_natural_language": lhs_text,
                        "formal_logic_before_edit": rhs_expr,
                        "formal_logic_after_edit": cand["edited_expr"],
                        "original_operator": cand["original_operator"],
                        "new_operator": cand["new_operator"],
                        "operator_occurrence_in_constraint": cand["operator_occurrence_in_constraint"],
                        "prediction_after_edit": prediction,
                        "error_message": error_message,
                        "logic_program": trial_logic_program,
                    }
                )
                break

    return successful_rows


def process_record_single(row: Dict, final_answer_preservation: bool) -> List[Dict]:
    validate_required_columns(row)
    if str(row["state"]).strip() != "Perfect":
        return []

    declarations, constraints_text, options = split_logic_program_sections(row["logic_program"])
    original_constraint_lines = constraints_text.splitlines()
    successful_rows: List[Dict] = []
    num_successes = 0
    original_label = normalize_label(row["label"])

    for constraint_idx in range(len(original_constraint_lines)):
        raw_line = original_constraint_lines[constraint_idx]
        parsed = split_constraint_line(raw_line)
        if not parsed:
            continue

        lhs_text, rhs_expr = parsed
        candidates = make_single_occurrence_replacement_candidates(rhs_expr)

        for cand in candidates:
            if cand["edited_expr"] == rhs_expr:
                continue

            trial_constraint_lines = deepcopy(original_constraint_lines)
            trial_constraint_lines[constraint_idx] = f"{lhs_text} ::: {cand['edited_expr']}"
            trial_constraints_text = "\n".join(trial_constraint_lines)
            trial_logic_program = rebuild_logic_program(declarations, trial_constraints_text, options)

            prediction, error_message = run_solver(trial_logic_program)

            if final_answer_preservation:
                if prediction != row["label"]:
                    continue

                num_successes += 1
                successful_rows.append(
                    {
                        "edited_id": f"{row['id']}__edit_{num_successes}",
                        "source_id": row["id"],
                        "question": row["question"],
                        "label": row["label"],
                        "state": row["state"],
                        "num_edits_no_alteration": 1,
                        "edited_constraint_index": constraint_idx,
                        "edited_constraint_text": lhs_text,
                        "edited_constraint_natural_language": lhs_text,
                        "formal_logic_before_edit": rhs_expr,
                        "formal_logic_after_edit": cand["edited_expr"],
                        "original_operator": cand["original_operator"],
                        "new_operator": cand["new_operator"],
                        "operator_occurrence_in_constraint": cand["operator_occurrence_in_constraint"],
                        "prediction_after_edit": prediction,
                        "error_message": error_message,
                        "logic_program": trial_logic_program,
                    }
                )
            else:
                changed_label = normalize_label(prediction)
                if not changed_label or changed_label == original_label:
                    continue

                num_successes += 1
                successful_rows.append(
                    {
                        "edited_id": f"{row['id']}__edit_{num_successes}",
                        "source_id": row["id"],
                        "label": row["label"],
                        "new_label": changed_label,
                        "question": row["question"],
                        "state": row["state"],
                        "num_edits_label_changed": 1,
                        "edited_constraint_index": constraint_idx,
                        "edited_constraint_text": lhs_text,
                        "edited_constraint_natural_language": lhs_text,
                        "formal_logic_before_edit": rhs_expr,
                        "formal_logic_after_edit": cand["edited_expr"],
                        "original_operator": cand["original_operator"],
                        "new_operator": cand["new_operator"],
                        "operator_occurrence_in_constraint": cand["operator_occurrence_in_constraint"],
                        "error_message": error_message,
                        "logic_program": trial_logic_program,
                    }
                )

    return successful_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified Z3 edit driver for cumulative and single edits."
    )
    parser.add_argument("--input", required=True, help="Input file: .jsonl, .json, or .csv")
    parser.add_argument("--output", required=True, help="Output file: .jsonl or .csv")
    parser.add_argument(
        "--edit-type",
        choices=["cumulative", "single"],
        default="single",
        help="Type of edit search to apply.",
    )
    parser.add_argument(
        "--final-answer-preservation",
        action="store_true",
        help="Only for single edit mode: preserve the original final answer label.",
    )
    parser.add_argument(
        "--save-edits",
        type=int,
        choices=[0, 1],
        default=1,
        help="Set to 0 to process without writing edited rows to the output file.",
    )
    args = parser.parse_args()

    if args.edit_type == "cumulative" and args.final_answer_preservation:
        sys.stderr.write(
            "[WARN] --final-answer-preservation is ignored when --edit-type=cumulative\n"
        )

    input_path = Path(args.input)
    output_path = Path(args.output)

    rows = read_input_table(input_path)

    output_rows: List[Dict] = []
    total_rows = 0
    perfect_rows = 0
    saved_rows = 0

    for row in rows:
        total_rows += 1
        if str(row.get("state", "")).strip() == "Perfect":
            perfect_rows += 1

        try:
            if args.edit_type == "cumulative":
                edited_versions = process_record_cumulative(row)
            else:
                edited_versions = process_record_single(row, args.final_answer_preservation)

            saved_rows += len(edited_versions)
            if args.save_edits:
                output_rows.extend(edited_versions)
        except Exception as e:
            sys.stderr.write(f"[WARN] Failed row id={row.get('id', '<unknown>')}: {e}\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.save_edits:
        write_output_table(output_rows, output_path)
    else:
        write_output_table([], output_path)

    print(f"Total rows read: {total_rows}")
    print(f"Perfect rows processed: {perfect_rows}")
    print(f"Edited rows found: {saved_rows}")
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()
