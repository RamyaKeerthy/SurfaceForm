#!/usr/bin/env python3
"""
Unified runner for logic-program alteration scripts.

Modes:
- edit_type: 'cumulative' -> greedy cumulative preserving edits
- edit_type: 'single' -> individual edits; controlled by final_answer_preservation:
    - True -> preserve final answer (like logic-program_alteration_noalter-individual.py)
    - False -> change final prediction (like logic-program_alteration_alter-individual.py)

--save-edits 0 disables including mutated logic programs in the output (sets mutated_logic_program to None).
"""

import json
import re
import argparse
from copy import deepcopy
from typing import Dict, List, Optional, Tuple
from cspsolver.csp_solver import CSP_Program


CONSTRAINTS_HEADER_RE = re.compile(r"\nConstraints:\n", re.MULTILINE)
QUERY_HEADER_RE = re.compile(r"\n\nQuery:\n", re.MULTILINE)

# Match operators carefully (avoid >=, <=, =>, etc.)
OP_PATTERNS = {
    "==": re.compile(r"=="),
    "!=": re.compile(r"!="),
    ">": re.compile(r"(?<![<>=!])>(?![=])"),
    "<": re.compile(r"(?<![<>=!])<(?![=])"),
}

SWAPS = {
    "==": ("!=", "EqualToNotEqual"),
    "!=": ("==", "NoEqualToEqual"),
    ">": ("<", "GreatToLess"),
    "<": (">", "LessToGreat"),
}


def split_logic_program_sections(logic_program: str) -> Tuple[str, str, str]:
    m1 = CONSTRAINTS_HEADER_RE.search(logic_program)
    if not m1:
        raise ValueError("Missing 'Constraints:' section")

    m2 = QUERY_HEADER_RE.search(logic_program)
    if not m2 or m2.start() <= m1.end():
        raise ValueError("Missing or malformed 'Query:' section")

    prefix = logic_program[: m1.end()]
    constraints_block = logic_program[m1.end() : m2.start()]
    suffix = logic_program[m2.start() :]
    return prefix, constraints_block, suffix


def parse_constraints_lines(constraints_block: str) -> List[str]:
    lines = constraints_block.splitlines()
    return [ln for ln in lines if ln.strip()]


def has_mutatable_operator(constraint_line: str) -> bool:
    if ":::" not in constraint_line:
        return False
    _, expr = constraint_line.split(":::", 1)
    expr = expr.strip()
    return any(p.search(expr) for p in OP_PATTERNS.values())


def mutate_constraint_once(constraint_line: str, op: str) -> Optional[Tuple[str, str, str]]:
    if ":::" not in constraint_line:
        return None
    left, expr = constraint_line.split(":::", 1)
    expr_str = expr.strip()

    pattern = OP_PATTERNS[op]
    if not pattern.search(expr_str):
        return None

    new_op, transition_type = SWAPS[op]
    mutated_expr = pattern.sub(new_op, expr_str, count=1)

    mutated_line = f"{left}::: {mutated_expr}"
    operator_name = f"{op}->{new_op}"
    return mutated_line, transition_type, operator_name


def replace_constraint_in_logic_program(logic_program: str, constraint_idx: int, new_constraint_line: str) -> str:
    prefix, constraints_block, suffix = split_logic_program_sections(logic_program)
    lines = constraints_block.splitlines()

    nonempty_positions = [i for i, ln in enumerate(lines) if ln.strip()]
    if constraint_idx < 0 or constraint_idx >= len(nonempty_positions):
        raise IndexError("constraint_idx out of range")

    target_pos = nonempty_positions[constraint_idx]
    lines[target_pos] = new_constraint_line

    new_constraints_block = "\n".join(lines)
    return prefix + new_constraints_block + suffix


def solve_prediction(logic_program: str) -> Tuple[str, int]:
    csp_program = CSP_Program(logic_program, "LogicalDeduction")
    ans = csp_program.execute_program()
    if ans:
        pred, ans_len = csp_program.answer_mapping(ans)
    else:
        pred, ans_len = "", 0
    return str(pred), ans_len


# ----- Finders -----
def find_next_preserving_edit(logic_program: str, gold_label: str, edited_constraint_indices: set) -> Optional[Dict]:
    prefix, constraints_block, suffix = split_logic_program_sections(logic_program)
    constraints = parse_constraints_lines(constraints_block)

    for idx, line in enumerate(constraints):
        if idx in edited_constraint_indices:
            continue
        if not has_mutatable_operator(line):
            continue

        for op in ("==", "!=", ">", "<"):
            mutated = mutate_constraint_once(line, op)
            if not mutated:
                continue
            mutated_line, transition_type, operator_name = mutated

            mutated_lp = replace_constraint_in_logic_program(logic_program, idx, mutated_line)

            try:
                pred, complexity = solve_prediction(mutated_lp)
            except Exception:
                continue

            if str(pred) == str(gold_label):
                return {
                    "constraint_idx": idx,
                    "original_constraint": line,
                    "mutated_constraint": mutated_line,
                    "mutated_logic_program": mutated_lp,
                    "transition_type": transition_type,
                    "operator_name": operator_name,
                    "new_label": pred,
                    "complexity": complexity,
                }

    return None


def find_preserving_edits(logic_program: str, gold_label: str) -> List[Dict]:
    _, constraints_block, _ = split_logic_program_sections(logic_program)
    constraints = parse_constraints_lines(constraints_block)
    preserving_edits: List[Dict] = []

    for idx, line in enumerate(constraints):
        if not has_mutatable_operator(line):
            continue

        for op in ("==", "!=", ">", "<"):
            mutated = mutate_constraint_once(line, op)
            if not mutated:
                continue
            mutated_line, transition_type, operator_name = mutated

            mutated_lp = replace_constraint_in_logic_program(logic_program, idx, mutated_line)

            try:
                pred, complexity = solve_prediction(mutated_lp)
            except Exception:
                continue

            if str(pred) == str(gold_label):
                preserving_edits.append(
                    {
                        "constraint_idx": idx,
                        "original_constraint": line,
                        "mutated_constraint": mutated_line,
                        "mutated_logic_program": mutated_lp,
                        "transition_type": transition_type,
                        "operator_name": operator_name,
                        "new_label": pred,
                        "complexity": complexity,
                    }
                )

    return preserving_edits


def find_prediction_changing_edits(logic_program: str, base_prediction: str) -> List[Dict]:
    _, constraints_block, _ = split_logic_program_sections(logic_program)
    constraints = parse_constraints_lines(constraints_block)
    prediction_changing_edits: List[Dict] = []

    for idx, line in enumerate(constraints):
        if not has_mutatable_operator(line):
            continue

        for op in ("==", "!=", ">", "<"):
            mutated = mutate_constraint_once(line, op)
            if not mutated:
                continue
            mutated_line, transition_type, operator_name = mutated

            mutated_lp = replace_constraint_in_logic_program(logic_program, idx, mutated_line)

            try:
                pred, complexity = solve_prediction(mutated_lp)
            except Exception:
                continue

            if str(pred) != str(base_prediction):
                prediction_changing_edits.append(
                    {
                        "constraint_idx": idx,
                        "original_constraint": line,
                        "mutated_constraint": mutated_line,
                        "mutated_logic_program": mutated_lp,
                        "transition_type": transition_type,
                        "operator_name": operator_name,
                        "new_prediction": pred,
                        "complexity": complexity,
                    }
                )

    return prediction_changing_edits


def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e
    return rows


def write_jsonl(path: str, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def process_record(row: Dict, edit_type: str, final_answer_preservation: bool, save_edits: int) -> List[Dict]:
    # Dispatch to the appropriate behavior matching the original scripts
    if edit_type == "cumulative":
        # Use the greedy cumulative preserving edits behavior (from logic-program_alteration_noalter.py)
        base_out = {"id": row.get("id"), "state": row.get("state"), "label": row.get("label")}

        if row.get("state") != "Perfect":
            out = deepcopy(base_out)
            out.update(
                {
                    "edit_step": 0,
                    "new_label": [],
                    "complexity": [],
                    "base_complexity": 0,
                    "num_edits_preserving_label": 0,
                    "transition_type": "Skipped",
                    "original_logic_constraint": [],
                    "mutated_logic_constraint": [],
                    "operator_name": None,
                    "constraint_idx": [],
                    "question": row.get("question"),
                    "logic_program": row.get("logic_program"),
                    "mutated_logic_program": None,
                }
            )
            return [out]

        logic_program = row.get("logic_program", "")
        question = row.get("question")
        gold_label = row.get("label")

        try:
            base_pred, base_comp = solve_prediction(logic_program)
        except Exception:
            out = deepcopy(base_out)
            out.update(
                {
                    "edit_step": 0,
                    "new_label": [],
                    "complexity": [],
                    "base_complexity": 0,
                    "num_edits_preserving_label": 0,
                    "transition_type": "IncorrectFormat",
                    "original_logic_constraint": [],
                    "mutated_logic_constraint": [],
                    "operator_name": None,
                    "constraint_idx": [],
                    "question": question,
                    "logic_program": logic_program,
                    "mutated_logic_program": None,
                }
            )
            return [out]

        if str(base_pred) != str(gold_label):
            out = deepcopy(base_out)
            out.update(
                {
                    "edit_step": 0,
                    "new_label": [],
                    "complexity": [],
                    "base_complexity": base_comp,
                    "num_edits_preserving_label": 0,
                    "transition_type": "Skipped",
                    "original_logic_constraint": [],
                    "mutated_logic_constraint": [],
                    "operator_name": None,
                    "constraint_idx": [],
                    "question": question,
                    "logic_program": logic_program,
                    "mutated_logic_program": None,
                }
            )
            return [out]

        outputs: List[Dict] = []

        out0 = deepcopy(base_out)
        out0.update(
            {
                "edit_step": 0,
                "new_label": [],
                "complexity": [],
                "base_complexity": base_comp,
                "num_edits_preserving_label": 0,
                "transition_type": None,
                "original_logic_constraint": [],
                "mutated_logic_constraint": [],
                "operator_name": None,
                "constraint_idx": [],
                "question": question,
                "logic_program": logic_program,
                "mutated_logic_program": None,
            }
        )
        outputs.append(out0)

        edited_indices = set()
        current_lp = logic_program
        edit_step = 0

        history_original_constraints: List[str] = []
        history_mutated_constraints: List[str] = []
        history_new_labels: List[str] = []
        history_complexity: List[str] = []
        history_constraint_idx: List[int] = []
        history_operator_names: List[str] = []
        history_transition_types: List[str] = []

        while True:
            found = find_next_preserving_edit(logic_program=current_lp, gold_label=gold_label, edited_constraint_indices=edited_indices)
            if not found:
                break

            edit_step += 1
            edited_indices.add(found["constraint_idx"])

            history_original_constraints.append(found["original_constraint"])
            history_mutated_constraints.append(found["mutated_constraint"])
            history_new_labels.append(found["new_label"])
            history_complexity.append(found["complexity"])
            history_constraint_idx.append(found["constraint_idx"])
            history_operator_names.append(found["operator_name"])
            history_transition_types.append(found["transition_type"])

            out = deepcopy(base_out)
            out.update(
                {
                    "edit_step": edit_step,
                    "new_label": deepcopy(history_new_labels),
                    "complexity": deepcopy(history_complexity),
                    "base_complexity": base_comp,
                    "num_edits_preserving_label": edit_step,
                    "transition_type": deepcopy(history_transition_types),
                    "original_logic_constraint": deepcopy(history_original_constraints),
                    "mutated_logic_constraint": deepcopy(history_mutated_constraints),
                    "operator_name": deepcopy(history_operator_names),
                    "constraint_idx": deepcopy(history_constraint_idx),
                    "question": question,
                    "logic_program": logic_program,
                    "mutated_logic_program": found["mutated_logic_program"] if save_edits else None,
                }
            )
            outputs.append(out)

            current_lp = found["mutated_logic_program"]

        if not outputs:
            out = deepcopy(base_out)
            out.update(
                {
                    "edit_step": 0,
                    "new_label": [],
                    "complexity": [],
                    "base_complexity": base_comp,
                    "num_edits_preserving_label": 0,
                    "transition_type": None,
                    "original_logic_constraint": [],
                    "mutated_logic_constraint": [],
                    "operator_name": None,
                    "constraint_idx": [],
                    "question": question,
                    "logic_program": logic_program,
                    "mutated_logic_program": None,
                }
            )
            return [out]

        return outputs

    elif edit_type == "single":
        # Single-edit mode: choose preserving vs changing via final_answer_preservation
        if row.get("state") != "Perfect":
            # original 'individual' scripts ignored non-Perfect rows (return nothing)
            return []

        logic_program = row.get("logic_program", "")
        question = row.get("question")
        gold_label = row.get("label")

        try:
            base_pred, base_comp = solve_prediction(logic_program)
        except Exception:
            return []

        if final_answer_preservation:
            # preserving edits (like noalter-individual)
            if str(base_pred) != str(gold_label):
                return []

            outputs: List[Dict] = []
            preserving_edits = find_preserving_edits(logic_program=logic_program, gold_label=gold_label)

            for edit_step, found in enumerate(preserving_edits, start=1):
                out = {
                    "id": row.get("id"),
                    "state": row.get("state"),
                    "label": row.get("label"),
                    "edit_step": edit_step,
                    "new_label": [found["new_label"]],
                    "complexity": [found["complexity"]],
                    "base_complexity": base_comp,
                    "num_edits_preserving_label": 1,
                    "transition_type": [found["transition_type"]],
                    "original_logic_constraint": [found["original_constraint"]],
                    "mutated_logic_constraint": [found["mutated_constraint"]],
                    "operator_name": [found["operator_name"]],
                    "constraint_idx": [found["constraint_idx"]],
                    "question": question,
                    "logic_program": logic_program,
                    "mutated_logic_program": found["mutated_logic_program"] if save_edits else None,
                }
                outputs.append(out)

            return outputs

        else:
            # prediction-changing edits (like alter-individual)
            outputs: List[Dict] = []
            prediction_changing_edits = find_prediction_changing_edits(logic_program=logic_program, base_prediction=base_pred)

            for edit_step, found in enumerate(prediction_changing_edits, start=1):
                out = {
                    "id": row.get("id"),
                    "state": row.get("state"),
                    "label": row.get("label"),
                    "edit_step": edit_step,
                    "base_prediction": base_pred,
                    "new_prediction": found["new_prediction"],
                    "complexity": [found["complexity"]],
                    "base_complexity": base_comp,
                    "num_edits_changing_prediction": 1,
                    "transition_type": [found["transition_type"]],
                    "original_logic_constraint": [found["original_constraint"]],
                    "mutated_logic_constraint": [found["mutated_constraint"]],
                    "operator_name": [found["operator_name"]],
                    "constraint_idx": [found["constraint_idx"]],
                    "question": question,
                    "logic_program": logic_program,
                    "mutated_logic_program": found["mutated_logic_program"] if save_edits else None,
                }
                outputs.append(out)

            return outputs

    else:
        raise ValueError(f"Unknown edit_type: {edit_type}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input JSONL")
    ap.add_argument("--output", required=True, help="Path to output JSONL")
    ap.add_argument("--edit-type", choices=["cumulative", "single"], default="cumulative", help="Type of edit to run")
    ap.add_argument("--final-answer-preservation", default="False", help="True|False (only used with --edit-type single)")
    ap.add_argument("--save-edits", type=int, choices=[0, 1], default=0, help="0 to avoid saving mutated logic programs, 1 to include them")
    args = ap.parse_args()

    edit_type = args.edit_type
    final_answer_preservation = str(args.final_answer_preservation).lower() in ("true", "1", "t", "yes", "y")
    if edit_type != "single" and final_answer_preservation:
        # warn and ignore
        print("Warning: --final-answer-preservation is only applicable for --edit-type single; ignoring.")
        final_answer_preservation = False

    save_edits = int(args.save_edits)

    in_rows = read_jsonl(args.input)
    out_rows: List[Dict] = []

    for row in in_rows:
        out_rows.extend(process_record(row, edit_type=edit_type, final_answer_preservation=final_answer_preservation, save_edits=save_edits))

    write_jsonl(args.output, out_rows)


if __name__ == "__main__":
    main()
