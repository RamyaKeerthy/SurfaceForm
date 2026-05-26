#!/usr/bin/env python3
"""
Unified runner for Prover9 FOL edit scripts.

Behaviors:
- --edit-type cumulative : behave like `fol_noalteration.py` (one output per input, pick best flip/first change)
- --edit-type single     : behave like the individual scripts; use `--final-answer-preservation` to choose
    - True  -> label-preserving single edits (fol_individual_noalter.py)
    - False -> label-changing single edits (fol_individual_with-alter.py)

--save-edits 0 will omit mutated premise programs/translations from outputs.
"""

import json
import re
import os
import argparse
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from fol_solver.prover9_solver import FOL_Prover9_Program

os.environ['PROVER9'] = '../LADR-2009-11A/bin'

def load_json_any(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)
        items = []
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
        return items


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    kind: str
    pattern: str
    repl: str


def find_operator_spans(formula: str, op: OperatorSpec) -> List[Tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(op.pattern, formula)]


def replace_nth_occurrence(formula: str, op: OperatorSpec, n: int) -> Optional[str]:
    spans = find_operator_spans(formula, op)
    if n < 0 or n >= len(spans):
        return None
    start, end = spans[n]
    return formula[:start] + op.repl + formula[end:]


# Label normalization helpers
DATASET_TO_CANON = {"TRUE": "TRUE", "FALSE": "FALSE", "UNCERTAIN": "UNKNOWN"}
CANON_TO_DATASET = {"TRUE": "A", "FALSE": "B", "UNKNOWN": "C"}


def normalize_dataset_label(label: Any) -> str:
    if label is None:
        return "UNKNOWN"
    s = str(label).strip().upper()
    return DATASET_TO_CANON.get(s, "UNKNOWN")


def prover9_answer_to_canon(answer: Any, error_message: Optional[str] = None) -> str:
    if error_message:
        return "ERROR"
    if answer is None:
        return "UNKNOWN"
    s = str(answer).strip().lower()
    if s == "true":
        return "TRUE"
    if s == "false":
        return "FALSE"
    if s in {"uncertain", "unknown"}:
        return "UNKNOWN"
    return "UNKNOWN"


def run_prover9(premise_program: List[str], conclusion_program: str) -> Tuple[Any, Optional[str], str]:
    prover9_program = FOL_Prover9_Program(premise_program, conclusion_program)
    answer, error_message = prover9_program.execute_program()
    canon_label = prover9_answer_to_canon(answer, error_message)
    return answer, error_message, canon_label


# ------------------
# Single-edit finders
# ------------------
def find_label_preserving_single_edits(
    premises_translations: List[str],
    conclusion_translation: str,
    original_label: str,
    operators_to_test: List[OperatorSpec],
) -> List[Dict[str, Any]]:
    original_premises = deepcopy(premises_translations)
    mutations: List[Dict[str, Any]] = []

    for premise_idx, prem_text in enumerate(original_premises):
        for op in operators_to_test:
            spans = find_operator_spans(prem_text, op)
            if not spans:
                continue

            for occ_idx in range(len(spans)):
                mutated = replace_nth_occurrence(prem_text, op, occ_idx)
                if mutated is None:
                    continue

                trial_premises = deepcopy(original_premises)
                trial_premises[premise_idx] = mutated

                raw_answer, error_message, new_label = run_prover9(trial_premises, conclusion_translation)

                if new_label != original_label:
                    continue

                mutations.append(
                    {
                        "operator_name": op.name,
                        "operator_kind": op.kind,
                        "operator_pattern": op.pattern,
                        "operator_repl": op.repl,
                        "premise_idx": premise_idx,
                        "occurrence_idx": occ_idx,
                        "new_label": new_label,
                        "raw_answer": raw_answer,
                        "prover9_error": error_message,
                        "original_premise_translation": prem_text,
                        "mutated_premise_translation": mutated,
                        "mutated_premise_translations": trial_premises,
                    }
                )

    return mutations


def find_label_changing_single_edits(
    premises_translations: List[str],
    conclusion_translation: str,
    original_label: str,
    operators_to_test: List[OperatorSpec],
) -> List[Dict[str, Any]]:
    original_premises = deepcopy(premises_translations)
    mutations: List[Dict[str, Any]] = []

    for premise_idx, prem_text in enumerate(original_premises):
        for op in operators_to_test:
            spans = find_operator_spans(prem_text, op)
            if not spans:
                continue

            for occ_idx in range(len(spans)):
                mutated = replace_nth_occurrence(prem_text, op, occ_idx)
                if mutated is None:
                    continue

                trial_premises = deepcopy(original_premises)
                trial_premises[premise_idx] = mutated

                raw_answer, error_message, new_label = run_prover9(trial_premises, conclusion_translation)

                if new_label == original_label:
                    continue

                mutations.append(
                    {
                        "operator_name": op.name,
                        "operator_kind": op.kind,
                        "operator_pattern": op.pattern,
                        "operator_repl": op.repl,
                        "premise_idx": premise_idx,
                        "occurrence_idx": occ_idx,
                        "new_label": new_label,
                        "raw_answer": raw_answer,
                        "prover9_error": error_message,
                        "original_premise_translation": prem_text,
                        "mutated_premise_translation": mutated,
                        "mutated_premise_translations": trial_premises,
                    }
                )

    return mutations


# ------------------
# No-alteration (single output) logic
# ------------------
def try_all_single_edits(
    premises_translations: List[str],
    conclusion_translation: str,
    original_label: str,
    operators_to_test: List[OperatorSpec],
) -> Dict[str, Any]:
    best_flip: Optional[Dict[str, Any]] = None
    first_change: Optional[Dict[str, Any]] = None

    for op in operators_to_test:
        for premise_idx, prem_text in enumerate(premises_translations):
            spans = find_operator_spans(prem_text, op)
            if not spans:
                continue

            for occ_idx in range(len(spans)):
                mutated_prem_text = replace_nth_occurrence(prem_text, op, occ_idx)
                if mutated_prem_text is None:
                    continue

                mutated_premises = deepcopy(premises_translations)
                mutated_premises[premise_idx] = mutated_prem_text

                answer, err, new_label = run_prover9(mutated_premises, conclusion_translation)

                rec = {
                    "operator_name": op.name,
                    "operator_kind": op.kind,
                    "operator_pattern": op.pattern,
                    "operator_repl": op.repl,
                    "premise_idx": premise_idx,
                    "occurrence_idx": occ_idx,
                    "original_label": original_label,
                    "new_label": new_label,
                    "original_premise_translation": prem_text,
                    "mutated_premise_translation": mutated_prem_text,
                    "conclusion_translation": conclusion_translation,
                    "prover9_answer": str(answer),
                    "prover9_error": err,
                    "mutated_premise_program": mutated_premises,
                }

                if new_label != original_label and first_change is None:
                    first_change = rec

                if original_label in {"TRUE", "FALSE"} and new_label in {"TRUE", "FALSE"} and new_label != original_label:
                    if best_flip is None:
                        best_flip = rec

                if best_flip is not None:
                    return {"best_flip": best_flip, "first_change": first_change, "no_change": False}

    return {"best_flip": best_flip, "first_change": first_change, "no_change": first_change is None}


def select_noalteration_record(item: Dict[str, Any], operators_to_test: List[OperatorSpec], save_edits: int) -> Dict[str, Any]:
    id = item.get("id", "")
    premises = item.get("premise_translations", [])
    conclusion = item.get("conclusion_translation", "")
    original_dataset_label = str(item.get("label", "C")).strip().upper()
    original_canon = normalize_dataset_label(original_dataset_label)
    state = item.get("state")

    base_out = {
        "id": id,
        "state": state,
        "original_label": CANON_TO_DATASET.get(original_canon, "C"),
        "new_label": None,
        "premise_translations": premises,
        "conclusion_translation": conclusion,
        "transition_type": None,
        "operator_name": None,
        "operator_kind": None,
        "operator_pattern": None,
        "operator_repl": None,
        "premise_idx": None,
        "occurrence_idx": None,
        "original_premise_translation": None,
        "mutated_premise_translation": None,
        "prover9_answer": None,
        "prover9_error": None,
    }

    if state != "Perfect":
        base_out["transition_type"] = "Skipped"
        base_out["new_label"] = base_out["original_label"]
        return base_out

    if not isinstance(premises, list) or not isinstance(conclusion, str):
        base_out["transition_type"] = "IncorrectFormat"
        base_out["new_label"] = base_out["original_label"]
        base_out["prover9_error"] = "Invalid input fields"
        return base_out

    results = try_all_single_edits(premises, conclusion, original_canon, operators_to_test)
    best_flip = results["best_flip"]
    first_change = results["first_change"]
    no_change = results["no_change"]

    if original_canon in {"TRUE", "FALSE"}:
        chosen = best_flip or first_change
        if chosen is None:
            base_out["transition_type"] = "NoChange"
            base_out["new_label"] = base_out["original_label"]
            return base_out

        base_out.update({
            "transition_type": "Flip" if best_flip is not None else "ToUnknown",
            "operator_name": chosen["operator_name"],
            "operator_kind": chosen["operator_kind"],
            "operator_pattern": chosen["operator_pattern"],
            "operator_repl": chosen["operator_repl"],
            "premise_idx": chosen["premise_idx"],
            "occurrence_idx": chosen["occurrence_idx"],
            "original_premise_translation": chosen["original_premise_translation"],
            "mutated_premise_translation": (chosen["mutated_premise_translation"] if save_edits else None),
            "new_label": chosen["new_label"],
            "prover9_answer": chosen["prover9_answer"],
            "prover9_error": chosen["prover9_error"],
        })
        return base_out

    if original_canon == "UNKNOWN":
        if first_change is None:
            base_out["transition_type"] = "NoChange"
            base_out["new_label"] = base_out["original_label"]
            return base_out

        base_out.update({
            "transition_type": "FirstChange",
            "operator_name": first_change["operator_name"],
            "operator_kind": first_change["operator_kind"],
            "operator_pattern": first_change["operator_pattern"],
            "operator_repl": first_change["operator_repl"],
            "premise_idx": first_change["premise_idx"],
            "occurrence_idx": first_change["occurrence_idx"],
            "original_premise_translation": first_change["original_premise_translation"],
            "mutated_premise_translation": (first_change["mutated_premise_translation"] if save_edits else None),
            "new_label": first_change["new_label"],
            "prover9_answer": first_change["prover9_answer"],
            "prover9_error": first_change["prover9_error"],
        })
        return base_out

    if no_change or first_change is None:
        base_out["transition_type"] = "NoChange"
        base_out["new_label"] = base_out["original_label"]
        return base_out

    base_out.update({
        "transition_type": "FirstChange",
        "operator_name": first_change["operator_name"],
        "operator_kind": first_change["operator_kind"],
        "operator_pattern": first_change["operator_pattern"],
        "operator_repl": first_change["operator_repl"],
        "premise_idx": first_change["premise_idx"],
        "occurrence_idx": first_change["occurrence_idx"],
        "original_premise_translation": first_change["original_premise_translation"],
        "mutated_premise_translation": (first_change["mutated_premise_translation"] if save_edits else None),
        "new_label": first_change["new_label"],
        "prover9_answer": first_change["prover9_answer"],
        "prover9_error": first_change["prover9_error"],
    })
    return base_out


def process_record(item: Dict[str, Any], edit_type: str, final_answer_preservation: bool, operators_to_test: List[OperatorSpec], save_edits: int) -> List[Dict[str, Any]]:
    if edit_type == "cumulative":
        # map to fol_noalteration behavior: one output per input
        return [select_noalteration_record(item, operators_to_test, save_edits)]

    if edit_type == "single":
        state = item.get("state")
        if state != "Perfect":
            return [] if True else []

        premises = item.get("premise_translations", [])
        conclusion = item.get("conclusion_translation", "")
        original_dataset_label = str(item.get("label", "")).strip().upper()
        original_canon = normalize_dataset_label(original_dataset_label)

        try:
            # quick sanity check via Prover9
            # run_prover9 expects premise list and conclusion
            _ans, _err, base_label = run_prover9(premises, conclusion)
        except Exception:
            return []

        outputs: List[Dict[str, Any]] = []
        if final_answer_preservation:
            if str(base_label) != str(original_canon):
                return []
            edits = find_label_preserving_single_edits(premises, conclusion, original_canon, operators_to_test)
            for edit_step, e in enumerate(edits, start=1):
                rec = {
                    "id": item.get("id"),
                    "state": item.get("state"),
                    "label": original_canon,
                    "edit_step": edit_step,
                    "new_label": e["new_label"],
                    "raw_answer": e.get("raw_answer"),
                    "prover9_error": e.get("prover9_error"),
                    "operator_name": e["operator_name"],
                    "operator_kind": e["operator_kind"],
                    "operator_pattern": e["operator_pattern"],
                    "operator_repl": e["operator_repl"],
                    "premise_idx": e["premise_idx"],
                    "occurrence_idx": e["occurrence_idx"],
                    "original_premise_translation": e["original_premise_translation"],
                    "mutated_premise_translation": (e["mutated_premise_translation"] if save_edits else None),
                    "premise_translations": item.get("premise_translations"),
                    "mutated_premise_translations": (e.get("mutated_premise_translations") if save_edits else None),
                    "conclusion_translation": conclusion,
                }
                outputs.append(rec)
            return outputs

        else:
            edits = find_label_changing_single_edits(premises, conclusion, original_canon, operators_to_test)
            for edit_step, e in enumerate(edits, start=1):
                rec = {
                    "id": item.get("id"),
                    "state": item.get("state"),
                    "label": original_canon,
                    "edit_step": edit_step,
                    "new_label": e["new_label"],
                    "raw_answer": e.get("raw_answer"),
                    "prover9_error": e.get("prover9_error"),
                    "operator_name": e["operator_name"],
                    "operator_kind": e["operator_kind"],
                    "operator_pattern": e["operator_pattern"],
                    "operator_repl": e["operator_repl"],
                    "premise_idx": e["premise_idx"],
                    "occurrence_idx": e["occurrence_idx"],
                    "original_premise_translation": e["original_premise_translation"],
                    "mutated_premise_translation": (e["mutated_premise_translation"] if save_edits else None),
                    "premise_translations": item.get("premise_translations"),
                    "mutated_premise_translations": (e.get("mutated_premise_translations") if save_edits else None),
                    "conclusion_translation": conclusion,
                }
                outputs.append(rec)
            return outputs

    raise ValueError(f"Unknown edit_type: {edit_type}")


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


DEFAULT_OPERATORS: List[OperatorSpec] = [
    OperatorSpec(name="or_to_and", kind="symbol", pattern=r"∨", repl="∧"),
    OperatorSpec(name="xor_to_or", kind="symbol", pattern=r"⊕", repl="∨"),
    OperatorSpec(name="or_to_xor", kind="symbol", pattern=r"∨", repl="⊕"),
    OperatorSpec(name="neg_to_no_neg", kind="symbol", pattern=r"¬", repl=""),
    OperatorSpec(name="imp_to_and", kind="symbol", pattern=r"→", repl="∧"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input JSON or JSONL")
    ap.add_argument("--output", required=True, help="Path to output JSONL")
    ap.add_argument("--edit-type", choices=["cumulative", "single"], default="cumulative")
    ap.add_argument("--final-answer-preservation", default="False", help="True|False (only used with --edit-type single)")
    ap.add_argument("--save-edits", type=int, choices=[0, 1], default=0, help="0 to avoid saving mutated premise programs, 1 to include them")
    args = ap.parse_args()

    edit_type = args.edit_type
    final_answer_preservation = str(args.final_answer_preservation).lower() in ("true", "1", "t", "yes", "y")
    if edit_type != "single" and final_answer_preservation:
        print("Warning: --final-answer-preservation only applies to --edit-type single; ignoring.")
        final_answer_preservation = False

    save_edits = int(args.save_edits)

    items = load_json_any(args.input)
    out_rows: List[Dict[str, Any]] = []

    for item in items:
        out_rows.extend(process_record(item, edit_type=edit_type, final_answer_preservation=final_answer_preservation, operators_to_test=DEFAULT_OPERATORS, save_edits=save_edits))

    write_jsonl(args.output, out_rows)


if __name__ == "__main__":
    main()
