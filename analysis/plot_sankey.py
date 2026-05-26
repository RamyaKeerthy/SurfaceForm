#!/usr/bin/env python3
"""Plot Sankey diagrams for correctness flow across cumulative edits."""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import plotly.graph_objects as go


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    family: str
    size: str
    label: str


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(family="gemma3", size="1b", label="gemma3-1b"),
    ModelSpec(family="gemma3", size="4b", label="gemma3-4b"),
    ModelSpec(family="gemma3", size="12b", label="gemma3-12b"),
    ModelSpec(family="gemma3", size="27b", label="gemma3-27b"),
    ModelSpec(family="llama3", size="8b", label="llama3-8b"),
    ModelSpec(family="qwen3", size="4b", label="qwen3-4b"),
    ModelSpec(family="phi4mini", size="4b", label="phi4mini-4b"),
)

METHOD_TEMPLATES = {
    "cot": "{data}_test_{model}-{size}_cot-{mutation}-mutation.jsonl",
    "scaling": "{data}_test_{model}-{size}_cot_32samples-{mutation}-mutation_5shot.jsonl",
    "symbcot": "{data}_test_{model}-{size}_solver-{mutation}-mutation_5shot.jsonl",
}

DEFAULT_MAX_EDIT_BY_METHOD = {
    "cot": None,
    "scaling": None,
    "symbcot": 3,
}

SUPPORTED_LEVELS = {
    ("folio", "operator"): [0, 1, 2],
    ("logicaldeduction", "operator"): [0, 1, 2, 3, 4],
    ("arlsat", "operator"): [0, 1, 2, 3],
}

CHOICE_TOKEN_RE = re.compile(r"\b([a-z])\b", re.IGNORECASE)
NUMBER_TOKEN_RE = re.compile(r"\b(\d+)\b")
BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
FINAL_ANSWER_RE = re.compile(
    r"(?:final answer|answer|prediction|therefore)\s*(?:is|=|:)?\s*([^\n.;]+)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="folio",
        choices=sorted({dataset for dataset, _ in SUPPORTED_LEVELS}),
        help="Dataset name used in the input filename template.",
    )
    parser.add_argument(
        "--mutation",
        default="operator",
        help="Mutation name used in the input filename template.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["cot", "scaling", "symbcot"],
        choices=sorted(METHOD_TEMPLATES),
        help="Methods to plot.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing the JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="If provided, write one HTML file per generated figure.",
    )
    parser.add_argument(
        "--max-edit",
        type=int,
        default=None,
        help="Global maximum edit level to plot. Overrides method defaults.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open generated figures interactively.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def get_levels(data: str, mutation_name: str) -> list[int]:
    key = (data.lower(), mutation_name.lower())
    try:
        return SUPPORTED_LEVELS[key]
    except KeyError as exc:
        supported = ", ".join(f"{dataset}/{mutation}" for dataset, mutation in SUPPORTED_LEVELS)
        raise ValueError(
            f"Unsupported data/mutation combination {data!r}/{mutation_name!r}. "
            f"Supported combinations: {supported}."
        ) from exc


def resolve_input_path(
    template: str,
    base_dir: Path,
    data: str,
    model_spec: ModelSpec,
    mutation: str,
) -> Path:
    relative_path = Path(
        template.format(
            data=data.lower(),
            model=model_spec.family.lower(),
            size=model_spec.size,
            mutation=mutation.lower(),
        )
    )
    if relative_path.is_absolute():
        return relative_path
    return base_dir / relative_path


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}.") from exc
    return pd.DataFrame(rows)


def normalize_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().lower()


def iter_text_chunks(value: Any) -> Iterable[str]:
    if value is None:
        return
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            yield stripped
        return
    if isinstance(value, list):
        for item in value:
            yield from iter_text_chunks(item)
        return
    if isinstance(value, dict):
        preferred_keys = ("text", "content", "output", "answer", "response", "completion")
        for key in preferred_keys:
            if key in value:
                yield from iter_text_chunks(value[key])
                return
        for item in value.values():
            yield from iter_text_chunks(item)
        return
    stripped = str(value).strip()
    if stripped:
        yield stripped


def majority_vote(predictions: Sequence[str]) -> str:
    cleaned = [prediction for prediction in predictions if prediction]
    if not cleaned:
        return ""
    counts = Counter(cleaned)
    top_count = max(counts.values())
    for prediction in reversed(cleaned):
        if counts[prediction] == top_count:
            return prediction
    return ""


def infer_label_space(label: Any) -> tuple[str, ...] | None:
    normalized_label = normalize_value(label)
    if not normalized_label:
        return None
    if len(normalized_label) == 1 and normalized_label.isalpha():
        return ("a", "b", "c", "d", "e")
    if normalized_label.isdigit():
        return tuple(str(index) for index in range(1, 11))
    word_spaces = {
        "true": ("true", "false"),
        "false": ("true", "false"),
        "yes": ("yes", "no"),
        "no": ("yes", "no"),
        "entailment": ("entailment", "contradiction", "unknown"),
        "contradiction": ("entailment", "contradiction", "unknown"),
        "unknown": ("entailment", "contradiction", "unknown"),
        "uncertain": ("true", "false", "uncertain"),
    }
    return word_spaces.get(normalized_label)


def extract_candidate_from_segment(segment: str, label_space: tuple[str, ...] | None) -> str:
    normalized_segment = normalize_value(segment)
    if not normalized_segment:
        return ""

    boxed_matches = BOXED_RE.findall(segment)
    if boxed_matches:
        normalized_boxed = normalize_value(boxed_matches[-1])
        if normalized_boxed:
            return normalized_boxed

    final_match = FINAL_ANSWER_RE.search(segment)
    if final_match:
        normalized_final = normalize_value(final_match.group(1))
        if normalized_final:
            segment = normalized_final
            normalized_segment = normalized_final

    if label_space:
        if all(len(option) == 1 and option.isalpha() for option in label_space):
            matches = [match.group(1).lower() for match in CHOICE_TOKEN_RE.finditer(normalized_segment)]
            for match in reversed(matches):
                if match in label_space:
                    return match
        elif all(option.isdigit() for option in label_space):
            matches = [match.group(1) for match in NUMBER_TOKEN_RE.finditer(normalized_segment)]
            for match in reversed(matches):
                if match in label_space:
                    return match
        else:
            pattern = re.compile(
                r"\b(" + "|".join(sorted(map(re.escape, label_space), key=len, reverse=True)) + r")\b",
                re.IGNORECASE,
            )
            matches = [normalize_value(match.group(1)) for match in pattern.finditer(normalized_segment)]
            if matches:
                return matches[-1]

    return normalized_segment


def extract_prediction(output: Any, label: Any) -> str:
    if isinstance(output, list):
        return majority_vote([extract_prediction(item, label) for item in output])

    label_space = infer_label_space(label)
    segments = list(iter_text_chunks(output))
    for segment in reversed(segments):
        candidate = extract_candidate_from_segment(segment, label_space)
        if candidate:
            return candidate
    return ""


def add_correctness_column(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"label", "output"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing columns required for correctness computation: {missing_columns}")

    result = df.copy()
    result["label_norm"] = result["label"].apply(normalize_value)
    result["prediction"] = result.apply(
        lambda row: extract_prediction(row["output"], row["label"]),
        axis=1,
    )
    result["prediction_norm"] = result["prediction"].apply(normalize_value)
    result["is_correct"] = (result["label_norm"] == result["prediction_norm"]).astype(int)
    return result


def state_name(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return "Correct" if int(value) == 1 else "Incorrect"


def build_flow_counts(
    df: pd.DataFrame,
    levels: Sequence[int],
) -> tuple[Counter[tuple[int, str, int, str]], Counter[tuple[int, str]], int]:
    if not levels:
        raise ValueError("At least one edit level is required.")

    df = add_correctness_column(df)

    needed_columns = {"id", "num_edits_preserving_label", "is_correct"}
    missing_columns = sorted(needed_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    filtered = df[df["num_edits_preserving_label"].isin(levels)].copy()
    filtered = filtered.sort_values(["id", "num_edits_preserving_label"]).drop_duplicates(
        subset=["id", "num_edits_preserving_label"],
        keep="first",
    )

    paths_df = (
        filtered.pivot(index="id", columns="num_edits_preserving_label", values="is_correct")
        .reindex(columns=levels)
    )
    paths_df = paths_df[~paths_df[levels[0]].isna()].copy()

    transition_counts: Counter[tuple[int, str, int, str]] = Counter()
    ended_counts: Counter[tuple[int, str]] = Counter()

    for _, row in paths_df.iterrows():
        for current_level, next_level in zip(levels, levels[1:]):
            current_state = state_name(row[current_level])
            next_state = state_name(row[next_level])
            if current_state is not None and next_state is not None:
                transition_counts[(current_level, current_state, next_level, next_state)] += 1
            elif current_state is not None and next_state is None:
                ended_counts[(current_level, current_state)] += 1

    return transition_counts, ended_counts, len(paths_df)


def build_sankey_inputs(
    transition_counts: Counter[tuple[int, str, int, str]],
    ended_counts: Counter[tuple[int, str]],
    levels: Sequence[int],
) -> dict[str, dict[str, Any]]:
    node_labels: list[str] = []
    node_colors: list[str] = []
    node_x_positions: list[float] = []
    node_y_positions: list[float] = []

    color_correct = "rgba(42, 126, 92, 0.85)"
    color_incorrect = "rgba(156, 64, 58, 0.85)"
    color_ended = "rgba(110, 110, 110, 0.85)"

    link_color_correct = "rgba(46, 204, 113, 0.30)"
    link_color_incorrect = "rgba(231, 76, 60, 0.30)"
    link_color_ended = "rgba(180, 180, 180, 0.30)"

    def get_x_pos_for_edit(index: int) -> float:
        if len(levels) == 1:
            return 0.5
        return 0.15 + 0.8 * (index / (len(levels) - 1))

    for index, level in enumerate(levels):
        x_current = get_x_pos_for_edit(index)

        node_labels.append(f"e{level} Correct")
        node_colors.append(color_correct)
        node_x_positions.append(x_current)
        node_y_positions.append(0.20)

        node_labels.append(f"e{level} Incorrect")
        node_colors.append(color_incorrect)
        node_x_positions.append(x_current)
        node_y_positions.append(0.50)

        if index < len(levels) - 1:
            x_ended = get_x_pos_for_edit(index + 1)
        else:
            x_ended = min(x_current + 0.05, 0.99)

        node_labels.append(f"e{level} Ended")
        node_colors.append(color_ended)
        node_x_positions.append(x_ended)
        node_y_positions.append(0.80)

    label_to_index = {label: index for index, label in enumerate(node_labels)}

    sources: list[int] = []
    targets: list[int] = []
    values: list[int] = []
    link_colors: list[str] = []

    for (current_level, current_state, next_level, next_state), count in transition_counts.items():
        sources.append(label_to_index[f"e{current_level} {current_state}"])
        targets.append(label_to_index[f"e{next_level} {next_state}"])
        values.append(count)
        link_colors.append(link_color_correct if next_state == "Correct" else link_color_incorrect)

    for (current_level, current_state), count in ended_counts.items():
        if count <= 0:
            continue
        sources.append(label_to_index[f"e{current_level} {current_state}"])
        targets.append(label_to_index[f"e{current_level} Ended"])
        values.append(count)
        link_colors.append(link_color_ended)

    return {
        "node": {
            "pad": 35,
            "thickness": 40,
            "line": {"color": "black", "width": 0.4},
            "label": [""] * len(node_labels),
            "customdata": node_labels,
            "color": node_colors,
            "x": node_x_positions,
            "y": node_y_positions,
            "hovertemplate": "%{customdata}<extra></extra>",
        },
        "link": {
            "source": sources,
            "target": targets,
            "value": values,
            "color": link_colors,
            "hovertemplate": "Count: %{value}<extra></extra>",
        },
    }


def plot_model_flow(
    model_spec: ModelSpec,
    template_path: str,
    base_dir: Path,
    data: str,
    mutation: str,
    levels: Sequence[int],
    max_edit: int | None = None,
) -> go.Figure | None:
    path = resolve_input_path(template_path, base_dir, data, model_spec, mutation)
    if not path.exists():
        LOGGER.warning("Missing input for %s: %s", model_spec.label, path)
        return None

    current_levels = [level for level in levels if max_edit is None or level <= max_edit]
    if not current_levels:
        LOGGER.warning("No levels left to plot for %s after max_edit=%s", model_spec.label, max_edit)
        return None

    df = load_jsonl(path)
    transition_counts, ended_counts, n_records = build_flow_counts(df, current_levels)
    sankey_data = build_sankey_inputs(transition_counts, ended_counts, current_levels)

    figure = go.Figure(data=[go.Sankey(arrangement="fixed", **sankey_data)])
    figure.update_layout(
        title=f"{model_spec.label} correctness flow across edits (n={n_records})",
        showlegend=False,
        font={"size": 13},
        width=1200,
        height=550,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    return figure


def write_figure(figure: go.Figure, output_dir: Path, method: str, model_spec: ModelSpec) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{method}_{model_spec.label}_sankey.html"
    figure.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    levels = get_levels(args.data, args.mutation)
    base_dir = args.base_dir.resolve()
    show_figures = args.show or args.output_dir is None
    generated = 0

    for method in args.methods:
        LOGGER.info("Plotting %s models", method)
        template = METHOD_TEMPLATES[method]
        method_max_edit = args.max_edit
        if method_max_edit is None:
            method_max_edit = DEFAULT_MAX_EDIT_BY_METHOD[method]

        for model_spec in MODEL_SPECS:
            figure = plot_model_flow(
                model_spec=model_spec,
                template_path=template,
                base_dir=base_dir,
                data=args.data,
                mutation=args.mutation,
                levels=levels,
                max_edit=method_max_edit,
            )
            if figure is None:
                continue
            generated += 1
            if args.output_dir is not None:
                output_path = write_figure(figure, args.output_dir.resolve(), method, model_spec)
                LOGGER.info("Wrote %s", output_path)
            if show_figures:
                figure.show()

    if generated == 0:
        LOGGER.warning("No figures were generated.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
