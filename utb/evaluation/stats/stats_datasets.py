                      
"""Compute intrinsic statistics for Personalized-Wild-Tool-Bench datasets.

The script is intentionally focused on dataset properties, not model accuracy.
When a metric JSON is supplied, its model/result metadata is included only as
context and for coverage checks against the dataset being summarized.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


EVALUATION_ROOT = Path(__file__).resolve().parents[1]
if str(EVALUATION_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATION_ROOT))

from wtb.dataset_utils import (              
    build_cross_topic_entry,
    load_jsonl,
    normalize_task_type,
    normalize_turn_subtype,
    parse_env_datetime,
)


DEFAULT_DATA_DIR = EVALUATION_ROOT / "data-40-50"
DEFAULT_OUTPUT_DIR = EVALUATION_ROOT / "stats-datasets"


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def counter_payload(counter: Counter, total: int | None = None) -> dict[str, dict[str, Any]]:
    denominator = sum(counter.values()) if total is None else total
    return {
        str(key): {
            "count": value,
            "rate": safe_divide(value, denominator),
        }
        for key, value in sorted(counter.items(), key=lambda item: str(item[0]))
    }


def numeric_summary(values: Iterable[int | float]) -> dict[str, Any]:
    values = list(values)
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p25": None,
            "p75": None,
            "sum": 0,
        }

    sorted_values = sorted(values)

    def percentile(percent: float) -> float:
        if len(sorted_values) == 1:
            return float(sorted_values[0])
        position = (len(sorted_values) - 1) * percent
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return float(sorted_values[int(position)])
        weight = position - lower
        return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)

    return {
        "count": len(sorted_values),
        "min": min(sorted_values),
        "max": max(sorted_values),
        "mean": statistics.fmean(sorted_values),
        "median": statistics.median(sorted_values),
        "p25": percentile(0.25),
        "p75": percentile(0.75),
        "sum": sum(sorted_values),
    }


def text_len(value: Any) -> int:
    return len(value) if isinstance(value, str) else 0


def first_non_empty(primary: Any, fallback: Any) -> Any:
    if primary not in (None, "", []):
        return primary
    return fallback


def extract_function(tool: dict[str, Any]) -> dict[str, Any]:
    function = tool.get("function")
    return function if isinstance(function, dict) else {}


def extract_tool_names(tools: list[dict[str, Any]]) -> list[str]:
    names = []
    for tool in tools:
        function = extract_function(tool)
        name = function.get("name")
        if name:
            names.append(str(name))
    return names


def parameter_count(tool: dict[str, Any]) -> int:
    function = extract_function(tool)
    parameters = function.get("parameters")
    if not isinstance(parameters, dict):
        return 0
    properties = parameters.get("properties")
    return len(properties) if isinstance(properties, dict) else 0


def required_parameter_count(tool: dict[str, Any]) -> int:
    function = extract_function(tool)
    parameters = function.get("parameters")
    if not isinstance(parameters, dict):
        return 0
    required = parameters.get("required")
    return len(required) if isinstance(required, list) else 0


def iter_answer_actions(answer_list: Any) -> Iterable[dict[str, Any]]:
    if not isinstance(answer_list, list):
        return
    for candidate in answer_list:
        if not isinstance(candidate, dict):
            continue
        action = candidate.get("action")
        if isinstance(action, dict):
            yield action
        elif isinstance(action, list):
            for item in action:
                if isinstance(item, dict):
                    yield item


def action_name(action: dict[str, Any]) -> str:
    name = action.get("name")
    if name:
        return str(name)
    function = action.get("function")
    if isinstance(function, dict) and function.get("name"):
        return str(function["name"])
    return "<unknown>"


def parse_persona_id(dataset_name: str) -> str | None:
    match = re.search(r"persona(\d+)$", dataset_name)
    return match.group(1) if match else None


def env_time_payload(raw_topics: list[dict[str, Any]]) -> dict[str, Any]:
    datetimes = []
    weekdays = Counter()
    for topic in raw_topics:
        env_info = first_non_empty(topic.get("env_info"), topic.get("english_env_info"))
        parsed = parse_env_datetime(env_info)
        if parsed != datetime.max:
            datetimes.append(parsed)
            weekday_match = re.search(r"星期.", str(env_info))
            if weekday_match:
                weekdays[weekday_match.group(0)] += 1

    return {
        "min": min(datetimes).isoformat(sep=" ") if datetimes else None,
        "max": max(datetimes).isoformat(sep=" ") if datetimes else None,
        "weekday_distribution": counter_payload(weekdays),
    }


def summarize_dataset_file(dataset_file: Path, seed: int) -> dict[str, Any]:
    raw_topics = load_jsonl(dataset_file)
    cross_topic = build_cross_topic_entry(dataset_file, seed=seed)

    topic_turn_counts = []
    topic_tool_counts = []
    tool_parameter_counts = []
    tool_required_counts = []
    topic_unique_tool_counts = []
    task_text_lengths = []
    user_message_lengths = []
    assistant_message_lengths = []
    all_tool_names = Counter()

    for topic in raw_topics:
        topic_tasks = first_non_empty(topic.get("tasks"), topic.get("english_tasks")) or []
        topic_messages = first_non_empty(topic.get("messages"), topic.get("english_messages")) or []
        topic_tools = first_non_empty(topic.get("tools"), topic.get("english_tools")) or []

        topic_turn_counts.append(len(topic_tasks))
        tool_names = extract_tool_names(topic_tools)
        topic_tool_counts.append(len(topic_tools))
        topic_unique_tool_counts.append(len(set(tool_names)))
        all_tool_names.update(tool_names)
        tool_parameter_counts.extend(parameter_count(tool) for tool in topic_tools)
        tool_required_counts.extend(required_parameter_count(tool) for tool in topic_tools)

        for task in topic_tasks:
            task_text_lengths.append(text_len(task))
        for message in topic_messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content_len = text_len(message.get("content"))
            if role == "user":
                user_message_lengths.append(content_len)
            elif role == "assistant":
                assistant_message_lengths.append(content_len)

    task_types = [normalize_task_type(item) for item in cross_topic.get("task_types", [])]
    turn_subtypes = [normalize_turn_subtype(item) for item in cross_topic.get("turn_subtypes", [])]
    layer_counts = Counter(
        metadata.get("turn_index_in_topic")
        for metadata in cross_topic.get("turn_metadata", [])
        if isinstance(metadata, dict)
    )
    task_type_counts = Counter(task_types)
    task_type_counts["Multi-Tool"] += sum(
        value
        for key, value in Counter(task_types).items()
        if key != "Multi-Tool" and "Multi-Tool" in key
    )

    session_first_turn_count = 1 if cross_topic["num_turns"] else 0
    topic_first_turn_count = sum(
        1 for metadata in cross_topic.get("turn_metadata", []) if metadata.get("turn_index_in_topic") == 0
    )
    turn_subtype_counts = Counter(turn_subtypes)
    turn_subtype_counts["First Turn"] = session_first_turn_count
    turn_subtype_counts["Subsequent Turn"] = len(turn_subtypes)
    topic_position_counts = Counter(
        "Topic First Turn" if metadata.get("turn_index_in_topic") == 0 else "Topic Subsequent Turn"
        for metadata in cross_topic.get("turn_metadata", [])
        if isinstance(metadata, dict)
    )
    assert topic_first_turn_count == topic_position_counts["Topic First Turn"]

    action_counts = Counter()
    answer_steps_per_turn = []
    tool_action_steps_per_turn = []
    for answer in cross_topic.get("answer_list", []):
        actions = list(iter_answer_actions(answer))
        answer_steps_per_turn.append(len(actions))
        tool_action_count = 0
        for action in actions:
            name = action_name(action)
            action_counts[name] += 1
            if name not in {"prepare_to_answer", "ask_user_for_required_parameters"}:
                tool_action_count += 1
        tool_action_steps_per_turn.append(tool_action_count)

    return {
        "dataset_name": cross_topic["dataset_name"],
        "persona_id": parse_persona_id(cross_topic["dataset_name"]),
        "source_file": str(dataset_file.resolve()),
        "num_topics_original": len(raw_topics),
        "num_topics_after_dedup": cross_topic["num_topics_after_dedup"],
        "num_turns": cross_topic["num_turns"],
        "dedup_removed_topics": len(raw_topics) - cross_topic["num_topics_after_dedup"],
        "turns_per_topic": numeric_summary(topic_turn_counts),
        "task_type_distribution": counter_payload(task_type_counts, cross_topic["num_turns"]),
        "turn_subtype_distribution": counter_payload(turn_subtype_counts, cross_topic["num_turns"]),
        "topic_position_distribution": counter_payload(topic_position_counts, cross_topic["num_turns"]),
        "turn_index_in_topic_distribution": counter_payload(layer_counts, cross_topic["num_turns"]),
        "tool_stats": {
            "tools_per_topic": numeric_summary(topic_tool_counts),
            "unique_tools_per_topic": numeric_summary(topic_unique_tool_counts),
            "unique_tool_name_count": len(all_tool_names),
            "tool_name_distribution": counter_payload(all_tool_names),
            "parameters_per_tool": numeric_summary(tool_parameter_counts),
            "required_parameters_per_tool": numeric_summary(tool_required_counts),
        },
        "answer_action_stats": {
            "answer_steps_per_turn": numeric_summary(answer_steps_per_turn),
            "tool_action_steps_per_turn": numeric_summary(tool_action_steps_per_turn),
            "action_name_distribution": counter_payload(action_counts),
        },
        "text_length_stats": {
            "task_chars": numeric_summary(task_text_lengths),
            "user_message_chars": numeric_summary(user_message_lengths),
            "assistant_message_chars": numeric_summary(assistant_message_lengths),
        },
        "env_time": env_time_payload(raw_topics),
    }


def combine_file_stats(file_stats: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "num_dataset_files": len(file_stats),
        "num_topics_original": sum(item["num_topics_original"] for item in file_stats),
        "num_topics_after_dedup": sum(item["num_topics_after_dedup"] for item in file_stats),
        "num_turns": sum(item["num_turns"] for item in file_stats),
        "dedup_removed_topics": sum(item["dedup_removed_topics"] for item in file_stats),
    }

    task_types = Counter()
    turn_subtypes = Counter()
    topic_positions = Counter()
    turn_indices_in_topic = Counter()
    tool_names = Counter()
    actions = Counter()
    for item in file_stats:
        task_types.update({key: value["count"] for key, value in item["task_type_distribution"].items()})
        turn_subtypes.update({key: value["count"] for key, value in item["turn_subtype_distribution"].items()})
        topic_positions.update({key: value["count"] for key, value in item["topic_position_distribution"].items()})
        turn_indices_in_topic.update(
            {key: value["count"] for key, value in item["turn_index_in_topic_distribution"].items()}
        )
        tool_names.update({key: value["count"] for key, value in item["tool_stats"]["tool_name_distribution"].items()})
        actions.update({key: value["count"] for key, value in item["answer_action_stats"]["action_name_distribution"].items()})

    return {
        **totals,
        "turns_per_file": numeric_summary([item["num_turns"] for item in file_stats]),
        "topics_per_file": numeric_summary([item["num_topics_after_dedup"] for item in file_stats]),
        "task_type_distribution": counter_payload(task_types, totals["num_turns"]),
        "turn_subtype_distribution": counter_payload(turn_subtypes, totals["num_turns"]),
        "topic_position_distribution": counter_payload(topic_positions, totals["num_turns"]),
        "turn_index_in_topic_distribution": counter_payload(turn_indices_in_topic, totals["num_turns"]),
        "unique_tool_name_count": len(tool_names),
        "tool_name_distribution": counter_payload(tool_names),
        "answer_action_distribution": counter_payload(actions),
    }


def load_metric_context(metric_path: Path | None, file_stats: list[dict[str, Any]]) -> dict[str, Any] | None:
    if metric_path is None:
        return None
    with metric_path.open(encoding="utf-8") as f:
        metric = json.load(f)

    dataset_name = metric.get("dataset_name")
    persona_id = parse_persona_id(str(dataset_name))
    exact_matches = [item["dataset_name"] for item in file_stats if item["dataset_name"] == dataset_name]
    persona_matches = [
        item["dataset_name"]
        for item in file_stats
        if persona_id is not None and item.get("persona_id") == persona_id
    ]

    metric_task_total = metric.get("total_info", {}).get("task", {}).get("total_count", 0)
    metric_num_turns = metric.get("num_turns", 0)

    return {
        "metric_file": str(metric_path.resolve()),
        "model_name": metric.get("model_name"),
        "dataset_name": dataset_name,
        "source_file": metric.get("source_file"),
        "metric_num_turns": metric_num_turns,
        "metric_num_scored_turns": metric.get("num_scored_turns"),
        "metric_task_total_count": metric_task_total,
        "metric_scored_turn_rate": safe_divide(metric_task_total, metric_num_turns),
        "inference_completed": metric.get("inference_completed"),
        "failed_task_idx": metric.get("failed_task_idx"),
        "failed_step": metric.get("failed_step"),
        "error_type": metric.get("error_type"),
        "exact_dataset_name_matches_in_data_dir": exact_matches,
        "persona_id_matches_in_data_dir": persona_matches,
        "metric_task_type_totals": {
            key: value.get("total_count", 0)
            for key, value in metric.get("task_type_info", {}).items()
            if isinstance(value, dict)
        },
        "metric_turn_subtype_totals": {
            key: value.get("total_count", 0)
            for key, value in metric.get("turn_subtype_info", {}).items()
            if isinstance(value, dict)
        },
    }


def write_markdown(report: dict[str, Any], output_path: Path) -> None:
    summary = report["summary"]
    metric = report.get("metric_context")

    lines = [
        "# Dataset Statistics",
        "",
        f"- Data dir: `{report['data_dir']}`",
        f"- Dataset files: {summary['num_dataset_files']}",
        f"- Topics after dedup: {summary['num_topics_after_dedup']}",
        f"- Turns: {summary['num_turns']}",
        f"- Unique tool names: {summary['unique_tool_name_count']}",
        "",
        "## Task Types",
        "",
        "| Type | Count | Rate |",
        "| --- | ---: | ---: |",
    ]
    for key, value in summary["task_type_distribution"].items():
        lines.append(f"| {key} | {value['count']} | {value['rate']:.4f} |")

    lines.extend(["", "## Turn Subtypes", "", "| Subtype | Count | Rate |", "| --- | ---: | ---: |"])
    for key, value in summary["turn_subtype_distribution"].items():
        lines.append(f"| {key} | {value['count']} | {value['rate']:.4f} |")

    if metric:
        lines.extend(
            [
                "",
                "## Metric Context",
                "",
                f"- Metric file: `{metric['metric_file']}`",
                f"- Model: `{metric.get('model_name')}`",
                f"- Metric dataset: `{metric.get('dataset_name')}`",
                f"- Metric source file: `{metric.get('source_file')}`",
                f"- Scored turns: {metric.get('metric_num_scored_turns')}",
                f"- Inference completed: {metric.get('inference_completed')}",
                f"- Persona matches in data dir: {', '.join(metric['persona_id_matches_in_data_dir']) or 'none'}",
            ]
        )

    lines.extend(["", "## Files", "", "| Dataset | Topics | Turns | Tools(unique) |", "| --- | ---: | ---: | ---: |"])
    for item in report["files"]:
        lines.append(
            "| {name} | {topics} | {turns} | {tools} |".format(
                name=item["dataset_name"],
                topics=item["num_topics_after_dedup"],
                turns=item["num_turns"],
                tools=item["tool_stats"]["unique_tool_name_count"],
            )
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory containing dataset JSONL files.")
    parser.add_argument("--metric-json", type=Path, default=None, help="Optional metric JSON used as result context.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for JSON/Markdown outputs.")
    parser.add_argument("--output-prefix", default=None, help="Output filename prefix. Defaults to the data directory name.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used by dataset_utils deduplication.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_files = sorted(data_dir.glob("*.jsonl"))
    if not dataset_files:
        raise FileNotFoundError(f"No JSONL files found under {data_dir}")

    file_stats = [summarize_dataset_file(dataset_file, args.seed) for dataset_file in dataset_files]
    report = {
        "data_dir": str(data_dir),
        "seed": args.seed,
        "summary": combine_file_stats(file_stats),
        "metric_context": load_metric_context(args.metric_json.resolve() if args.metric_json else None, file_stats),
        "files": file_stats,
    }

    output_prefix = args.output_prefix or data_dir.name
    json_path = output_dir / f"{output_prefix}_dataset_stats.json"
    md_path = output_dir / f"{output_prefix}_dataset_stats.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(report, md_path)

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(
        "Summary: "
        f"{report['summary']['num_dataset_files']} files, "
        f"{report['summary']['num_topics_after_dedup']} topics, "
        f"{report['summary']['num_turns']} turns"
    )


if __name__ == "__main__":
    main()
