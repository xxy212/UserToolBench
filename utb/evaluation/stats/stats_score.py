                      
"""Summarize score/evaluation outputs by profile.

The default input is the score-40-100/Kimi-K2.6 result directory. Outputs are
named with the dataset directory name so repeated runs for different datasets
do not overwrite each other accidentally.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


EVALUATION_ROOT = Path(__file__).resolve().parents[1]
if str(EVALUATION_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATION_ROOT))

from wtb.dataset_utils import build_cross_topic_entry, normalize_task_type

DEFAULT_SCORE_DIR = EVALUATION_ROOT / "score-10-30" / "qwen3.6-plus"
DEFAULT_OUTPUT_DIR = EVALUATION_ROOT / "stats_score"
TURN_SEGMENTS = ("first_third", "middle_third", "last_third")
SEGMENT_TASK_TYPES = ("task", "Single-Tool", "Multi-Tool", "Clarify", "Chat")


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def parse_profile_id(value: str) -> str:
    match = re.search(r"persona(\d+)", value)
    return match.group(1) if match else value


def profile_sort_key(profile: str) -> tuple[int, int | str]:
    return (0, int(profile)) if str(profile).isdigit() else (1, profile)


def percentile(sorted_values: list[float], percent: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * percent
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[int(position)])
    weight = position - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def numeric_summary(values: Iterable[int | float]) -> dict[str, Any]:
    clean_values = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
    ]
    clean_values.sort()
    if not clean_values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "variance_population": None,
            "variance_sample": None,
            "stdev_population": None,
            "stdev_sample": None,
            "median": None,
            "p25": None,
            "p75": None,
            "sum": 0,
        }

    return {
        "count": len(clean_values),
        "min": min(clean_values),
        "max": max(clean_values),
        "mean": statistics.fmean(clean_values),
        "variance_population": statistics.pvariance(clean_values),
        "variance_sample": statistics.variance(clean_values) if len(clean_values) > 1 else None,
        "stdev_population": statistics.pstdev(clean_values),
        "stdev_sample": statistics.stdev(clean_values) if len(clean_values) > 1 else None,
        "median": statistics.median(clean_values),
        "p25": percentile(clean_values, 0.25),
        "p75": percentile(clean_values, 0.75),
        "sum": sum(clean_values),
    }


def flatten_numeric(value: Any, prefix: str = "") -> dict[str, float]:
    flattened: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(flatten_numeric(child, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}.{index}" if prefix else str(index)
            flattened.update(flatten_numeric(child, child_prefix))
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        flattened[prefix] = float(value)
    return flattened


def read_first_json_line(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                return json.loads(line)
    return {}


def load_metric_profiles(score_dir: Path) -> list[dict[str, Any]]:
    profiles = []
    for path in sorted(score_dir.glob("*_metric.json")):
        with path.open(encoding="utf-8") as handle:
            metric = json.load(handle)
        dataset_name = str(metric.get("dataset_name") or path.name)
        profile = parse_profile_id(dataset_name)
        profiles.append(
            {
                "profile": profile,
                "dataset_name": dataset_name,
                "model_name": metric.get("model_name"),
                "source_file": metric.get("source_file"),
                "metric_file": str(path),
                "inference_completed": metric.get("inference_completed"),
                "error_type": metric.get("error_type"),
                "raw_metric": metric,
                "numeric": flatten_numeric(metric),
            }
        )
    return sorted(profiles, key=lambda item: profile_sort_key(item["profile"]))


def average_metric_leaf(values: list[Any], path: tuple[str, ...], dataset_name: str, model_name: str) -> Any:
    present_values = [value for value in values if value is not None]
    numeric_values = [
        float(value)
        for value in present_values
        if isinstance(value, (int, float, bool)) and math.isfinite(float(value))
    ]
    if numeric_values and len(numeric_values) == len(present_values):
        return statistics.fmean(numeric_values)

    field_name = path[-1] if path else ""
    if field_name == "model_name":
        unique_values = {value for value in present_values if isinstance(value, str)}
        return unique_values.pop() if len(unique_values) == 1 else model_name
    if field_name == "dataset_name":
        return f"{dataset_name}_summary_average"
    if field_name in {"source_file", "error_type", "error_message", "api_error"}:
        return None

    unique_json_values = {json.dumps(value, ensure_ascii=False, sort_keys=True) for value in present_values}
    if len(unique_json_values) == 1 and present_values:
        return present_values[0]
    return None


def average_metric_values(values: list[Any], path: tuple[str, ...], dataset_name: str, model_name: str) -> Any:
    dict_values = [value for value in values if isinstance(value, dict)]
    if dict_values and len(dict_values) == len([value for value in values if value is not None]):
        ordered_keys: list[str] = []
        for value in dict_values:
            for key in value:
                if key not in ordered_keys:
                    ordered_keys.append(key)
        return {
            key: average_metric_values(
                [value.get(key) for value in dict_values],
                (*path, str(key)),
                dataset_name,
                model_name,
            )
            for key in ordered_keys
        }

    list_values = [value for value in values if isinstance(value, list)]
    if list_values and len(list_values) == len([value for value in values if value is not None]):
        max_len = max((len(value) for value in list_values), default=0)
        return [
            average_metric_values(
                [value[index] for value in list_values if index < len(value)],
                (*path, str(index)),
                dataset_name,
                model_name,
            )
            for index in range(max_len)
        ]

    return average_metric_leaf(values, path, dataset_name, model_name)


def average_metric_summary(metric_profiles: list[dict[str, Any]], dataset_name: str, model_name: str) -> dict[str, Any]:
    metrics = [profile["raw_metric"] for profile in metric_profiles if isinstance(profile.get("raw_metric"), dict)]
    if not metrics:
        return {}
    averaged = average_metric_values(metrics, (), dataset_name, model_name)
    if not isinstance(averaged, dict):
        return {}
    return averaged


def load_score_profiles(score_dir: Path, max_file_mb: float) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    categorical_keys = ("label", "action_name_label", "is_optimal")
    numeric_list_keys = ("latency", "input_token_count", "output_token_count")
    max_bytes = max_file_mb * 1024 * 1024

    for path in sorted(score_dir.glob("*_score.jsonl")):
        profile_from_name = parse_profile_id(path.name)
        if max_file_mb >= 0 and path.stat().st_size > max_bytes:
            profiles[profile_from_name] = {
                "profile": profile_from_name,
                "dataset_name": path.name.split("__", 1)[0],
                "model_name": None,
                "score_file": str(path),
                "score_file_size_bytes": path.stat().st_size,
                "skipped_raw_parse": True,
                "skip_reason": f"file larger than --max-score-file-mb ({max_file_mb})",
                "evaluated_turns": None,
                "inference_completed": None,
                "error_type": None,
                "result_count": 0,
                "categorical": {key: Counter() for key in categorical_keys},
                "numeric": {},
            }
            continue

        payload = read_first_json_line(path)
        dataset_name = str(payload.get("dataset_name") or payload.get("id") or path.name)
        profile = parse_profile_id(dataset_name)
        categorical: dict[str, Counter] = {key: Counter() for key in categorical_keys}
        numeric: dict[str, list[float]] = defaultdict(list)
        results = payload.get("results") if isinstance(payload.get("results"), list) else []

        for result in results:
            if not isinstance(result, dict):
                continue
            for key in categorical_keys:
                if key in result:
                    categorical[key][str(result.get(key))] += 1
            if isinstance(result.get("is_optimal"), bool):
                numeric["is_optimal"].append(1.0 if result["is_optimal"] else 0.0)
            for key in numeric_list_keys:
                values = result.get(key)
                if isinstance(values, list):
                    numeric[key].extend(
                        float(value)
                        for value in values
                        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
                    )
                elif isinstance(values, (int, float)) and not isinstance(values, bool) and math.isfinite(float(values)):
                    numeric[key].append(float(values))

        profiles[profile] = {
            "profile": profile,
            "dataset_name": dataset_name,
            "model_name": payload.get("model_name"),
            "score_file": str(path),
            "score_file_size_bytes": path.stat().st_size,
            "skipped_raw_parse": False,
            "skip_reason": None,
            "evaluated_turns": payload.get("evaluated_turns"),
            "inference_completed": payload.get("inference_completed"),
            "error_type": payload.get("error_type"),
            "result_count": len(results),
            "categorical": categorical,
            "numeric": dict(numeric),
        }

    return dict(sorted(profiles.items(), key=lambda item: profile_sort_key(item[0])))


def metric_outcome_frequency_rows(metric_profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for profile in metric_profiles:
        numeric = profile["numeric"]
        for key, total in sorted(numeric.items()):
            if not key.endswith(".total_count"):
                continue
            prefix = key[: -len(".total_count")]
            correct = numeric.get(f"{prefix}.correct_count")
            if correct is None:
                continue
            total_int = int(total)
            correct_int = int(correct)
            error_int = max(total_int - correct_int, 0)
            for value, count in (("correct", correct_int), ("error", error_int)):
                rows.append(
                    {
                        "profile": profile["profile"],
                        "dataset_name": profile["dataset_name"],
                        "key": prefix,
                        "value": value,
                        "count": count,
                        "total": total_int,
                        "rate": safe_divide(count, total_int),
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def metric_by_profile_rows(metric_profiles: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    numeric_keys = sorted({key for profile in metric_profiles for key in profile["numeric"]})
    base_fields = [
        "profile",
        "dataset_name",
        "model_name",
        "source_file",
        "metric_file",
        "inference_completed",
        "error_type",
    ]
    rows = []
    for profile in metric_profiles:
        row = {key: profile.get(key) for key in base_fields}
        row.update(profile["numeric"])
        rows.append(row)
    return rows, base_fields + numeric_keys


def metric_key_stats_rows(metric_profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    profile_count = len(metric_profiles)
    by_key: dict[str, list[float]] = defaultdict(list)
    for profile in metric_profiles:
        for key, value in profile["numeric"].items():
            by_key[key].append(value)

    rows = []
    for key in sorted(by_key):
        summary = numeric_summary(by_key[key])
        rows.append(
            {
                "key": key,
                "profile_count": summary["count"],
                "missing_profile_count": profile_count - summary["count"],
                **summary,
            }
        )
    return rows


def score_frequency_rows(score_profiles: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for profile, payload in score_profiles.items():
        for key, counter in payload["categorical"].items():
            total = sum(counter.values())
            for value, count in sorted(counter.items(), key=lambda item: str(item[0])):
                rows.append(
                    {
                        "profile": profile,
                        "dataset_name": payload["dataset_name"],
                        "key": key,
                        "value": value,
                        "count": count,
                        "total": total,
                        "rate": safe_divide(count, total),
                    }
                )
    return rows


def raw_score_file_rows(score_profiles: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "profile": profile,
            "dataset_name": payload["dataset_name"],
            "score_file": payload["score_file"],
            "score_file_size_bytes": payload["score_file_size_bytes"],
            "skipped_raw_parse": payload["skipped_raw_parse"],
            "skip_reason": payload["skip_reason"],
            "result_count": payload["result_count"],
        }
        for profile, payload in score_profiles.items()
    ]


def score_numeric_profile_rows(score_profiles: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for profile, payload in score_profiles.items():
        for key, values in sorted(payload["numeric"].items()):
            rows.append(
                {
                    "profile": profile,
                    "dataset_name": payload["dataset_name"],
                    "key": key,
                    **numeric_summary(values),
                }
            )
    return rows


def score_key_stats_across_profiles(rows_by_profile: list[dict[str, Any]]) -> list[dict[str, Any]]:
    means_by_key: dict[str, list[float]] = defaultdict(list)
    for row in rows_by_profile:
        mean = row.get("mean")
        if isinstance(mean, (int, float)) and not isinstance(mean, bool):
            means_by_key[str(row["key"])].append(float(mean))

    rows = []
    for key in sorted(means_by_key):
        rows.append({"key": key, "statistic": "profile_mean", **numeric_summary(means_by_key[key])})
    return rows


def get_result_task_idx(result: dict[str, Any], fallback_index: int) -> int:
    inference_log = result.get("inference_log")
    if isinstance(inference_log, dict):
        task_idx = inference_log.get("task_idx")
        if isinstance(task_idx, int) and not isinstance(task_idx, bool):
            return task_idx
    return fallback_index


def split_turn_segments(total_turns: int) -> list[tuple[str, int, int]]:
    quotient, remainder = divmod(total_turns, len(TURN_SEGMENTS))
    sizes = [quotient + (1 if index < remainder else 0) for index in range(len(TURN_SEGMENTS))]
    segments = []
    start = 0
    for name, size in zip(TURN_SEGMENTS, sizes):
        end = start + size
        segments.append((name, start, end))
        start = end
    return segments


def load_task_types_for_score_payload(payload: dict[str, Any]) -> list[str]:
    source_file = payload.get("source_file")
    if not source_file:
        return []
    source_path = Path(str(source_file))
    if not source_path.exists():
        return []
    entry = build_cross_topic_entry(source_path)
    task_types = entry.get("task_types") if isinstance(entry, dict) else []
    if not isinstance(task_types, list):
        return []
    return [normalize_task_type(task_type) for task_type in task_types]


def increment_segment_counts(counts: dict[str, dict[str, int]], task_type: str | None, correct: bool) -> None:
    counts["task"]["total_count"] += 1
    if correct:
        counts["task"]["correct_count"] += 1

    if not task_type:
        return
    normalized_task_type = normalize_task_type(task_type)
    task_types = [normalized_task_type]
    if normalized_task_type != "Multi-Tool" and "Multi-Tool" in normalized_task_type:
        task_types.append("Multi-Tool")

    for task_type_name in task_types:
        if task_type_name not in counts:
            continue
        counts[task_type_name]["total_count"] += 1
        if correct:
            counts[task_type_name]["correct_count"] += 1


def turn_segment_accuracy_by_profile_rows(score_dir: Path, max_file_mb: float) -> list[dict[str, Any]]:
    rows = []
    max_bytes = max_file_mb * 1024 * 1024

    for path in sorted(score_dir.glob("*_score.jsonl")):
        if max_file_mb >= 0 and path.stat().st_size > max_bytes:
            continue

        payload = read_first_json_line(path)
        dataset_name = str(payload.get("dataset_name") or payload.get("id") or path.name)
        profile = parse_profile_id(dataset_name)
        results = payload.get("results") if isinstance(payload.get("results"), list) else []
        if not results:
            continue

        task_types = load_task_types_for_score_payload(payload)
        scored_turns = []
        for fallback_index, result in enumerate(results):
            if not isinstance(result, dict):
                continue
            task_idx = get_result_task_idx(result, fallback_index)
            task_type = task_types[task_idx] if 0 <= task_idx < len(task_types) else None
            scored_turns.append(
                {
                    "sort_key": (task_idx, fallback_index),
                    "task_type": task_type,
                    "correct": result.get("label") == "correct",
                }
            )

        scored_turns.sort(key=lambda item: item["sort_key"])
        for segment_name, segment_start, segment_end in split_turn_segments(len(scored_turns)):
            counts = {
                task_type: {"correct_count": 0, "total_count": 0}
                for task_type in SEGMENT_TASK_TYPES
            }
            for scored_turn in scored_turns[segment_start:segment_end]:
                increment_segment_counts(counts, scored_turn["task_type"], scored_turn["correct"])

            for task_type in SEGMENT_TASK_TYPES:
                correct_count = counts[task_type]["correct_count"]
                total_count = counts[task_type]["total_count"]
                rows.append(
                    {
                        "profile": profile,
                        "dataset_name": dataset_name,
                        "segment": segment_name,
                        "segment_start": segment_start,
                        "segment_end": segment_end,
                        "task_type": task_type,
                        "correct_count": correct_count,
                        "total_count": total_count,
                        "accuracy": safe_divide(correct_count, total_count) if total_count else None,
                    }
                )

    return rows


def turn_segment_accuracy_across_profiles_rows(rows_by_profile: list[dict[str, Any]]) -> list[dict[str, Any]]:
    accuracies: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows_by_profile:
        accuracy = row.get("accuracy")
        total_count = row.get("total_count")
        if (
            isinstance(accuracy, (int, float))
            and not isinstance(accuracy, bool)
            and isinstance(total_count, int)
            and total_count > 0
        ):
            accuracies[(str(row["segment"]), str(row["task_type"]))].append(float(accuracy))

    rows = []
    for segment in TURN_SEGMENTS:
        for task_type in SEGMENT_TASK_TYPES:
            summary = numeric_summary(accuracies.get((segment, task_type), []))
            rows.append(
                {
                    "segment": segment,
                    "task_type": task_type,
                    "profile_count": summary["count"],
                    "mean_accuracy": summary["mean"],
                    "min": summary["min"],
                    "max": summary["max"],
                    "variance_population": summary["variance_population"],
                    "median": summary["median"],
                }
            )
    return rows


def nested_score_frequency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        payload[row["profile"]][row["key"]][row["value"]] = {
            "count": row["count"],
            "rate": row["rate"],
        }
    return json.loads(json.dumps(payload, ensure_ascii=False))


def write_summary_md(
    path: Path,
    score_dir: Path,
    output_prefix: str,
    metric_profiles: list[dict[str, Any]],
    score_profiles: dict[str, dict[str, Any]],
    metric_stats: list[dict[str, Any]],
    score_numeric_rows: list[dict[str, Any]],
    frequency_rows: list[dict[str, Any]],
    metric_frequency_rows: list[dict[str, Any]],
) -> None:
    accuracy_keys = [
        "total_info.task.accuracy",
        "total_info.session.accuracy",
        "task_type_info.Single-Tool.accuracy",
        "task_type_info.Multi-Tool.accuracy",
        "task_type_info.Clarify.accuracy",
        "task_type_info.Chat.accuracy",
    ]
    metric_lookup = {row["key"]: row for row in metric_stats}
    raw_correct_rates = {
        row["profile"]: row["rate"]
        for row in frequency_rows
        if row["key"] == "label" and row["value"] == "correct"
    }
    metric_correct_rates = {
        row["profile"]: row["rate"]
        for row in metric_frequency_rows
        if row["key"] == "total_info.task" and row["value"] == "correct"
    }
    metric_total_counts = {
        row["profile"]: row["total"]
        for row in metric_frequency_rows
        if row["key"] == "total_info.task" and row["value"] == "correct"
    }
    metric_lookup_by_profile = {item["profile"]: item for item in metric_profiles}

    lines = [
        "# Score Statistics",
        "",
        f"- Score dir: `{score_dir}`",
        f"- Output prefix: `{output_prefix}`",
        f"- Profiles with metric files: {len(metric_profiles)}",
        f"- Profiles with score files: {len(score_profiles)}",
        "",
        "## Metric Keys Across Profiles",
        "",
        "| Key | Profiles | Mean | Variance(pop) | Min | Max |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in accuracy_keys:
        row = metric_lookup.get(key)
        if not row:
            continue
        lines.append(
            f"| {key} | {row['profile_count']} | {row['mean']:.6f} | "
            f"{row['variance_population']:.6f} | {row['min']:.6f} | {row['max']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Profiles",
            "",
            "| Profile | Scored turns | Task accuracy | Raw label correct rate | Raw parsed turns | Inference completed | Error type |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for profile, payload in score_profiles.items():
        raw_rate = raw_correct_rates.get(profile)
        metric_payload = metric_lookup_by_profile.get(profile, {})
        raw_rate_text = f"{raw_rate:.6f}" if raw_rate is not None else "N/A"
        completed = metric_payload.get("inference_completed")
        completed_text = "" if completed is None else str(completed)
        lines.append(
            "| "
            f"{profile} | "
            f"{metric_total_counts.get(profile, 0)} | "
            f"{metric_correct_rates.get(profile, 0.0):.6f} | "
            f"{raw_rate_text} | "
            f"{payload['result_count']} | "
            f"{completed_text} | "
            f"{payload.get('error_type') or metric_payload.get('error_type') or ''} |"
        )

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- `{output_prefix}_summary.json`",
            f"- `{output_prefix}_summary_average.json`",
            f"- `{output_prefix}_summary.md`",
            f"- `{output_prefix}_metric_by_profile.csv`",
            f"- `{output_prefix}_metric_key_stats.csv`",
            f"- `{output_prefix}_metric_outcome_frequency_by_profile.csv`",
            f"- `{output_prefix}_score_frequency_by_profile.csv`",
            f"- `{output_prefix}_score_numeric_stats_by_profile.csv`",
            f"- `{output_prefix}_score_key_stats_across_profiles.csv`",
            f"- `{output_prefix}_turn_segment_accuracy_by_profile.csv`",
            f"- `{output_prefix}_turn_segment_accuracy_across_profiles.csv`",
            f"- `{output_prefix}_raw_score_files.csv`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def infer_dataset_name(score_dir: Path) -> str:
    return score_dir.parent.name


def infer_model_name(score_dir: Path) -> str:
    return score_dir.name


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-dir", type=Path, default=DEFAULT_SCORE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument(
        "--max-score-file-mb",
        type=float,
        default=64.0,
        help="Only parse raw *_score.jsonl files up to this size. Use -1 to parse all.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    score_dir = args.score_dir.resolve()
    output_dir = args.output_dir.resolve()
    dataset_name = args.dataset_name or infer_dataset_name(score_dir)
    model_name = args.model_name or infer_model_name(score_dir)
    output_prefix = f"{dataset_name}_{model_name}"

    if not score_dir.exists():
        raise FileNotFoundError(f"score dir does not exist: {score_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_profiles = load_metric_profiles(score_dir)
    score_profiles = load_score_profiles(score_dir, args.max_score_file_mb)

    metric_rows, metric_fields = metric_by_profile_rows(metric_profiles)
    metric_stats = metric_key_stats_rows(metric_profiles)
    metric_frequency_rows = metric_outcome_frequency_rows(metric_profiles)
    frequency_rows = score_frequency_rows(score_profiles)
    score_numeric_rows = score_numeric_profile_rows(score_profiles)
    score_across_profile_rows = score_key_stats_across_profiles(score_numeric_rows)
    raw_score_rows = raw_score_file_rows(score_profiles)
    turn_segment_rows = turn_segment_accuracy_by_profile_rows(score_dir, args.max_score_file_mb)
    turn_segment_across_profile_rows = turn_segment_accuracy_across_profiles_rows(turn_segment_rows)
    metric_average_summary = average_metric_summary(metric_profiles, dataset_name, model_name)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "score_dir": str(score_dir),
        "dataset_name": dataset_name,
        "model_name": model_name,
        "profile_count": len(set(score_profiles) | {item["profile"] for item in metric_profiles}),
        "metric_profile_count": len(metric_profiles),
        "score_profile_count": len(score_profiles),
        "max_score_file_mb": args.max_score_file_mb,
        "raw_score_files_skipped": sum(1 for row in raw_score_rows if row["skipped_raw_parse"]),
        "profiles": sorted(set(score_profiles) | {item["profile"] for item in metric_profiles}, key=profile_sort_key),
        "metric_key_stats": {row["key"]: row for row in metric_stats},
        "metric_outcome_frequency_by_profile": nested_score_frequency(metric_frequency_rows),
        "score_numeric_stats_by_profile": {
            f"{row['profile']}::{row['key']}": row for row in score_numeric_rows
        },
        "score_frequency_by_profile": nested_score_frequency(frequency_rows),
        "turn_segment_accuracy_across_profiles": turn_segment_across_profile_rows,
    }

    write_csv(output_dir / f"{output_prefix}_metric_by_profile.csv", metric_rows, metric_fields)
    write_csv(
        output_dir / f"{output_prefix}_metric_key_stats.csv",
        metric_stats,
        [
            "key",
            "profile_count",
            "missing_profile_count",
            "count",
            "min",
            "max",
            "mean",
            "variance_population",
            "variance_sample",
            "stdev_population",
            "stdev_sample",
            "median",
            "p25",
            "p75",
            "sum",
        ],
    )
    write_csv(
        output_dir / f"{output_prefix}_metric_outcome_frequency_by_profile.csv",
        metric_frequency_rows,
        ["profile", "dataset_name", "key", "value", "count", "total", "rate"],
    )
    write_csv(
        output_dir / f"{output_prefix}_score_frequency_by_profile.csv",
        frequency_rows,
        ["profile", "dataset_name", "key", "value", "count", "total", "rate"],
    )
    write_csv(
        output_dir / f"{output_prefix}_score_numeric_stats_by_profile.csv",
        score_numeric_rows,
        [
            "profile",
            "dataset_name",
            "key",
            "count",
            "min",
            "max",
            "mean",
            "variance_population",
            "variance_sample",
            "stdev_population",
            "stdev_sample",
            "median",
            "p25",
            "p75",
            "sum",
        ],
    )
    write_csv(
        output_dir / f"{output_prefix}_raw_score_files.csv",
        raw_score_rows,
        [
            "profile",
            "dataset_name",
            "score_file",
            "score_file_size_bytes",
            "skipped_raw_parse",
            "skip_reason",
            "result_count",
        ],
    )
    write_csv(
        output_dir / f"{output_prefix}_score_key_stats_across_profiles.csv",
        score_across_profile_rows,
        [
            "key",
            "statistic",
            "count",
            "min",
            "max",
            "mean",
            "variance_population",
            "variance_sample",
            "stdev_population",
            "stdev_sample",
            "median",
            "p25",
            "p75",
            "sum",
        ],
    )
    write_csv(
        output_dir / f"{output_prefix}_turn_segment_accuracy_by_profile.csv",
        turn_segment_rows,
        [
            "profile",
            "dataset_name",
            "segment",
            "segment_start",
            "segment_end",
            "task_type",
            "correct_count",
            "total_count",
            "accuracy",
        ],
    )
    write_csv(
        output_dir / f"{output_prefix}_turn_segment_accuracy_across_profiles.csv",
        turn_segment_across_profile_rows,
        [
            "segment",
            "task_type",
            "profile_count",
            "mean_accuracy",
            "min",
            "max",
            "variance_population",
            "median",
        ],
    )
    (output_dir / f"{output_prefix}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / f"{output_prefix}_summary_average.json").write_text(
        json.dumps(metric_average_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_summary_md(
        output_dir / f"{output_prefix}_summary.md",
        score_dir,
        output_prefix,
        metric_profiles,
        score_profiles,
        metric_stats,
        score_numeric_rows,
        frequency_rows,
        metric_frequency_rows,
    )

    print(f"Wrote score statistics to {output_dir} with prefix {output_prefix}")


if __name__ == "__main__":
    main()
