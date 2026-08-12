import json
import random
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path


TASK_TYPE_MAP = {
    "单工具": "Single-Tool",
    "Single-Tool": "Single-Tool",
    "多工具": "Multi-Tool",
    "Multi-Tool": "Multi-Tool",
    "Parallel Multi-Tool": "Parallel Multi-Tool",
    "Sequential Multi-Tool": "Sequential Multi-Tool",
    "Mixed Multi-Tool": "Mixed Multi-Tool",
    "澄清": "Clarify",
    "Clarification": "Clarify",
    "Clarify": "Clarify",
    "闲聊": "Chat",
    "Chit-Chat": "Chat",
    "Chat": "Chat",
}

TURN_SUBTYPE_MAP = {
    "指代": "Coreferential Reference",
    "Coreferential Reference": "Coreferential Reference",
    "省略成分": "Partial Information",
    "Partial Information": "Partial Information",
    "长期记忆": "Long-Range Dependency",
    "Long-Term Memory": "Long-Range Dependency",
    "Long-Range Dependency": "Long-Range Dependency",
    "Cross-Topic": "Cross-Topic",
}


def sanitize_name(name):
    return name.replace("/", "_")


def result_file_name(dataset_name, model_name):
    return f"{dataset_name}__{sanitize_name(model_name)}_result.jsonl"


def score_file_name(dataset_name, model_name):
    return f"{dataset_name}__{sanitize_name(model_name)}_score.jsonl"


def metric_file_name(dataset_name, model_name):
    return f"{dataset_name}__{sanitize_name(model_name)}_metric.json"


def discover_dataset_files(data_dir):
    return sorted(Path(data_dir).glob("*.jsonl"))


def load_jsonl(file_path):
    with open(file_path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _non_empty(value):
    return value is not None and value != [] and value != ""


def _field(entry, primary_key, fallback_key):
    value = entry.get(primary_key)
    if _non_empty(value):
        return value
    return entry.get(fallback_key)


def parse_env_datetime(env_info):
    if not env_info:
        return datetime.max
    match = re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", env_info)
    if not match:
        return datetime.max
    return datetime.strptime(match.group(0), "%Y-%m-%d %H:%M:%S")


def dedupe_and_sort_topics(topics, seed):
    groups = defaultdict(list)
    for topic in topics:
        groups[parse_env_datetime(topic.get("env_info"))].append(topic)

    rng = random.Random(seed)
    selected_topics = []
    for env_time, grouped_topics in groups.items():
        grouped_topics = sorted(grouped_topics, key=lambda item: item.get("id", ""))
        selected_topics.append(rng.choice(grouped_topics))

    return sorted(selected_topics, key=lambda item: (parse_env_datetime(item.get("env_info")), item.get("id", "")))


def normalize_task_type(task_type):
    return TASK_TYPE_MAP.get(task_type, task_type)


def normalize_turn_subtype(turn_subtype):
    return TURN_SUBTYPE_MAP.get(turn_subtype, turn_subtype)


def build_cross_topic_entry(dataset_file, seed=42):
    raw_topics = load_jsonl(dataset_file)
    topics = dedupe_and_sort_topics(raw_topics, seed)
    dataset_name = Path(dataset_file).stem

    tasks = []
    answer_lists = []
    task_types = []
    turn_subtypes = []
    turn_tools = []
    turn_env_infos = []
    turn_metadata = []

    for topic_index, topic in enumerate(topics):
        topic_id = topic.get("id", f"{dataset_name}_topic_{topic_index}")
        topic_env_info = topic.get("env_info") or topic.get("english_env_info")
        topic_tools = _field(topic, "tools", "english_tools") or []
        topic_tasks = _field(topic, "tasks", "english_tasks") or []
        topic_answer_lists = _field(topic, "answer_list", "english_answer_list") or []
        topic_task_types = _field(topic, "task_types", "english_task_types") or []
        topic_turn_subtypes = _field(topic, "turn_subtypes", "english_turn_subtypes") or []

        if not (len(topic_tasks) == len(topic_answer_lists) == len(topic_task_types)):
            raise ValueError(
                f"{dataset_file} topic {topic_id} has inconsistent tasks, answer_list, and task_types lengths"
            )

        for turn_index, (task, answer_list, task_type) in enumerate(
                zip(topic_tasks, topic_answer_lists, topic_task_types)
        ):
            global_turn_index = len(tasks)
            if global_turn_index > 0:
                if turn_index == 0:
                    turn_subtypes.append("Cross-Topic")
                else:
                    turn_subtype = topic_turn_subtypes[turn_index - 1]
                    turn_subtypes.append(normalize_turn_subtype(turn_subtype))

            tasks.append(task)
            answer_lists.append(answer_list)
            task_types.append(normalize_task_type(task_type))
            turn_tools.append(topic_tools)
            turn_env_infos.append(topic_env_info)
            turn_metadata.append(
                {
                    "topic_id": topic_id,
                    "topic_env_info": topic_env_info,
                    "topic_source_id": topic_id,
                    "topic_index": topic_index,
                    "turn_index_in_topic": turn_index,
                    "is_cross_topic_boundary": global_turn_index > 0 and turn_index == 0,
                }
            )

    return {
        "id": dataset_name,
        "dataset_name": dataset_name,
        "source_file": str(Path(dataset_file).resolve()),
        "num_topics_original": len(raw_topics),
        "num_topics_after_dedup": len(topics),
        "num_turns": len(tasks),
        "tasks": tasks,
        "answer_list": answer_lists,
        "task_types": task_types,
        "turn_subtypes": turn_subtypes,
        "turn_tools": turn_tools,
        "turn_env_infos": turn_env_infos,
        "turn_metadata": turn_metadata,
        "english_env_info": turn_env_infos[0] if turn_env_infos else None,
        "english_tools": turn_tools[0] if turn_tools else [],
        "english_tasks": tasks,
        "english_answer_list": answer_lists,
        "english_task_types": task_types,
        "english_turn_subtypes": turn_subtypes,
    }


def build_cross_topic_entries(data_dir, seed=42, selected_ids=None):
    entries = []
    selected_ids = set(selected_ids or [])
    for dataset_file in discover_dataset_files(data_dir):
        entry = build_cross_topic_entry(dataset_file, seed=seed)
        if selected_ids and entry["id"] not in selected_ids:
            continue
        entries.append(entry)
    return entries
