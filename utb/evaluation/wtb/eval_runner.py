import argparse
import json

from tqdm import tqdm
from wtb.constant import DATA_DIR, DEFAULT_RANDOM_SEED, PROJECT_ROOT, RESULT_PATH, SCORE_PATH
from wtb.utils import load_file, write_list_of_dicts_to_file, write_dicts_to_file
from wtb.checker_utils import ToolArgsChecker
from wtb.dataset_utils import build_cross_topic_entries, metric_file_name, sanitize_name, score_file_name


def params_checker(result):
    tool_args_checker = ToolArgsChecker()
    action_arguments_label = "correct"
    inference_log = result["inference_log"]
    for key in inference_log.keys():
        if not key.startswith("step_"):
            continue

                                                                                        
        if action_arguments_label == "error":
            result["action_name_label"] = "correct"
            del inference_log[key]
            break

        step_data = inference_log[key]
        inference_input = step_data["inference_input"]
        inference_output = step_data["inference_output"]

        current_action_name_label = inference_output["current_action_name_label"]
        if current_action_name_label == "error":
            break
        inference_answer = step_data["inference_answer"]

                                                                                                                                 
        candidate_answer_function = inference_answer["candidate_0_answer_function_list"]
        assert "candidate_1_answer_function_list" not in inference_answer

        tools = inference_input["tools"]
        predict_tool_calls = inference_output["tool_calls"]
        answer_actions = candidate_answer_function["action"]

        if answer_actions[0]["name"] in ["prepare_to_answer", "ask_user_for_required_parameters"]:
            continue

        assert len(predict_tool_calls) == len(answer_actions)

        predict_tool_calls = sorted(predict_tool_calls, key=lambda x: (x["function"]["name"], x["function"]["arguments"]))
        predict_actions = [item["function"] for item in predict_tool_calls]
        inference_output["tool_calls"] = predict_tool_calls

        answer_actions = sorted(answer_actions, key=lambda x: (x["name"], x["arguments"]))
        candidate_answer_function["action"] = answer_actions

        current_action_arguments_label = "correct"
        arguments_check_result = []
        for predict_action, answer_action in zip(predict_actions, answer_actions):
            predict_name = predict_action["name"]
            predict_arguments = predict_action["arguments"]

            answer_name = answer_action["name"]
            answer_arguments = answer_action["arguments"]

            assert predict_name == answer_name

            check_result = tool_args_checker.check(tools, predict_name, predict_arguments, answer_arguments)
            arguments_check_result.append(check_result)
            if check_result != "correct":
                current_action_arguments_label = "error"
                action_arguments_label = "error"

        step_data["inference_output"]["current_action_arguments_label"] = current_action_arguments_label
        if current_action_arguments_label == "error":
            step_data["inference_output"]["current_action_arguments_check_result"] = arguments_check_result

    action_name_label = result["action_name_label"]
    if action_name_label == "correct":
        items = list(result.items())
        items.insert(1, ("action_arguments_label", action_arguments_label))
        result.clear()
        result.update(items)
        if action_arguments_label == "error" and result["is_optimal"] is True:
            result["is_optimal"] = False

    return action_name_label, action_arguments_label


def add_accuracy_field(variable_name, info_dict):
    """
    遍历字典，计算 accuracy 并将其插入到每个子字典的第一个位置。
    原地修改 info_dict 中的子对象。
    """
    for key, stats in info_dict.items():
        correct = stats.get("correct_count", 0)
        total = stats.get("total_count", 0)

                                
        acc = correct / total if total > 0 else 0.0

                     
                                  
        new_content = {"accuracy": acc}
                  
        new_content.update(stats)

                               
        stats.clear()
        stats.update(new_content)
    print(f"{variable_name}:")
    print(json.dumps(info_dict, ensure_ascii=False, indent=4))
    print("\n" + "=" * 100 + "\n")


def add_rate_field(variable_name, info_dict):
    """
    遍历字典，计算 accuracy 并将其插入到每个子字典的第一个位置。
    原地修改 info_dict 中的子对象。
    """
    for key, stats in info_dict.items():
        correct = stats.get("complete_step", 0)
        total = stats.get("total_step", 0)

                                
        acc = correct / total if total > 0 else 0.0

                     
                                  
        new_content = {"rate": acc}
                  
        new_content.update(stats)

                               
        stats.clear()
        stats.update(new_content)
    print(f"{variable_name}:")
    print(json.dumps(info_dict, ensure_ascii=False, indent=4) + "\n")


def _ensure_counter(info_dict, key):
    if key not in info_dict:
        info_dict[key] = {"correct_count": 0, "total_count": 0}


def _ensure_progress_counter(info_dict, key):
    if key not in info_dict:
        info_dict[key] = {"complete_step": 0, "total_step": 0}


def calc_accuracy(model_name, all_test_entries, score_results):
    test_entry_by_id = {entry["id"]: entry for entry in all_test_entries}
    total_info = {
        "task": {"correct_count": 0, "total_count": 0},
        "session": {"correct_count": 0, "total_count": 0}
    }
    task_type_info = {
        "Single-Tool": {"correct_count": 0, "total_count": 0},
        "Multi-Tool": {"correct_count": 0, "total_count": 0},
        "Parallel Multi-Tool": {"correct_count": 0, "total_count": 0},
        "Sequential Multi-Tool": {"correct_count": 0, "total_count": 0},
        "Mixed Multi-Tool": {"correct_count": 0, "total_count": 0},
        "Clarify": {"correct_count": 0, "total_count": 0},
        "Chat": {"correct_count": 0, "total_count": 0}
    }
    layer_info = {
        "0": {"correct_count": 0, "total_count": 0},
        "1": {"correct_count": 0, "total_count": 0},
        "2": {"correct_count": 0, "total_count": 0},
        "3": {"correct_count": 0, "total_count": 0}
    }
    turn_subtype_info = {
        "First Turn": {"correct_count": 0, "total_count": 0},
        "Subsequent Turn": {"correct_count": 0, "total_count": 0},
        "Coreferential Reference": {"correct_count": 0, "total_count": 0},
        "Partial Information": {"correct_count": 0, "total_count": 0},
        "Long-Range Dependency": {"correct_count": 0, "total_count": 0},
        "Cross-Topic": {"correct_count": 0, "total_count": 0}
    }
    progress_info = {
        "Total": {"complete_step": 0, "total_step": 0},
        "Sequential Multi-Tool": {"complete_step": 0, "total_step": 0},
        "Mixed Multi-Tool": {"complete_step": 0, "total_step": 0}
    }
    optimal_info = {
        "Total": {"correct_count": 0, "total_count": 0},
        "Parallel Multi-Tool": {"correct_count": 0, "total_count": 0},
        "Mixed Multi-Tool": {"correct_count": 0, "total_count": 0}
    }
    for score_result in score_results:
        id_ = score_result["id"]
        results = score_result["results"]
        test_entry = test_entry_by_id[id_]
        english_task_types = test_entry["task_types"]
        english_turn_subtypes = test_entry["turn_subtypes"]
        answer_list = test_entry["answer_list"]

        total_info["session"]["total_count"] += 1
        session_correct = True
        if not results:
            session_correct = False
        for i, result in enumerate(results):
            label = result["label"]
            is_optimal = result["is_optimal"]
            task_type = english_task_types[i]
            if i == 0:
                turn_subtype = "First Turn"
            else:
                turn_subtype = english_turn_subtypes[i - 1]
                turn_subtype_info["Subsequent Turn"]["total_count"] += 1

            total_info["task"]["total_count"] += 1
            _ensure_counter(task_type_info, task_type)
            _ensure_counter(layer_info, str(i))
            _ensure_counter(turn_subtype_info, turn_subtype)
            if task_type != "Multi-Tool" and "Multi-Tool" in task_type:
                task_type_info["Multi-Tool"]["total_count"] += 1
            task_type_info[task_type]["total_count"] += 1
            layer_info[str(i)]["total_count"] += 1
            turn_subtype_info[turn_subtype]["total_count"] += 1

            if label == "correct":
                total_info["task"]["correct_count"] += 1
                if task_type != "Multi-Tool" and "Multi-Tool" in task_type:
                    task_type_info["Multi-Tool"]["correct_count"] += 1
                task_type_info[task_type]["correct_count"] += 1
                layer_info[str(i)]["correct_count"] += 1
                turn_subtype_info[turn_subtype]["correct_count"] += 1
                if i > 0:
                    turn_subtype_info["Subsequent Turn"]["correct_count"] += 1
            else:
                session_correct = False

            if task_type in ["Parallel Multi-Tool", "Mixed Multi-Tool"]:
                _ensure_counter(optimal_info, task_type)
                optimal_info["Total"]["total_count"] += 1
                optimal_info[task_type]["total_count"] += 1
                if is_optimal:
                    optimal_info["Total"]["correct_count"] += 1
                    optimal_info[task_type]["correct_count"] += 1

            if task_type in ["Sequential Multi-Tool", "Mixed Multi-Tool"]:
                _ensure_progress_counter(progress_info, task_type)
                inference_log = result["inference_log"]
                complete_step = len([k for k in inference_log.keys() if k.startswith("step")])
                answer = answer_list[i]
                total_step = len(answer)
                progress_info["Total"]["total_step"] += total_step
                progress_info["Total"]["complete_step"] += complete_step
                progress_info[task_type]["total_step"] += total_step
                progress_info[task_type]["complete_step"] += complete_step

        if session_correct:
            total_info["session"]["correct_count"] += 1

    add_accuracy_field("total_info", total_info)
    add_accuracy_field("task_type_info", task_type_info)
    add_accuracy_field("layer_info", layer_info)
    add_accuracy_field("turn_subtype_info", turn_subtype_info)
    add_accuracy_field("optimal_info", optimal_info)
    add_rate_field("progress_info", progress_info)
    first_score_result = score_results[0] if score_results else {}
    metric_info = {
        "model_name": model_name,
        "dataset_name": all_test_entries[0].get("dataset_name") if all_test_entries else None,
        "source_file": all_test_entries[0].get("source_file") if all_test_entries else None,
        "num_topics_after_dedup": all_test_entries[0].get("num_topics_after_dedup") if all_test_entries else 0,
        "num_turns": all_test_entries[0].get("num_turns") if all_test_entries else 0,
        "num_scored_turns": sum(len(score_result.get("results", [])) for score_result in score_results),
        "inference_completed": first_score_result.get("inference_completed"),
        "failed_task_idx": first_score_result.get("failed_task_idx"),
        "failed_step": first_score_result.get("failed_step"),
        "error_type": first_score_result.get("error_type"),
        "error_message": first_score_result.get("error_message"),
        "api_error": first_score_result.get("api_error"),
        "total_info": total_info,
        "task_type_info": task_type_info,
        "layer_info": layer_info,
        "turn_subtype_info": turn_subtype_info,
        "optimal_info": optimal_info,
        "progress_info": progress_info
    }
    return metric_info


def runner(model_names, result_dir, score_dir, data_dir, random_seed):
    all_test_entries = build_cross_topic_entries(data_dir, seed=random_seed)
    test_entry_by_dataset = {entry["dataset_name"]: entry for entry in all_test_entries}
                                             
    entries = result_dir.iterdir()

                                   
    subdirs = [entry for entry in entries if entry.is_dir()]

                                
    for subdir in tqdm(subdirs, desc="Number of models evaluated"):
        model_name = subdir.relative_to(result_dir).name

        selected_model_dirs = {sanitize_name(name) for name in model_names} if model_names is not None else None
        if selected_model_dirs is not None and model_name not in selected_model_dirs:
            continue

        print(f"Model: {model_name}")

        for model_result_jsonl in sorted(subdir.glob("*_result.jsonl")):
            model_results = load_file(model_result_jsonl)
            if not model_results:
                continue

            first_result = model_results[0]
            dataset_name = first_result.get("dataset_name")
            effective_model_name = first_result.get("model_name", model_name)
            if dataset_name not in test_entry_by_dataset:
                print(f"Skip {model_result_jsonl}: dataset {dataset_name} not found in {data_dir}")
                continue

            score_results = []
            for model_result in model_results:
                id_ = model_result["id"]
                results = model_result["result"]
                if not isinstance(results, list):
                    score_results.append(
                        {
                            "id": id_,
                            "model_name": effective_model_name,
                            "dataset_name": dataset_name,
                            "source_file": model_result.get("source_file"),
                            "inference_completed": model_result.get("inference_completed", False),
                            "evaluated_turns": model_result.get("evaluated_turns", 0),
                            "failed_task_idx": model_result.get("failed_task_idx"),
                            "failed_step": model_result.get("failed_step"),
                            "error_type": model_result.get("error_type"),
                            "error_message": model_result.get("error_message"),
                            "api_error": model_result.get("api_error"),
                            "results": [],
                            "error": results
                        }
                    )
                    continue

                for result in results:
                    action_name_label, action_arguments_label = params_checker(result)
                    if action_name_label == "error" or action_arguments_label == "error":
                        label = "error"
                    else:
                        label = "correct"
                    items = list(result.items())
                    items.insert(0, ("label", label))
                    result.clear()
                    result.update(items)

                score_results.append(
                    {
                        "id": id_,
                        "model_name": model_result.get("model_name", effective_model_name),
                        "dataset_name": dataset_name,
                        "source_file": model_result.get("source_file"),
                        "inference_completed": model_result.get("inference_completed", len(results) == test_entry_by_dataset[dataset_name]["num_turns"]),
                        "evaluated_turns": model_result.get("evaluated_turns", len(results)),
                        "failed_task_idx": model_result.get("failed_task_idx"),
                        "failed_step": model_result.get("failed_step"),
                        "error_type": model_result.get("error_type"),
                        "error_message": model_result.get("error_message"),
                        "api_error": model_result.get("api_error"),
                        "results": results
                    }
                )

            output_file_dir = score_dir / model_name
            write_list_of_dicts_to_file(score_file_name(dataset_name, effective_model_name), score_results, output_file_dir)

            metric_info = calc_accuracy(effective_model_name, [test_entry_by_dataset[dataset_name]], score_results)
            write_dicts_to_file(metric_file_name(dataset_name, effective_model_name), metric_info, output_file_dir)


def main(model, result_dir, score_dir, data_dir, random_seed):
    if result_dir is None:
        result_dir = RESULT_PATH
    else:
        result_dir = (PROJECT_ROOT / result_dir).resolve()

    if score_dir is None:
        score_dir = SCORE_PATH
    else:
        score_dir = (PROJECT_ROOT / score_dir).resolve()

    if data_dir is None:
        data_dir = DATA_DIR
    else:
        data_dir = (PROJECT_ROOT / data_dir).resolve()

    runner(model, result_dir, score_dir, data_dir, random_seed)


def get_args():
    parser = argparse.ArgumentParser()
                                                 
    parser.add_argument("--model", type=str, default="deepseek-chat", nargs="+")

                                                     
    parser.add_argument("--result-dir", default='./evaluation/result-10-30', type=str)
    parser.add_argument("--score-dir", default='./evaluation/score-10-30', type=str)
    parser.add_argument("--data-dir", default='./evaluation/data-10-30', type=str)
    parser.add_argument("--random-seed", default=DEFAULT_RANDOM_SEED, type=int)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    main(
        args.model,
        args.result_dir,
        args.score_dir,
        args.data_dir,
        args.random_seed,
    )
