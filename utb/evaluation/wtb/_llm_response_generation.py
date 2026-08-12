import json
import argparse
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from tqdm import tqdm

current_path_list = os.getcwd().split("/")[:-2]
current_path = "/".join(current_path_list)
print(f"current_path: {current_path}\n", flush=True)
sys.path.append(current_path)

from wtb.constant import DATA_DIR, DEFAULT_RANDOM_SEED, PROJECT_ROOT, RESULT_PATH, TEST_IDS_TO_GENERATE_PATH
from wtb.utils import load_file, sort_key
from wtb.dataset_utils import build_cross_topic_entries, result_file_name
from wtb.model_handler.handler_map import HANDLER_MAP


RETRY_LIMIT = 1
                                                                                                             
RETRY_DELAY = 20                    


def get_involved_test_entries(run_ids, data_dir, random_seed):
    selected_ids = None
    if run_ids:
        with open(TEST_IDS_TO_GENERATE_PATH) as f:
            test_ids = json.load(f)
        if len(test_ids) != 0:
            selected_ids = test_ids

    return build_cross_topic_entries(data_dir, seed=random_seed, selected_ids=selected_ids)


def result_path_for_test_case(result_dir, model_name, test_case):
    model_name_dir = model_name.replace("/", "_")
    dataset_name = test_case["dataset_name"]
    return result_dir / model_name_dir / result_file_name(dataset_name, model_name)


def collect_test_cases(args, model_name, all_test_entries_involved):
    test_cases_to_generate = []
    for test_case in all_test_entries_involved:
        result_file_path = result_path_for_test_case(args.result_dir, model_name, test_case)
        existing_result = []
        if result_file_path.exists():
                                                                       
            if not args.allow_overwrite:
                existing_result.extend(load_file(result_file_path))
                                                                                                                                      
            elif not args.run_ids:
                result_file_path.unlink()
                                                                                    
            else:
                pass

        existing_ids = [entry["id"] for entry in existing_result]
        if test_case["id"] not in existing_ids:
            test_cases_to_generate.append(test_case)

    return sorted(test_cases_to_generate, key=sort_key)


def build_handler(model_name, temperature):
    handler = HANDLER_MAP[model_name](model_name, temperature)
    return handler


def generate_results(args, model_name, test_cases_total):
    handler = build_handler(model_name, args.temperature)

    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        with tqdm(
            total=len(test_cases_total), desc=f"Generating results for {model_name}"
        ) as pbar:

            futures = []
            for test_case in test_cases_total:
                future = executor.submit(
                    multi_threaded_inference,
                    handler,
                    model_name,
                    test_case
                )
                futures.append(future)

            for future in as_completed(futures):
                result = future.result()
                handler.write(
                    result,
                    result_dir=args.result_dir,
                    update_mode=args.run_ids,
                    dataset_name=result.get("dataset_name")
                )                                                                                                       
                pbar.update()


def multi_threaded_inference(handler, model_name, test_case):
    retry_count = 0

    while True:
        try:
            result = handler.inference(deepcopy(test_case))
            break                          
        except Exception as e:
                                                                                                                                     
            if retry_count < RETRY_LIMIT and (
                "rate limit reached" in str(e).lower()
                or (hasattr(e, "status_code") and e.status_code in {429, 503, 500})
            ):
                print(
                    f"Rate limit reached. Sleeping for 65 seconds. Retry {retry_count + 1}/{RETRY_LIMIT}"
                )
                time.sleep(RETRY_DELAY)
                retry_count += 1
            else:
                                                                                                    
                                                                                         
                                                                                                       
                                                                                                          
                print("-" * 100)
                print(
                    "❗️❗️ Error occurred during inference. Maximum reties reached for rate limit or other error. Continuing to next test case."
                )
                print(f"❗️❗️ Test case ID: {test_case['id']}, Error: {str(e)}")
                traceback.print_exc()
                print("-" * 100)

                return {
                    "id": test_case["id"],
                    "model_name": model_name,
                    "dataset_name": test_case["dataset_name"],
                    "source_file": test_case["source_file"],
                    "num_topics_after_dedup": test_case["num_topics_after_dedup"],
                    "num_turns": test_case["num_turns"],
                    "inference_completed": False,
                    "evaluated_turns": 0,
                    "failed_task_idx": None,
                    "failed_step": None,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "api_error": {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "status_code": getattr(e, "status_code", None),
                        "body": getattr(e, "body", None),
                    },
                    "result": f"Error during inference: {str(e)}"
                }

    failure_result = next((item for item in result if isinstance(item, dict) and item.get("inference_error")), None)
    result_to_write = {
        "id": test_case["id"],
        "model_name": model_name,
        "dataset_name": test_case["dataset_name"],
        "source_file": test_case["source_file"],
        "num_topics_after_dedup": test_case["num_topics_after_dedup"],
        "num_turns": test_case["num_turns"],
        "inference_completed": failure_result is None and len(result) == test_case["num_turns"],
        "evaluated_turns": len(result),
        "failed_task_idx": failure_result.get("failed_task_idx") if failure_result else None,
        "failed_step": failure_result.get("failed_step") if failure_result else None,
        "error_type": failure_result.get("error_type") if failure_result else None,
        "error_message": failure_result.get("error_message") if failure_result else None,
        "api_error": failure_result.get("api_error") if failure_result else None,
        "result": result
    }

    return result_to_write


def main(args):
    if type(args.model) != list:
        args.model = [args.model]

    if args.data_dir is not None:
        args.data_dir = PROJECT_ROOT / args.data_dir
    else:
        args.data_dir = DATA_DIR

    all_test_entries_involved = get_involved_test_entries(args.run_ids, args.data_dir, args.random_seed)

    print(f"Generating results for {args.model}")
    if args.run_ids:
        print("Running specific test cases.")
    else:
        print("Running full test cases.")

    if args.result_dir is not None:
        args.result_dir = PROJECT_ROOT / args.result_dir
    else:
        args.result_dir = RESULT_PATH

    for model_name in args.model:
        test_cases_total = collect_test_cases(
            args,
            model_name,
            all_test_entries_involved
        )

        if len(test_cases_total) == 0:
            print(
                f"All selected test cases have been previously generated for {model_name}. No new test cases to generate."
            )
        else:
            generate_results(args, model_name, test_cases_total)


def get_args():
    parser = argparse.ArgumentParser()
                                                 
    parser.add_argument("--model", type=str, default="deepseek-chat", nargs="+")

                                                     
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-threads", default=1, type=int)
    parser.add_argument("--result-dir", default=None, type=str)
    parser.add_argument("--data-dir", default=None, type=str)
    parser.add_argument("--run-ids", action="store_true", default=False)
    parser.add_argument("--allow-overwrite", action="store_true", default=False)
    parser.add_argument("--random-seed", default=DEFAULT_RANDOM_SEED, type=int)

    args = parser.parse_args()
    return args
