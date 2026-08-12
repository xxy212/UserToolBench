import json

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, TimeoutError

try:
    from overrides import final
except ImportError:
    def final(func):
        return func

from wtb.tool_call_graph import ToolCallGraph
from wtb.utils import sort_key, load_file, generate_random_string
from wtb.dataset_utils import result_file_name


class BaseHandler:
    def __init__(self, model_name, temperature):
        self.model_name = model_name
        self.temperature = temperature
        self.model_messages = []
        self.consecutive_tool_messages = True

    def _request_tool_call(self, inference_data):
        raise NotImplementedError

    def _parse_api_response(self, api_response):
        raise NotImplementedError

    def convert_to_tool(self, tools):
        tools = json.dumps(tools, ensure_ascii=False).replace('"type": "float"', '"type": "number"')
        tools = json.loads(tools)
        new_tools = []
        if "claude" in self.model_name:
            for tool in tools:
                tool["inputSchema"] = tool["function"]["parameters"]
                del tool["function"]["parameters"]
                new_tools.append(tool)
            tools = new_tools
        return tools

    def _add_action_observation(self, task, answer_list, consecutive_tool_messages):
        tool_call_graph = ToolCallGraph(answer_list)
        tool_call_graph.add_node_list()
        tool_call_graph.generate_all_path()
        optimal_path = tool_call_graph.optimal_path_list[0]

        current_messages = [{"role": "user", "content": task}]
        for idx_action_list in optimal_path:
            format_action_list = []
            observation_list = []
            for idx in idx_action_list:
                answer = answer_list[idx]
                action = answer["action"]
                action_name = action["name"]
                action_arguments = action["arguments"]
                observation = answer["observation"]
                if action_name == "ask_user_for_required_parameters":
                    assert len(idx_action_list) == 1
                    user_input = answer["user_input"]
                    current_messages.extend([
                        {
                            "role": "assistant",
                            "content": observation
                        },
                        {
                            "role": "user",
                            "content": user_input
                        }
                    ])

                elif action_name == "prepare_to_answer":
                    assert len(idx_action_list) == 1
                    current_messages.extend([
                        {
                            "role": "assistant",
                            "content": observation
                        }
                    ])

                else:
                    format_action_list.append(
                        {
                            "type": "function",
                            "function": {
                                "name": action_name,
                                "arguments": action_arguments
                            }
                        }
                    )
                    observation_list.append(observation)

            if len(format_action_list) > 0:
                current_messages.extend([
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": format_action_list
                    }
                ])
                if consecutive_tool_messages:
                    for observation in observation_list:
                        current_messages.extend([
                            {
                                "role": "tool",
                                "content": observation
                            }
                        ])
                else:
                    current_messages.extend([
                        {
                            "role": "tool",
                            "content": json.dumps(observation_list, ensure_ascii=False)
                        }
                    ])

        return current_messages

    def _convert_to_tool_calls(self, messages):
        tool_call_id_list = []
        new_messages = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            tool_calls = message.get("tool_calls", None)
            if role == "assistant":
                if tool_calls:
                    new_tool_calls = []
                    for tool_call in tool_calls:
                        arguments = tool_call["function"]["arguments"]
                        if isinstance(arguments, dict):
                            tool_call["function"]["arguments"] = json.dumps(arguments, ensure_ascii=False)
                        if "id" not in tool_call:
                            tool_call_id = "toolu_bdrk_" + generate_random_string(24)
                            tool_call["id"] = tool_call_id
                            tool_call_id_list.append(tool_call_id)
                        new_tool_calls.append(tool_call)
                    message["tool_calls"] = new_tool_calls
            elif role == "tool":
                tool_call_id = tool_call_id_list[0]
                tool_call_id_list.pop(0)
                message["tool_call_id"] = tool_call_id
                message["content"] = json.dumps(content, ensure_ascii=False)
            new_messages.append(message)

        return new_messages

    def _pre_messages_processing(self, env_info, current_task, history_tasks, history_answer_lists, consecutive_tool_messages=True):
        messages = [{"role": "system", "content": f"Current Date: {env_info}"}]
        for history_task, history_answer_list in zip(history_tasks, history_answer_lists):
            history_messages = self._add_action_observation(history_task, history_answer_list, consecutive_tool_messages)
            messages.extend(history_messages)
        messages.append({"role": "user", "content": current_task})
        messages = self._convert_to_tool_calls(messages)

        return messages

    def inference(self, test_entry: dict):
        return self.inference_multi_turn(test_entry)

    @final
    def inference_multi_turn(self, test_entry: dict):
        test_entry_id = test_entry["id"]
        env_info = test_entry.get("english_env_info") or test_entry.get("env_info")
        tools = test_entry.get("english_tools") or test_entry.get("tools") or []
        tasks = test_entry.get("english_tasks") or test_entry.get("tasks")
        answer_lists = test_entry.get("english_answer_list") or test_entry.get("answer_list")
        turn_tools = test_entry.get("turn_tools")
        turn_env_infos = test_entry.get("turn_env_infos")
        turn_metadata = test_entry.get("turn_metadata") or [{} for _ in tasks]

        all_task_result_data = []
        for task_idx, (current_task, answer_list) in enumerate(zip(tasks, answer_lists)):
            history_tasks = tasks[:task_idx]
            history_answer_lists = answer_lists[:task_idx]
            current_tools = turn_tools[task_idx] if turn_tools else tools
            current_tools = self.convert_to_tool(current_tools)
            current_env_info = turn_env_infos[task_idx] if turn_env_infos else env_info
            messages = self._pre_messages_processing(current_env_info, current_task, history_tasks, history_answer_lists)

            inference_data = {
                "test_entry_id": test_entry_id,
                "task_idx": task_idx,
                "tools": current_tools,
                "messages": messages,
                "answer_list": answer_list,
                "turn_metadata": turn_metadata[task_idx] if task_idx < len(turn_metadata) else {},
                "env_info": current_env_info,
            }
            result_data = self.inference_and_eval_multi_step(inference_data)
            all_task_result_data.append(result_data)
            if result_data.get("inference_error"):
                break

        return all_task_result_data

    def run_with_timeout(self, func, timeout, *args, **kwargs):
        with ThreadPoolExecutor() as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                result = future.result(timeout=timeout)
                return result
            except TimeoutError:
                raise TimeoutError(f"Function '{func.__name__}' exceeded timeout of {timeout} seconds")

    def _extract_api_error(self, error):
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "status_code": getattr(error, "status_code", None),
            "code": getattr(error, "code", None),
            "param": getattr(error, "param", None),
            "request_id": getattr(error, "request_id", None),
            "body": getattr(error, "body", None),
            "response_body": None,
        }
        response = getattr(error, "response", None)
        if response is not None:
            error_info["status_code"] = error_info["status_code"] or getattr(response, "status_code", None)
            try:
                error_info["response_body"] = response.json()
            except Exception:
                error_info["response_body"] = getattr(response, "text", None)
        return error_info

    def _build_inference_error_result(self, inference_log, latency, input_token_count, output_token_count, task_idx, step, error):
        api_error = self._extract_api_error(error)
        error_type = api_error["error_type"]
        error_message = api_error["error_message"]
        inference_log[f"step_{step}"] = {
            "inference_input": {
                "messages": deepcopy(inference_log.get("messages_for_failed_step", [])),
                "tools": inference_log.get("tools_for_failed_step", [])
            },
            "inference_output": {
                "reasoning_content": None,
                "content": None,
                "tool_calls": None,
                "current_action_name_label": "error",
                "error_reason": f"inference exception: {error_type}: {error_message}",
                "api_error": api_error,
            }
        }
        inference_log.pop("messages_for_failed_step", None)
        inference_log.pop("tools_for_failed_step", None)
        inference_log["inference_error"] = {
            "failed_task_idx": task_idx,
            "failed_step": step,
            "error_type": error_type,
            "error_message": error_message,
            "api_error": api_error,
        }
        return {
            "action_name_label": "error",
            "is_optimal": False,
            "inference_error": True,
            "failed_task_idx": task_idx,
            "failed_step": step,
            "error_type": error_type,
            "error_message": error_message,
            "api_error": api_error,
            "inference_log": inference_log,
            "latency": latency,
            "input_token_count": input_token_count,
            "output_token_count": output_token_count
        }

    @final
    def inference_and_eval_multi_step(self, inference_data):
        '''
        Only the action is evaluated here
        If the action is incorrect, the process will be terminated early to improve evaluation efficiency.
        The correctness of the parameters will be evaluated later in the eval_checker.
        '''
        test_entry_id = inference_data["test_entry_id"]
        task_idx = inference_data["task_idx"]
        tools = inference_data["tools"]
        messages = inference_data["messages"]
        answer_list = inference_data["answer_list"]
        tool_call_graph = ToolCallGraph(answer_list)
        tool_call_graph.add_node_list()
        tool_call_graph.generate_all_path()
              
                                                                      
                                                                          
                                
                                              
                                              

                                                         

        step = 0
        action_name_label = "error"
        predict_result = []
                                                                                         
        inference_log = {
            "task_idx": task_idx,
            "env_info": inference_data.get("env_info"),
            "turn_metadata": inference_data.get("turn_metadata", {}),
            "begin_of_current_task": messages[-1]
        }
        answer_result = []
        latency = []
        input_token_count = []
        output_token_count = []
        while True:
            print("-" * 100, flush=True)
            print(
                f"ID: {test_entry_id.replace('wild_tool_bench_', '')}, Task: {task_idx}, Step: {step}", flush=True
            )
                                           
                                      
                                                                                             
            inference_log["messages_for_failed_step"] = messages
            inference_log["tools_for_failed_step"] = tools
            try:
                api_response, query_latency = self._request_tool_call(inference_data)
                model_response_data = self._parse_api_response(api_response)
            except Exception as e:
                print(
                    f"Error during inference at task {task_idx}, step {step}: {type(e).__name__}: {str(e)}",
                    flush=True
                )
                return self._build_inference_error_result(
                    inference_log,
                    latency,
                    input_token_count,
                    output_token_count,
                    task_idx,
                    step,
                    e
                )
            inference_log.pop("messages_for_failed_step", None)
            inference_log.pop("tools_for_failed_step", None)
            reasoning_content = model_response_data["reasoning_content"]
            content = model_response_data["content"]
            tool_calls = model_response_data["tool_calls"]
            input_token = model_response_data["input_token"]
            output_token = model_response_data["output_token"]
            latency.append(query_latency)
            input_token_count.append(input_token)
            output_token_count.append(output_token)

                                           
                                                                              
                                                          
                                                                                                          

            inference_log[f"step_{step}"] = {
                "inference_input": {
                    "messages": deepcopy(messages),
                    "tools": tools
                },
                "inference_output": {
                    "reasoning_content": reasoning_content,
                    "content": content,
                    "tool_calls": tool_calls
                }
            }

            if tool_calls is None or len(tool_calls) == 0:
                if content is None or content == "":
                    action_name_label = "error"
                    inference_log[f"step_{step}"]["inference_output"].update(
                        {
                            "current_action_name_label": "error",
                            "error_reason": f"tool_calls and content are None or empty"
                        }
                    )
                    break

                else:
                    current_step_function_name_list = tool_call_graph.step_to_function_name_list[step]
                    current_step_function_arguments_list = tool_call_graph.step_to_function_arguments_list[step]
                    current_step_function_observation_list = tool_call_graph.step_to_function_observation_list[step]
                    current_step_user_input_list = tool_call_graph.step_to_user_input_list[step]
                    for i, (answer_function_name_list, answer_function_arguments_list, answer_function_observation_list, answer_user_input_list) in enumerate(
                            zip(
                                current_step_function_name_list,
                                current_step_function_arguments_list,
                                current_step_function_observation_list,
                                current_step_user_input_list
                            )
                    ):
                        if "ask_user_for_required_parameters" in answer_function_name_list:
                            messages.append(
                                {"role": "assistant", "content": content}
                            )

                            assert len(answer_function_observation_list) == 1
                            function_observation = answer_function_observation_list[0]

                            assert len(answer_user_input_list) == 1
                            user_input = answer_user_input_list[0]
                            messages.append(
                                {"role": "user", "content": user_input}
                            )

                            answer_function_list = {"action": [], "observation": function_observation, "user_input": user_input}
                            for answer_function_name, answer_function_arguments in zip(answer_function_name_list, answer_function_arguments_list):
                                answer_function_list["action"].append(
                                    {"arguments": json.dumps(answer_function_arguments, ensure_ascii=False), "name": answer_function_name}
                                )
                            inference_log[f"step_{step}"].setdefault("inference_answer", {})[f"candidate_0_answer_function_list"] = answer_function_list
                            inference_log[f"step_{step}"]["inference_output"]["current_action_name_label"] = "correct"

                            break
                        elif "prepare_to_answer" in answer_function_name_list:
                            assert len(answer_function_observation_list) == 1
                            function_observation = answer_function_observation_list[0]

                            messages.append(
                                {"role": "assistant", "content": content}
                            )
                            action_name_label = "correct"

                            answer_function_list = {"action": [], "observation": function_observation}
                            for answer_function_name, answer_function_arguments in zip(answer_function_name_list, answer_function_arguments_list):
                                answer_function_list["action"].append(
                                    {"arguments": json.dumps(answer_function_arguments, ensure_ascii=False), "name": answer_function_name}
                                )
                            inference_log[f"step_{step}"].setdefault("inference_answer", {})[f"candidate_0_answer_function_list"] = answer_function_list
                            inference_log[f"step_{step}"]["inference_output"]["current_action_name_label"] = "correct"

                            break
                    else:
                        action_name_label = "error"
                                                           
                        for i, (answer_function_name_list, answer_function_arguments_list, answer_function_observation_list, answer_user_input_list) in enumerate(
                                zip(
                                    current_step_function_name_list,
                                    current_step_function_arguments_list,
                                    current_step_function_observation_list,
                                    current_step_user_input_list
                                )
                        ):
                            answer_function_list = {"action": []}
                            for answer_function_name, answer_function_arguments, answer_function_observation, answer_user_input in zip(
                                    answer_function_name_list,
                                    answer_function_arguments_list,
                                    answer_function_observation_list,
                                    answer_user_input_list
                            ):
                                answer_function_list["action"].append(
                                    {"arguments": json.dumps(answer_function_arguments, ensure_ascii=False), "name": answer_function_name}
                                )
                                if answer_function_name == "ask_user_for_required_parameters":
                                    answer_function_list["observation"] = answer_function_observation
                                    answer_function_list["user_input"] = answer_user_input
                                elif answer_function_name == "prepare_to_answer":
                                    answer_function_list["observation"] = answer_function_observation

                            inference_log[f"step_{step}"].setdefault("inference_answer", {})[f"candidate_{i}_answer_function_list"] = answer_function_list

                        inference_log[f"step_{step}"]["inference_output"].update(
                            {
                                "current_action_name_label": "error",
                                "error_reason": f"action name not in candidate_answer_function_list"
                            }
                        )
                        break

            else:
                tool_calls_len = len(tool_calls)
                predict_function_name_list = []
                predict_function_arguments_list = []
                predict_function_id_list = []
                try:
                    for tool_call in tool_calls:
                        function_id = tool_call["id"]
                        function = tool_call["function"]
                        function_name = function["name"]
                        function_arguments = function["arguments"]
                        predict_function_id_list.append(function_id)
                        predict_function_name_list.append(function_name)
                        predict_function_arguments_list.append(function_arguments)
                except Exception as e:
                    print(f"{json.dumps(tool_calls, ensure_ascii=False)} parse failed.", flush=True)
                    action_name_label = "error"
                    inference_log[f"step_{step}"]["inference_output"].update(
                        {
                            "current_action_name_label": "error",
                            "error_reason": f"parse tool_calls failed, error: {str(e)}"
                        }
                    )
                    break

                idx_predict_function_name_list = list(enumerate(predict_function_name_list))
                sorted_idx_predict_function_name_list = sorted(idx_predict_function_name_list, key=lambda x: x[1])
                sorted_indices = [idx for idx, predict_function_name in sorted_idx_predict_function_name_list]
                sorted_predict_function_name_list = [predict_function_name_list[i] for i in sorted_indices]
                sorted_predict_function_arguments_list = [predict_function_arguments_list[i] for i in sorted_indices]
                predict_function_name_list = sorted_predict_function_name_list
                predict_function_arguments_list = sorted_predict_function_arguments_list

                idx_list = tool_call_graph.step_to_idx_list.get(step, None)
                if idx_list is None:
                    action_name_label = "error"
                    inference_log[f"step_{step}"]["inference_output"].update(
                        {
                            "current_action_name_label": "error",
                            "error_reason": f"current step: {step}, idx_list is None"
                        }
                    )
                    break
                else:
                    current_step_function_name_list = tool_call_graph.step_to_function_name_list[step]
                    current_step_function_arguments_list = tool_call_graph.step_to_function_arguments_list[step]
                    current_step_function_observation_list = tool_call_graph.step_to_function_observation_list[step]
                    current_step_user_input_list = tool_call_graph.step_to_user_input_list[step]
                    for i, (answer_function_name_list, answer_function_arguments_list) in enumerate(zip(current_step_function_name_list, current_step_function_arguments_list)):
                        if predict_function_name_list != answer_function_name_list:
                            continue
                        else:
                            if reasoning_content is not None:
                                messages.append(
                                    {"role": "assistant", "reasoning_content": reasoning_content, "content": content, "tool_calls": tool_calls}
                                )
                            else:
                                messages.append(
                                    {"role": "assistant", "content": content, "tool_calls": tool_calls}
                                )

                            function_observation_list = tool_call_graph.step_to_function_observation_list[step][i]
                            if self.consecutive_tool_messages:
                                                                             
                                for j, function_observation in enumerate(function_observation_list):
                                    if not isinstance(function_observation, str):
                                        function_observation = json.dumps(function_observation, ensure_ascii=False)
                                    function_id = predict_function_id_list[j]
                                    messages.append(
                                        {"role": "tool", "content": function_observation, "tool_call_id": function_id}
                                    )
                            else:
                                                                                     
                                function_observation_list = json.dumps(function_observation_list, ensure_ascii=False)
                                function_id = predict_function_id_list[0]
                                messages.append(
                                    {"role": "tool", "content": function_observation_list, "tool_call_id": function_id}
                                )

                                     
                            idx_list = tool_call_graph.step_to_idx_list[step][i]
                            tool_call_graph.update_updating_all_path_list(step, idx_list)
                            tool_call_graph.init_step_to_answer()

                            answer_function_list = {"action": []}
                            for answer_function_name, answer_function_arguments in zip(answer_function_name_list, answer_function_arguments_list):
                                answer_function_list["action"].append({"arguments": json.dumps(answer_function_arguments, ensure_ascii=False), "name": answer_function_name})
                            inference_log[f"step_{step}"].setdefault("inference_answer", {})[f"candidate_0_answer_function_list"] = answer_function_list
                            inference_log[f"step_{step}"]["inference_output"]["current_action_name_label"] = "correct"
                            break
                    else:
                        action_name_label = "error"
                                                           
                        for i, (answer_function_name_list, answer_function_arguments_list, answer_function_observation_list, answer_user_input_list) in enumerate(zip(
                                current_step_function_name_list,
                                current_step_function_arguments_list,
                                current_step_function_observation_list,
                                current_step_user_input_list
                            )
                        ):
                            answer_function_list = {"action": []}
                            for answer_function_name, answer_function_arguments, answer_function_observation, answer_user_input in zip(
                                    answer_function_name_list,
                                    answer_function_arguments_list,
                                    answer_function_observation_list,
                                    answer_user_input_list
                            ):
                                answer_function_list["action"].append(
                                    {"arguments": json.dumps(answer_function_arguments, ensure_ascii=False), "name": answer_function_name}
                                )
                                if answer_function_name == "ask_user_for_required_parameters":
                                    answer_function_list["observation"] = answer_function_observation
                                    answer_function_list["user_input"] = answer_user_input
                                elif answer_function_name == "prepare_to_answer":
                                    answer_function_list["observation"] = answer_function_observation

                            inference_log[f"step_{step}"].setdefault("inference_answer", {})[f"candidate_{i}_answer_function_list"] = answer_function_list

                        inference_log[f"step_{step}"]["inference_output"].update(
                            {
                                "current_action_name_label": "error",
                                "error_reason": f"action name not in candidate_answer_function_list"
                            }
                        )
                        break

            if action_name_label == "correct":
                inference_log[f"step_{step}"]["inference_output"]["current_action_name_label"] = "correct"
                break

            step += 1
            inference_data["messages"] = messages

                                               
                                  
                                                                                         
                                                                        
        if action_name_label == "correct":
            if step == (tool_call_graph.min_length - 1):
                is_optimal = True
            else:
                is_optimal = False
        else:
            is_optimal = False

        return {
            "action_name_label": action_name_label,
            "is_optimal": is_optimal,
            "inference_log": inference_log,
            "latency": latency,
            "input_token_count": input_token_count,
            "output_token_count": output_token_count
        }

    @final
    def write(self, result, result_dir, update_mode=False, dataset_name=None):
        model_name_dir = self.model_name.replace("/", "_")
        model_result_dir = result_dir / model_name_dir
        model_result_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(result, dict):
            result = [result]

        if dataset_name is None:
            dataset_name = result[0].get("dataset_name")
        file_path = model_result_dir / result_file_name(dataset_name, self.model_name)

        if update_mode:
                                                 
            existing_entries = {}
            if file_path.exists():
                existing_entries = {entry["id"]: entry for entry in load_file(file_path)}

                                                   
            for entry in result:
                existing_entries[entry["id"]] = entry

                                                                                  
            sorted_entries = sorted(existing_entries.values(), key=sort_key)
            with open(file_path, "w") as fout:
                for entry in sorted_entries:
                    fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

        else:
                                                 
            result.sort(key=sort_key)
            with open(file_path, "a") as fout:
                for entry in result:
                    fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
