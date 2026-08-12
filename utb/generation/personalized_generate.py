import argparse
import base64
import copy
import json
import os
import pickle
import random
import re
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

from agent import (
    agent_ask,
    agent_answer,
    agent_answer_chat,
    checker_planner,
    checker_tool,
    planner,
    tool,
    user_answer_ask,
    user_ask,
    user_chat,
    user_continue_question,
    user_multi_tool,
    user_multi_tool_parallel,
    user_multi_tool_serial_parallel,
    user_single_tool,
    user_vague_answer_ask,
)
from constant import DOTENV_PATH
from handle.handler_map import agent_handle_map
from utils import ask_user_for_help_tool, get_random_date, logger, parse_answer, prepare_to_answer_tool, read_json_file_to_list


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
PERSONA_DIR = PROJECT_DIR / "data" / "zh_persona"
RESULT_DIR = SCRIPT_DIR / "result"
CHECKPOINT_DIR = RESULT_DIR / "checkpoints"
TASK_TYPES = ("ST", "MT", "CQ", "CC")
PERSONA_IGNORED_FIELDS = {"matched_images", "preference_updates", "conversations"}
MODEL_CALL_LOG_PATH: Path | None = None
MODEL_CALL_COUNTER = 0

ROLE_MAX_TOKENS = {
    "user": 8192,
    "planner": 8192,
    "tool": 8192,
    "checker": 4096,
    "agent": 8192,
}

node_to_user_agent = {
    "ST": [user_single_tool],
    "MT": [user_multi_tool, user_multi_tool_parallel, user_multi_tool_serial_parallel],
    "CQ": [user_ask],
    "CC": [user_chat],
}


class RetryableGenerationError(RuntimeError):
    pass


@dataclass
class RetryPolicy:
    max_attempts: int = 6
    base_delay: float = 1.5
    max_delay: float = 30.0
    jitter: float = 0.35
    sleep: bool = True


@dataclass
class GenerationConfig:
    persona_limit: int = 1
    min_user_turns: int = 45
    max_user_turns: int = 55
    min_topic_turns: int = 2
    max_topic_turns: int = 5
    max_topic_failures: int = 3
    seed: int = 20260430
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)


def read_json_from_fence(text: str) -> Any:
    match = re.search(r"```json\s*(.+?)\s*```", text, re.S)
    if not match:
        raise ValueError("missing json code fence")
    return json.loads(match.group(1))


def is_truncated_response(text: str) -> bool:
    if text is None or not str(text).strip():
        return True
    stripped = str(text).rstrip()
    if "```json" in stripped and "```" not in stripped.split("```json", 1)[1]:
        return True
    return False


def validate_user_prefix(text: str) -> None:
    if is_truncated_response(text) or not text.lstrip().startswith(("User:", "用户：")):
        raise RetryableGenerationError("user response must start with User:/用户：")


def is_agent_response(text: str) -> bool:
    return text.lstrip().startswith(("Agent:", "Agent助手：", "**Agent:**"))


def validate_agent_prefix(text: str) -> None:
    if is_truncated_response(text) or not is_agent_response(text):
        raise RetryableGenerationError("agent response must start with Agent:/Agent助手：/**Agent:**")


def ensure_prefixed_text(text: str, prefixes: tuple[str, ...], default_prefix: str) -> str:
    if is_truncated_response(text):
        return text
    stripped = text.lstrip()
    if stripped.startswith(prefixes):
        return text
    return f"{default_prefix} {stripped}"


def normalize_user_text(text: str) -> str:
    language = os.getenv("LANGUAGE")
    default_prefix = "用户：" if language == "zh" else "User:"
    return ensure_prefixed_text(text, ("User:", "用户："), default_prefix)


def normalize_agent_text(text: str) -> str:
    language = os.getenv("LANGUAGE")
    default_prefix = "Agent助手：" if language == "zh" else "Agent:"
    return ensure_prefixed_text(text, ("Agent:", "Agent助手：", "**Agent:**"), default_prefix)


def normalize_initial_user_result(result: tuple[list[dict[str, str]], dict[str, Any]]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    user_messages, fetch_data = result
    if user_messages:
        user_messages = copy.deepcopy(user_messages)
        user_messages[0]["content"] = normalize_user_text(user_messages[0]["content"])
    return user_messages, fetch_data


def normalize_user_result(result: tuple[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    text, fetch_data = result
    return normalize_user_text(text), fetch_data


def normalize_agent_result(result: tuple[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    text, fetch_data = result
    return normalize_agent_text(text), fetch_data


def validate_planner_output(text: str) -> None:
    if is_truncated_response(text) or not text.lstrip().startswith(("Planner:", "Planner：")):
        raise RetryableGenerationError("planner response must start with Planner:")
    parsed = parse_answer(text)
    action_list = parsed.get("Action_List")
    if not isinstance(action_list, list) or not action_list:
        raise RetryableGenerationError("planner Action_List must be a non-empty list")
    for action in action_list:
        if not isinstance(action, dict) or not isinstance(action.get("arguments"), dict):
            raise RetryableGenerationError("planner action must contain dict arguments")


def validate_tool_output(text: str) -> None:
    if is_truncated_response(text) or not text.lstrip().startswith(("Tool:", "Tool：")):
        raise RetryableGenerationError("tool response must start with Tool:")
    parsed = read_json_from_fence(text)
    if not isinstance(parsed.get("Observation_List"), list):
        raise RetryableGenerationError("tool Observation_List must be a list")


def validate_checker_output(text: str) -> None:
    if is_truncated_response(text) or "```json" not in text:
        raise RetryableGenerationError("checker response must contain json fence")
    parsed = read_json_from_fence(text)
    if parsed.get("correct") not in {"yes", "no"}:
        raise RetryableGenerationError("checker result must include correct=yes/no")


def summarize_prompt(messages: list[dict[str, str]], limit: int = 240) -> str:
    if not messages:
        return ""
    content = messages[-1].get("content", "")
    return re.sub(r"\s+", " ", content)[:limit]


def set_model_call_log_path(path: Path | None) -> None:
    global MODEL_CALL_LOG_PATH
    MODEL_CALL_LOG_PATH = path
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)


def log_model_call(record: dict[str, Any]) -> None:
    if MODEL_CALL_LOG_PATH is None:
        return
    global MODEL_CALL_COUNTER
    MODEL_CALL_COUNTER += 1
    record = {
        "call_id": MODEL_CALL_COUNTER,
        "logged_at": datetime.now().isoformat(timespec="seconds"),
        **record,
    }
    with MODEL_CALL_LOG_PATH.open("a", encoding="utf-8") as fout:
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def call_with_retry(
    func: Callable[[], Any],
    *,
    persona_id: int,
    topic_id: int,
    agent_name: str,
    prompt_messages: list[dict[str, str]] | None,
    retry_policy: RetryPolicy,
    normalizer: Callable[[Any], Any] | None = None,
    validator: Callable[[Any], None] | None = None,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, retry_policy.max_attempts + 1):
        try:
            result = func()
            if normalizer is not None:
                result = normalizer(result)
            if validator is not None:
                validator(result)
            return result
        except Exception as exc:
            last_error = exc
            logger.warning(
                "generation retry persona_id=%s topic_id=%s agent=%s attempt=%s/%s error=%s prompt=%s",
                persona_id,
                topic_id,
                agent_name,
                attempt,
                retry_policy.max_attempts,
                type(exc).__name__,
                summarize_prompt(prompt_messages or []),
            )
            if attempt == retry_policy.max_attempts:
                break
            if retry_policy.sleep:
                delay = min(retry_policy.max_delay, retry_policy.base_delay * (2 ** (attempt - 1)))
                delay += random.uniform(0, retry_policy.jitter)
                time.sleep(delay)
    raise RetryableGenerationError(f"{agent_name} failed after retries: {last_error}")


def load_persona_file(path: Path) -> dict[str, Any]:
    with path.open() as fin:
        raw = json.load(fin)
    if isinstance(raw, dict) and len(raw) == 1 and isinstance(next(iter(raw.values())), dict):
        raw = next(iter(raw.values()))
    if not isinstance(raw, dict):
        raise ValueError(f"persona file is not a dict: {path}")
    return {k: v for k, v in raw.items() if k not in PERSONA_IGNORED_FIELDS}


def persona_number(path: Path) -> int:
    match = re.search(r"persona(\d+)", path.name)
    if not match:
        return 10**12
    return int(match.group(1))


def load_personas(persona_dir: Path = PERSONA_DIR, limit: int = 10) -> list[dict[str, Any]]:
    persona_files = sorted(persona_dir.glob("*.json"), key=lambda p: (persona_number(p), p.name))
    personas = []
    for path in persona_files[:limit]:
        personas.append(
            {
                "persona_id": persona_number(path),
                "persona_file": str(path),
                "persona": load_persona_file(path),
            }
        )
    return personas


def persona_instruction(role: str, persona: dict[str, Any]) -> str:
    persona_json = json.dumps(persona, ensure_ascii=False, indent=2)
    if os.getenv("LANGUAGE") == 'en':
        if role == "user":
            role_text = (
                "You are the same persona as the end user. Use the persona's identity, preferences, "
                "background, constraints, sensitive information, and language style when proposing tasks and follow-up questions."
            )
        elif role == "planner":
            role_text = (
                "You are the Planner, but you must understand the task from the same persona's perspective. "
                "Use the persona's context, preferences, and conversation history to interpret intent and plan tool calls."
            )
        elif role == "agent":
            role_text = (
                "You are the Agent assistant, and your final user-facing reply must reflect the same persona's style, "
                "background, preferences, and constraints while still completing the user's request."
            )
        else:
            raise ValueError(f"unknown persona role: {role}")
    elif os.getenv("LANGUAGE") == 'zh':
        if role == "user":
            role_text = (
            "你与最终用户具有相同的人设。提出任务和后续问题时，请使用该人设的身份、偏好、"
            "背景、约束条件、敏感信息和语言风格。"
        )
        elif role == "planner":
            role_text = (
                "你是规划器，但你必须从同一人设的视角理解任务。"
                "使用该人设的上下文、偏好和对话历史来解释意图并规划工具调用。"
            )
        elif role == "agent":
            role_text = (
                "你是智能体助手，你最终面向用户的回复必须体现同一人设的风格、"
                "背景、偏好和约束条件，同时仍然完成用户的请求。"
            )
        else:
            raise ValueError(f"unknown persona role: {role}")
    return f"[Persona]\n{persona_json}\n\n[Persona Role Instruction]\n{role_text}"


def inject_persona_prompt(messages: list[dict[str, str]], role: str, persona: dict[str, Any]) -> list[dict[str, str]]:
    injected = copy.deepcopy(messages)
    instruction = persona_instruction(role, persona)
    if injected:
        injected[0]["content"] = instruction + "\n\n" + injected[0].get("content", "")
    else:
        injected.append({"role": "system", "content": instruction})
    return injected


def role_request_func(
    base_request_func: Callable[..., str],
    role: str,
    persona: dict[str, Any] | None,
    *,
    call_name: str | None = None,
    persona_id: int | None = None,
    topic_id: int | None = None,
) -> Callable[[list[dict[str, str]]], str]:
    def request(messages: list[dict[str, str]]) -> str:
        outbound = inject_persona_prompt(messages, role, persona) if persona is not None and role in {"user", "planner", "agent"} else messages
        max_tokens = ROLE_MAX_TOKENS[role]
        started_at = time.time()
        try:
            output = base_request_func(outbound, max_tokens=max_tokens)
            log_model_call(
                {
                    "status": "success",
                    "role": role,
                    "call_name": call_name or role,
                    "persona_id": persona_id,
                    "topic_id": topic_id,
                    "max_tokens": max_tokens,
                    "latency_seconds": round(time.time() - started_at, 6),
                    "prompt": outbound,
                    "output": output,
                }
            )
            return output
        except TypeError:
            try:
                output = base_request_func(outbound)
            except Exception as exc:
                log_model_call(
                    {
                        "status": "error",
                        "role": role,
                        "call_name": call_name or role,
                        "persona_id": persona_id,
                        "topic_id": topic_id,
                        "max_tokens": None,
                        "latency_seconds": round(time.time() - started_at, 6),
                        "prompt": outbound,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                raise
            log_model_call(
                {
                    "status": "success",
                    "role": role,
                    "call_name": call_name or role,
                    "persona_id": persona_id,
                    "topic_id": topic_id,
                    "max_tokens": None,
                    "latency_seconds": round(time.time() - started_at, 6),
                    "prompt": outbound,
                    "output": output,
                }
            )
            return output
        except Exception as exc:
            log_model_call(
                {
                    "status": "error",
                    "role": role,
                    "call_name": call_name or role,
                    "persona_id": persona_id,
                    "topic_id": topic_id,
                    "max_tokens": max_tokens,
                    "latency_seconds": round(time.time() - started_at, 6),
                    "prompt": outbound,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            raise

    return request


def switch_prompt(role: str) -> dict[str, str]:
    prompts = {
        "planner": "Switch to the role to Planner and continue to output the Planner's decisions. Note: 1. Each time you generate, if you have previously generated incorrect decisions, please do not explain the previously generated incorrect results or plan adjustments in the Thought and Plan sections for this round. Instead, treat it as a brand new round and provide your Thought and Plan. \n2. Be sure not to mention the use of the prepare_to_answer tool and the ask_user_for_required_parameters tool in the Plan.",
        "checker": "Switch the role to Checker and continue to output the Checker's inspection results.",
        "tool": "Switch the role to Tool and continue to output the execution results of Tool.",
        "agent_inquiry": "Switch the role to Agent and continue to output the inquiry information.",
        "agent_summary": "Switch the role to Agent and continue to output summary replies, be careful not to output words like ```markdown```.",
        "agent_direct": "Switch the role to Agent and continue to output direct replies.",
        "user": "Switch the role to user and continue to propose new tasks.",
        "user_reply": "Switch the role to User and continue to output the User's responses.",
    }
    return {"role": "user", "content": prompts[role]}


def tools_with_runtime_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = copy.deepcopy(tools)
    merged.append(copy.deepcopy(ask_user_for_help_tool))
    merged.append(copy.deepcopy(prepare_to_answer_tool))
    return merged


def tool_names(tools: list[dict[str, Any]]) -> set[str]:
    names = {item["function"]["name"] for item in tools}
    names.add("ask_user_for_required_parameters")
    names.add("prepare_to_answer")
    return names


def extract_initial_task(fetch_data: dict[str, Any] | None, user_message: list[dict[str, str]]) -> tuple[str | None, Any]:
    selected = user_message[0]["content"] if user_message else None
    candidates = None
    if fetch_data and fetch_data.get("answer"):
        try:
            candidates = read_json_from_fence(fetch_data["answer"])
        except Exception:
            candidates = None
    return selected, candidates


def run_agent_until_answer(
    *,
    node_path: list[str],
    messages: list[dict[str, str]],
    tools: list[dict[str, Any]],
    visible_tools: list[dict[str, Any]],
    env_info: str,
    fetch_data_list: list[dict[str, Any]],
    handles: dict[str, Any],
    persona: dict[str, Any],
    persona_id: int,
    topic_id: int,
    retry_policy: RetryPolicy,
) -> list[dict[str, str]]:
    all_tool_name = tool_names(tools)
    turns = 0
    while turns < 100:
        last_content = messages[-1]["content"]
        if last_content.startswith(("用户", "User")) or "Tool：\n```json" in last_content or "Tool:\n```json" in last_content or last_content.startswith("Checker_Tool"):
            tool_flag = "Tool：\n```json" in last_content or "Tool:\n```json" in last_content
            checker_passed = False
            for _ in range(3):
                messages.append(switch_prompt("planner"))
                planner_res, planner_fetch = call_with_retry(
                    lambda: planner(
                        messages,
                        tools,
                        env_info,
                        role_request_func(
                            handles["planner"].request_model,
                            "planner",
                            persona,
                            call_name="planner",
                            persona_id=persona_id,
                            topic_id=topic_id,
                        ),
                    ),
                    persona_id=persona_id,
                    topic_id=topic_id,
                    agent_name="planner",
                    prompt_messages=messages,
                    retry_policy=retry_policy,
                    validator=lambda result: validate_planner_output(result[0]),
                )
                messages.append({"role": "assistant", "content": planner_res})
                messages.append(switch_prompt("checker"))
                correct, checker_res, checker_fetch = call_with_retry(
                    lambda: checker_planner(
                        messages,
                        tools,
                        env_info,
                        tool_flag,
                        role_request_func(
                            handles["checker"].request_model,
                            "checker",
                            None,
                            call_name="checker_planner",
                            persona_id=persona_id,
                            topic_id=topic_id,
                        ),
                        True,
                    ),
                    persona_id=persona_id,
                    topic_id=topic_id,
                    agent_name="checker_planner",
                    prompt_messages=messages,
                    retry_policy=retry_policy,
                    validator=lambda result: validate_checker_output(result[1]),
                )
                fetch_data_list.append(checker_fetch)
                if correct == "yes":
                    messages.pop()
                    fetch_data_list.append(planner_fetch)
                    checker_passed = True
                    break
                messages.append({"role": "assistant", "content": checker_res})
            if not checker_passed:
                raise RetryableGenerationError("planner checker failed")

        elif is_agent_response(last_content):
            messages.append(switch_prompt("user_reply"))
            user_func = user_answer_ask if random.choice([True, True, True, False]) else user_vague_answer_ask
            res, fetch_data = call_with_retry(
                lambda: user_func(
                    messages,
                    visible_tools,
                    env_info,
                    role_request_func(
                        handles["user"].request_model,
                        "user",
                        persona,
                        call_name=user_func.__name__,
                        persona_id=persona_id,
                        topic_id=topic_id,
                    ),
                ),
                persona_id=persona_id,
                topic_id=topic_id,
                agent_name=user_func.__name__,
            prompt_messages=messages,
            retry_policy=retry_policy,
            normalizer=normalize_user_result,
            validator=lambda result: validate_user_prefix(result[0]),
        )
            messages.append({"role": "assistant", "content": res})
            fetch_data_list.append(fetch_data)

        elif last_content.startswith(("Planner", "Checker_Planner")):
            parse_source = messages[-3]["content"] if last_content.startswith("Checker_Planner") else last_content
            parse_content = parse_answer(parse_source)
            action_list = parse_content["Action_List"]
            if not isinstance(action_list, list):
                raise RetryableGenerationError("planner Action_List is not a list")
            for action in action_list:
                if action["name"] not in all_tool_name:
                    raise RetryableGenerationError(f"planner selected unknown tool: {action['name']}")

            if action_list[0]["name"] == "ask_user_for_required_parameters":
                messages.append(switch_prompt("agent_inquiry"))
                ask_res, fetch_data = call_with_retry(
                    lambda: agent_ask(
                        messages,
                        tools,
                        env_info,
                        role_request_func(
                            handles["agent"].request_model,
                            "agent",
                            persona,
                            call_name="agent_ask",
                            persona_id=persona_id,
                            topic_id=topic_id,
                        ),
                    ),
                    persona_id=persona_id,
                    topic_id=topic_id,
                    agent_name="agent_ask",
                    prompt_messages=messages,
                    retry_policy=retry_policy,
                    normalizer=normalize_agent_result,
                    validator=lambda result: validate_agent_prefix(result[0]),
                )
                messages.append({"role": "assistant", "content": ask_res})
                fetch_data_list.append(fetch_data)

            elif action_list[0]["name"] == "prepare_to_answer":
                answer_type = action_list[0]["arguments"].get("answer_type")
                prompt_type = "agent_summary" if answer_type == "tool" else "agent_direct"
                answer_func = agent_answer if answer_type == "tool" else agent_answer_chat
                messages.append(switch_prompt(prompt_type))
                answer_res, fetch_data = call_with_retry(
                    lambda: answer_func(
                        messages,
                        tools,
                        env_info,
                        role_request_func(
                            handles["agent"].request_model,
                            "agent",
                            persona,
                            call_name=answer_func.__name__,
                            persona_id=persona_id,
                            topic_id=topic_id,
                        ),
                    ),
                    persona_id=persona_id,
                    topic_id=topic_id,
                    agent_name=answer_func.__name__,
                    prompt_messages=messages,
                    retry_policy=retry_policy,
                    normalizer=normalize_agent_result,
                    validator=lambda result: validate_agent_prefix(result[0]),
                )
                messages.append({"role": "assistant", "content": answer_res})
                fetch_data_list.append(fetch_data)
                return messages

            else:
                checker_passed = False
                for _ in range(3):
                    messages.append(switch_prompt("tool"))
                    tool_res, fetch_data = call_with_retry(
                        lambda: tool(
                            messages,
                            tools,
                            env_info,
                            role_request_func(
                                handles["tool"].request_model,
                                "tool",
                                None,
                                call_name="tool",
                                persona_id=persona_id,
                                topic_id=topic_id,
                            ),
                        ),
                        persona_id=persona_id,
                        topic_id=topic_id,
                        agent_name="tool",
                        prompt_messages=messages,
                        retry_policy=retry_policy,
                        validator=lambda result: validate_tool_output(result[0]),
                    )
                    messages.append({"role": "user", "content": tool_res})
                    messages.append(switch_prompt("checker"))
                    correct, checker_res = call_with_retry(
                        lambda: checker_tool(
                            messages,
                            action_list,
                            tools,
                            env_info,
                            role_request_func(
                                handles["checker"].request_model,
                                "checker",
                                None,
                                call_name="checker_tool",
                                persona_id=persona_id,
                                topic_id=topic_id,
                            ),
                        ),
                        persona_id=persona_id,
                        topic_id=topic_id,
                        agent_name="checker_tool",
                        prompt_messages=messages,
                        retry_policy=retry_policy,
                        validator=lambda result: validate_checker_output(result[1]),
                    )
                    if correct == "yes":
                        messages.pop()
                        fetch_data_list.append(fetch_data)
                        checker_passed = True
                        break
                    messages.append({"role": "assistant", "content": checker_res})
                if not checker_passed:
                    raise RetryableGenerationError("tool checker failed")
        else:
            raise RetryableGenerationError(f"unexpected conversation state: {last_content[:80]}")

        turns += 1

    raise RetryableGenerationError("agent loop exceeded 100 turns")


def random_node_path(rng: random.Random, topic_turns: int) -> list[str]:
    return [rng.choice(TASK_TYPES) for _ in range(topic_turns)]


def select_initial_user_agent(node: str, rng: random.Random) -> Callable[..., Any]:
    if node not in node_to_user_agent:
        raise ValueError(f"unknown node type: {node}")
    return rng.choice(node_to_user_agent[node])


def generate_initial_user_turn(
    *,
    node: str,
    messages: list[dict[str, str]],
    visible_tools: list[dict[str, Any]],
    handles: dict[str, Any],
    persona: dict[str, Any],
    persona_id: int,
    topic_id: int,
    retry_policy: RetryPolicy,
    rng: random.Random,
) -> tuple[list[dict[str, str]], dict[str, Any], str | None, Any]:
    user_start_func = select_initial_user_agent(node, rng)
    user_message, fetch_data = call_with_retry(
        lambda: user_start_func(
            messages,
            visible_tools,
            role_request_func(
                handles["user"].request_model,
                "user",
                persona,
                call_name=user_start_func.__name__,
                persona_id=persona_id,
                topic_id=topic_id,
            ),
        ),
        persona_id=persona_id,
        topic_id=topic_id,
        agent_name=f"{user_start_func.__name__}[{node}]",
        prompt_messages=messages,
        retry_policy=retry_policy,
        normalizer=normalize_initial_user_result,
        validator=lambda result: validate_user_prefix(result[0][0]["content"]),
    )
    selected_initial_task, candidate_tasks = extract_initial_task(fetch_data, user_message)
    return user_message, fetch_data, selected_initial_task, candidate_tasks


def generate_continue_user_turn(
    *,
    node: str,
    messages: list[dict[str, str]],
    visible_tools: list[dict[str, Any]],
    env_info: str,
    handles: dict[str, Any],
    persona: dict[str, Any],
    persona_id: int,
    topic_id: int,
    retry_policy: RetryPolicy,
) -> tuple[str, dict[str, Any]]:
    messages.append(switch_prompt("user"))
    return call_with_retry(
        lambda: user_continue_question(
            messages,
            visible_tools,
            env_info,
            role_request_func(
                handles["user"].request_model,
                "user",
                persona,
                call_name=f"user_continue_question[{node}]",
                persona_id=persona_id,
                topic_id=topic_id,
            ),
            node,
        ),
        persona_id=persona_id,
        topic_id=topic_id,
        agent_name=f"user_continue_question[{node}]",
        prompt_messages=messages,
        retry_policy=retry_policy,
        normalizer=normalize_user_result,
        validator=lambda result: validate_user_prefix(result[0]),
    )


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fout:
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")


def atomic_write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fout:
        json.dump(obj, fout, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def encode_random_state(rng: random.Random) -> str:
    return base64.b64encode(pickle.dumps(rng.getstate())).decode("ascii")


def decode_random_state(data: str) -> Any:
    return pickle.loads(base64.b64decode(data.encode("ascii")))


def checkpoint_path(persona_id: int) -> Path:
    return CHECKPOINT_DIR / f"persona_{persona_id}.json"


def load_checkpoint(persona_id: int) -> dict[str, Any] | None:
    path = checkpoint_path(persona_id)
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fin:
        return json.load(fin)


def save_checkpoint(
    *,
    persona_id: int,
    messages: list[dict[str, str]],
    topics: list[dict[str, Any]],
    completed_user_turns: int,
    next_topic_id: int,
    rng: random.Random,
    output_file: Path,
    task_file: Path,
    failed_topics: list[dict[str, Any]],
    seed: int,
) -> None:
    atomic_write_json(
        checkpoint_path(persona_id),
        {
            "persona_id": persona_id,
            "messages": messages,
            "topics": topics,
            "completed_user_turns": completed_user_turns,
            "next_topic_id": next_topic_id,
            "random_seed": seed,
            "random_state": encode_random_state(rng),
            "output_file": str(output_file),
            "task_file": str(task_file),
            "failed_topics": failed_topics,
        },
    )


def completed_persona_ids(result_dir: Path = RESULT_DIR) -> set[int]:
    completed = set()
    for path in result_dir.glob("persona_train_segmented_*.jsonl"):
        with path.open(encoding="utf-8") as fin:
            for line in fin:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "persona_id" in obj:
                    completed.add(int(obj["persona_id"]))
    return completed


def generate_topic(
    *,
    persona_record: dict[str, Any],
    topic_id: int,
    messages: list[dict[str, str]],
    completed_user_turns: int,
    tools_all_list: list[list[dict[str, Any]]],
    handles: dict[str, Any],
    rng: random.Random,
    retry_policy: RetryPolicy,
    config: GenerationConfig,
) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any], int]:
    persona = persona_record["persona"]
    persona_id = persona_record["persona_id"]
    selected_tools = rng.choice(tools_all_list)
    visible_tools = copy.deepcopy(selected_tools)
    tools = tools_with_runtime_tools(selected_tools)
    topic_turns = rng.randint(config.min_topic_turns, config.max_topic_turns)
    node_path = random_node_path(rng, topic_turns)
    first_turn_mode = "initial" if not messages else rng.choice(["initial", "continue"])
    env_info = "Current Time：" + get_random_date()
    fetch_data_list: list[dict[str, Any]] = []
    message_start = len(messages)
    user_turn_start = completed_user_turns + 1
    selected_initial_task = None
    candidate_tasks = None

    if first_turn_mode == "initial":
        user_message, fetch_data, selected_initial_task, candidate_tasks = generate_initial_user_turn(
            node=node_path[0],
            messages=messages,
            visible_tools=visible_tools,
            handles=handles,
            persona=persona,
            persona_id=persona_id,
            topic_id=topic_id,
            retry_policy=retry_policy,
            rng=rng,
        )
        messages.extend(user_message)
        fetch_data_list.append(fetch_data)
    else:
        user_res, fetch_data = generate_continue_user_turn(
            node=node_path[0],
            messages=messages,
            visible_tools=visible_tools,
            env_info=env_info,
            handles=handles,
            persona=persona,
            persona_id=persona_id,
            topic_id=topic_id,
            retry_policy=retry_policy,
        )
        selected_initial_task = user_res
        messages.append({"role": "assistant", "content": user_res})
        fetch_data_list.append(fetch_data)

    messages = run_agent_until_answer(
        node_path=node_path,
        messages=messages,
        tools=tools,
        visible_tools=visible_tools,
        env_info=env_info,
        fetch_data_list=fetch_data_list,
        handles=handles,
        persona=persona,
        persona_id=persona_id,
        topic_id=topic_id,
        retry_policy=retry_policy,
    )
    completed_user_turns += 1

    for node in node_path[1:]:
        user_res, fetch_data = generate_continue_user_turn(
            node=node,
            messages=messages,
            visible_tools=visible_tools,
            env_info=env_info,
            handles=handles,
            persona=persona,
            persona_id=persona_id,
            topic_id=topic_id,
            retry_policy=retry_policy,
        )
        messages.append({"role": "assistant", "content": user_res})
        fetch_data_list.append(fetch_data)
        messages = run_agent_until_answer(
            node_path=node_path,
            messages=messages,
            tools=tools,
            visible_tools=visible_tools,
            env_info=env_info,
            fetch_data_list=fetch_data_list,
            handles=handles,
            persona=persona,
            persona_id=persona_id,
            topic_id=topic_id,
            retry_policy=retry_policy,
        )
        completed_user_turns += 1

    topic_meta = {
        "topic_id": topic_id,
        "message_start": message_start,
        "message_end": len(messages),
        "user_turn_start": user_turn_start,
        "user_turn_end": completed_user_turns,
        "tools": selected_tools,
        "node_path": node_path,
        "topic_turns": topic_turns,
        "first_turn_mode": first_turn_mode,
    }
    task_record = {
        "persona_id": persona_id,
        "persona_file": persona_record["persona_file"],
        "topic_id": topic_id,
        "topic_turns": topic_turns,
        "first_turn_mode": first_turn_mode,
        "node_path": node_path,
        "tools": selected_tools,
        "selected_initial_task": selected_initial_task,
        "candidate_tasks": candidate_tasks,
        "persona_judge": {"ignored_fields": sorted(PERSONA_IGNORED_FIELDS), "persona_injected_roles": ["user", "planner", "agent"]},
        "fetch_data": fetch_data_list,
    }
    return messages, topic_meta, task_record, completed_user_turns


def generate_persona(
    *,
    persona_record: dict[str, Any],
    tools_all_list: list[list[dict[str, Any]]],
    handles: dict[str, Any],
    output_file: Path,
    task_file: Path,
    config: GenerationConfig,
) -> None:
    persona_id = persona_record["persona_id"]
    rng = random.Random(config.seed + persona_id)
    messages: list[dict[str, str]] = []
    topics: list[dict[str, Any]] = []
    completed_user_turns = 0
    next_topic_id = 0
    failed_topics: list[dict[str, Any]] = []

    checkpoint = load_checkpoint(persona_id)
    if checkpoint:
        messages = checkpoint["messages"]
        topics = checkpoint["topics"]
        completed_user_turns = checkpoint["completed_user_turns"]
        next_topic_id = checkpoint["next_topic_id"]
        failed_topics = checkpoint.get("failed_topics", [])
        if checkpoint.get("random_state"):
            rng.setstate(decode_random_state(checkpoint["random_state"]))
        output_file = Path(checkpoint.get("output_file", output_file))
        task_file = Path(checkpoint.get("task_file", task_file))
        logger.info("resuming persona_id=%s from topic_id=%s user_turns=%s", persona_id, next_topic_id, completed_user_turns)

    consecutive_failures = 0
    while completed_user_turns < config.min_user_turns:
        topic_id = next_topic_id
        try:
            new_messages, topic_meta, task_record, completed_user_turns = generate_topic(
                persona_record=persona_record,
                topic_id=topic_id,
                messages=messages,
                completed_user_turns=completed_user_turns,
                tools_all_list=tools_all_list,
                handles=handles,
                rng=rng,
                retry_policy=config.retry_policy,
                config=config,
            )
            messages = new_messages
            topics.append(topic_meta)
            append_jsonl(task_file, task_record)
            next_topic_id += 1
            consecutive_failures = 0
            save_checkpoint(
                persona_id=persona_id,
                messages=messages,
                topics=topics,
                completed_user_turns=completed_user_turns,
                next_topic_id=next_topic_id,
                rng=rng,
                output_file=output_file,
                task_file=task_file,
                failed_topics=failed_topics,
                seed=config.seed,
            )
            if completed_user_turns >= config.min_user_turns:
                break
        except Exception as exc:
            traceback.print_exc()
            failed_topics.append({"topic_id": topic_id, "error_type": type(exc).__name__, "error": str(exc)})
            next_topic_id += 1
            consecutive_failures += 1
            save_checkpoint(
                persona_id=persona_id,
                messages=messages,
                topics=topics,
                completed_user_turns=completed_user_turns,
                next_topic_id=next_topic_id,
                rng=rng,
                output_file=output_file,
                task_file=task_file,
                failed_topics=failed_topics,
                seed=config.seed,
            )
            if consecutive_failures >= config.max_topic_failures:
                logger.warning("persona_id=%s reached consecutive failed topic limit; continuing with next persona", persona_id)
                break

    if completed_user_turns >= config.min_user_turns:
        env_infos ={"language": "en", "source": "WildToolBench tools_en.jsonl"} if os.getenv("LANGUAGE") == 'en' else {"language": "zh", "source": "WildToolBench tools_zh.jsonl"}
        append_jsonl(
            output_file,
            {
                "persona_id": persona_id,
                "persona": persona_record["persona"],
                "env_info": env_infos,                                                               
                "messages": messages,
                "topics": topics,
            },
        )
        checkpoint_path(persona_id).unlink(missing_ok=True)


def build_handles(args: argparse.Namespace) -> dict[str, Any]:
    user_model_list = [agent_handle_map[user_model] for user_model in args.user_model]
    planner_model = agent_handle_map[args.planner_model]
    tool_model = agent_handle_map[args.tool_model]
    agent_model = agent_handle_map[args.agent_model]
    checker_model = agent_handle_map[args.checker_model]
    return {
        "user": random.choice([user_model(args.user_model[i], args.user_temperature) for i, user_model in enumerate(user_model_list)]),
        "planner": planner_model(args.planner_model, args.temperature),
        "tool": tool_model(args.tool_model, args.temperature),
        "agent": agent_model(args.agent_model, args.temperature),
        "checker": checker_model(args.checker_model, args.temperature),
    }


def main() -> None:
    load_dotenv(dotenv_path=DOTENV_PATH, verbose=True, override=True)
    os.environ["LANGUAGE"] = "zh"

    parser = argparse.ArgumentParser(description="Generate personalized segmented WildToolBench conversations.")
    parser.add_argument("--persona-limit", type=int, default=1)
    parser.add_argument("--min-user-turns", type=int, default=15)     
    parser.add_argument("--max-user-turns", type=int, default=25)     
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--user-model", type=str, default=["qwen3-32b"], nargs="+")                 
    parser.add_argument("--planner-model", type=str, default="qwen3-32b")
    parser.add_argument("--tool-model", type=str, default="qwen3-32b")
    parser.add_argument("--agent-model", type=str, default="qwen3-32b")
    parser.add_argument("--checker-model", type=str, default="qwen3-32b")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--user-temperature" , type=float , default=0.5)
    parser.add_argument("--model-call-log-path", type=str, default=None)
    args = parser.parse_args()

    config = GenerationConfig(
        persona_limit=args.persona_limit,
        min_user_turns=args.min_user_turns,
        max_user_turns=args.max_user_turns,
        seed=args.seed,
    )
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    formatted_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    output_file = RESULT_DIR / f"persona_train_segmented_{formatted_time}.jsonl"
    task_file = RESULT_DIR / f"persona_tasks_{formatted_time}.jsonl"
    model_call_log_file = Path(args.model_call_log_path) if args.model_call_log_path else RESULT_DIR / f"model_calls_{formatted_time}.jsonl"
    set_model_call_log_path(model_call_log_file)
    tools_all_list = read_json_file_to_list(str(SCRIPT_DIR / "tools" / "tools_en.jsonl")) if os.getenv("LANGUAGE")=='en' else read_json_file_to_list(str(SCRIPT_DIR / "tools" / "tools_zh.jsonl"))
    handles = build_handles(args)
    completed = completed_persona_ids()

    for persona_record in load_personas(limit=config.persona_limit):
        if persona_record["persona_id"] in completed:
            logger.info("skip completed persona_id=%s", persona_record["persona_id"])
            continue
        generate_persona(
            persona_record=persona_record,
            tools_all_list=tools_all_list,
            handles=handles,
            output_file=output_file,
            task_file=task_file,
            config=config,
        )


if __name__ == "__main__":
    main()
