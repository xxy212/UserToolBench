                      
"""Render User/Planner/Agent messages from a JSON or JSONL file as chat HTML."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ROLES = ("User", "Planner", "Agent")


def load_records(input_path: Path) -> list[dict[str, Any]]:
    """Load records from a JSON object/list file or a JSONL file."""
    text = input_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        records: list[dict[str, Any]] = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
            if isinstance(item, dict):
                records.append(item)
            else:
                raise ValueError(f"Line {line_no} is not a JSON object.")
        return records

    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        if not all(isinstance(item, dict) for item in parsed):
            raise ValueError("JSON list must contain objects.")
        return parsed
    raise ValueError("Input must be a JSON object, JSON list, or JSONL objects.")


def infer_task_path(input_path: Path) -> Path | None:
    if "persona_train_segmented_" not in input_path.name:
        return None
    task_name = input_path.name.replace("persona_train_segmented_", "persona_tasks_", 1)
    task_path = input_path.with_name(task_name)
    return task_path if task_path.exists() else None


def get_tool_names(tools: Any) -> list[str]:
    if not isinstance(tools, list):
        return []
    names: list[str] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if isinstance(function, dict) and function.get("name"):
            names.append(str(function["name"]))
    return names


def strip_role_prefix(text: str) -> str:
    stripped = text.strip()
    for prefix in ("User:", "User：", "用户:", "用户："):
        if stripped.startswith(prefix):
            return stripped[len(prefix) :].lstrip()
    return stripped


def build_task_index(task_records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    task_index: dict[tuple[str, str], dict[str, Any]] = {}
    for task in task_records:
        persona_id = task.get("persona_id")
        topic_id = task.get("topic_id")
        if persona_id is None or topic_id is None:
            continue
        task_index[(str(persona_id), str(topic_id))] = task
    return task_index


def build_topic_segments(
    record: dict[str, Any], task_index: dict[tuple[str, str], dict[str, Any]]
) -> list[dict[str, Any]]:
    persona_id = str(record.get("persona_id", ""))
    topics = record.get("topics", [])
    if not isinstance(topics, list):
        return []

    segments: list[dict[str, Any]] = []
    for idx, topic in enumerate(topics, start=1):
        if not isinstance(topic, dict):
            continue
        topic_id = topic.get("topic_id", idx)
        task = task_index.get((persona_id, str(topic_id)), {})
        selected_task = task.get("selected_initial_task", "")
        node_path = topic.get("node_path", task.get("node_path"))
        tools = get_tool_names(task.get("tools", topic.get("tools")))
        segment = {
            "topic_id": str(topic_id),
            "task": strip_role_prefix(str(selected_task)) if selected_task else "",
            "first_turn_mode": task.get("first_turn_mode", topic.get("first_turn_mode", "")),
            "node_path": "->".join(str(item) for item in node_path)
            if isinstance(node_path, list)
            else "",
            "topic_turns": topic.get("topic_turns", task.get("topic_turns", "")),
            "message_start": topic.get("message_start"),
            "message_end": topic.get("message_end"),
            "user_turn_start": topic.get("user_turn_start"),
            "user_turn_end": topic.get("user_turn_end"),
            "tools": tools,
        }
        segments.append(segment)

    return segments


def find_segment_for_message(
    segments: list[dict[str, Any]], message_index: int
) -> dict[str, Any] | None:
    for segment in segments:
        start = segment.get("message_start")
        end = segment.get("message_end")
        if isinstance(start, int) and isinstance(end, int) and start <= message_index < end:
            return segment
    return None


def extract_role_messages(
    record: dict[str, Any], roles: Iterable[str], segments: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    role_prefixes = tuple(
        (role, prefix) for role in roles for prefix in (f"{role}:", f"{role}：")
    )
    extracted: list[dict[str, Any]] = []

    messages = record.get("messages", [])
    if not isinstance(messages, list):
        return extracted

    user_turn = 0
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue

        stripped = content.lstrip()
        for role, prefix in role_prefixes:
            if stripped.startswith(prefix):
                if role == "User":
                    user_turn += 1
                segment = find_segment_for_message(segments, message_index)
                extracted.append(
                    {
                        "role": role,
                        "content": stripped[len(prefix) :].lstrip(),
                        "message_index": message_index,
                        "user_turn": user_turn,
                        "topic_id": segment.get("topic_id") if segment else "",
                    }
                )
                break
    return extracted


def summarize_topics(topics: Any) -> str:
    if not isinstance(topics, list):
        return str(topics) if topics else ""

    summaries: list[str] = []
    for idx, topic in enumerate(topics, start=1):
        if not isinstance(topic, dict):
            summaries.append(str(topic))
            continue

        topic_id = topic.get("topic_id", idx)
        node_path = topic.get("node_path")
        topic_turns = topic.get("topic_turns")
        message_start = topic.get("message_start")
        message_end = topic.get("message_end")

        parts = [f"topic {topic_id}"]
        if isinstance(node_path, list):
            parts.append("path=" + "->".join(str(item) for item in node_path))
        if topic_turns is not None:
            parts.append(f"turns={topic_turns}")
        if message_start is not None and message_end is not None:
            parts.append(f"messages={message_start}-{message_end}")
        summaries.append(" ".join(parts))
    return "; ".join(summaries)


def render_text(content: str) -> str:
    return html.escape(content)


def render_html(
    input_path: Path,
    records: list[dict[str, Any]],
    roles: Iterable[str],
    task_records: list[dict[str, Any]] | None = None,
    task_path: Path | None = None,
) -> str:
    task_index = build_task_index(task_records or [])
    sessions: list[dict[str, Any]] = []
    total_messages = 0
    for idx, record in enumerate(records, start=1):
        segments = build_topic_segments(record, task_index)
        messages = extract_role_messages(record, roles, segments)
        total_messages += len(messages)
        persona_id = record.get("persona_id", "")
        topic_text = summarize_topics(record.get("topics", []))
        sessions.append(
            {
                "index": idx,
                "persona_id": str(persona_id),
                "topics": topic_text,
                "segments": segments,
                "messages": messages,
            }
        )

    nav_items = []
    session_blocks = []
    for session in sessions:
        title = f"Session {session['index']}"
        if session["persona_id"]:
            title += f" · {session['persona_id']}"
        nav_items.append(
            f'<a href="#session-{session["index"]}">{html.escape(title)}'
            f'<span>{len(session["messages"])} messages</span></a>'
        )

        meta_parts = [f"{len(session['messages'])} extracted messages"]
        if session["topics"]:
            meta_parts.append(f"topics: {session['topics']}")
        meta = " | ".join(meta_parts)

        task_cards = []
        for segment in session["segments"]:
            turn_start = segment.get("user_turn_start")
            turn_end = segment.get("user_turn_end")
            turn_range = (
                f"turn {turn_start}-{turn_end}"
                if turn_start is not None and turn_end is not None
                else f"{segment.get('topic_turns', '')} turns"
            )
            tool_text = ", ".join(segment["tools"][:8])
            if len(segment["tools"]) > 8:
                tool_text += f", +{len(segment['tools']) - 8} more"
            details = [
                f"topic {segment['topic_id']}",
                turn_range,
            ]
            if segment.get("first_turn_mode"):
                details.append(str(segment["first_turn_mode"]))
            if segment.get("node_path"):
                details.append(str(segment["node_path"]))
            if tool_text:
                details.append(f"tools: {tool_text}")
            task_cards.append(
                '<article class="task-card">'
                f'<div class="task-title">{html.escape(" | ".join(details))}</div>'
                f'<div class="task-text">{render_text(str(segment.get("task") or "No selected task found."))}</div>'
                "</article>"
            )
        task_overview = (
            f'<div class="task-overview">{"".join(task_cards)}</div>' if task_cards else ""
        )

        bubbles = []
        last_topic_id = None
        for message in session["messages"]:
            role = message["role"]
            topic_id = message.get("topic_id", "")
            if topic_id and topic_id != last_topic_id:
                segment = next(
                    (
                        item
                        for item in session["segments"]
                        if item.get("topic_id") == topic_id
                    ),
                    {},
                )
                turn_start = segment.get("user_turn_start")
                turn_end = segment.get("user_turn_end")
                turn_label = (
                    f"turn {turn_start}-{turn_end}"
                    if turn_start is not None and turn_end is not None
                    else "turn range unknown"
                )
                divider_task = segment.get("task") or "No selected task found."
                bubbles.append(
                    '<div class="topic-divider">'
                    f'<div class="topic-label">Task topic {html.escape(str(topic_id))} · {html.escape(turn_label)}</div>'
                    f'<div class="topic-task">{render_text(str(divider_task))}</div>'
                    "</div>"
                )
                last_topic_id = topic_id
            role_detail = f'{role} · turn {message.get("user_turn", "")}'
            if topic_id:
                role_detail += f' · topic {topic_id}'
            bubbles.append(
                f'<article class="message {role.lower()}">'
                f'<div class="avatar">{html.escape(role[0])}</div>'
                f'<div class="bubble">'
                f'<div class="role">{html.escape(role_detail)}</div>'
                f'<div class="content">{render_text(message["content"])}</div>'
                f"</div>"
                f"</article>"
            )
        if not bubbles:
            bubbles.append('<p class="empty">No User/Planner/Agent messages found.</p>')

        session_blocks.append(
            f'<section id="session-{session["index"]}" class="session">'
            f"<header><h2>{html.escape(title)}</h2>"
            f'<p class="meta">{html.escape(meta)}</p></header>'
            f"{task_overview}"
            f'<div class="chat-window">{"".join(bubbles)}</div>'
            f"</section>"
        )

    task_source = f"Task file: {task_path}" if task_path else "Task file: not provided"
    css = """
    :root {
      color-scheme: light;
      --bg: #f4f6f8;
      --panel: #ffffff;
      --text: #1f2933;
      --muted: #667085;
      --line: #d8dee8;
      --user: #dbeafe;
      --planner: #fef3c7;
      --agent: #dcfce7;
      --task: #f8fafc;
      --shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
      line-height: 1.55;
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(220px, 300px) minmax(0, 1fr);
      min-height: 100vh;
    }
    aside {
      position: sticky;
      top: 0;
      height: 100vh;
      overflow: auto;
      border-right: 1px solid var(--line);
      background: #ffffff;
      padding: 20px;
    }
    aside h1 {
      margin: 0 0 8px;
      font-size: 20px;
      line-height: 1.2;
    }
    aside p {
      margin: 0 0 18px;
      color: var(--muted);
      font-size: 13px;
      word-break: break-word;
    }
    nav {
      display: grid;
      gap: 8px;
    }
    nav a {
      display: block;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      color: var(--text);
      text-decoration: none;
      background: #fafbfc;
      font-size: 14px;
    }
    nav a:hover { background: #eef2f7; }
    nav span {
      display: block;
      margin-top: 3px;
      color: var(--muted);
      font-size: 12px;
    }
    main {
      width: min(1120px, 100%);
      margin: 0 auto;
      padding: 28px;
    }
    .summary {
      margin-bottom: 20px;
      color: var(--muted);
      font-size: 14px;
    }
    .session {
      margin-bottom: 28px;
    }
    .session header {
      margin-bottom: 12px;
    }
    .session h2 {
      margin: 0;
      font-size: 22px;
    }
    .meta {
      margin: 4px 0 0;
      color: var(--muted);
      font-size: 13px;
    }
    .chat-window {
      display: grid;
      gap: 14px;
      padding: 20px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: var(--shadow);
    }
    .task-overview {
      display: grid;
      gap: 10px;
      margin-bottom: 14px;
    }
    .task-card {
      border: 1px solid var(--line);
      border-left: 4px solid #2563eb;
      border-radius: 8px;
      background: var(--task);
      padding: 12px 14px;
    }
    .task-title {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }
    .task-text {
      margin-top: 6px;
      white-space: pre-wrap;
      font-size: 14px;
    }
    .topic-divider {
      border: 1px solid #bfdbfe;
      border-radius: 8px;
      background: #eff6ff;
      padding: 10px 12px;
    }
    .topic-label {
      color: #1d4ed8;
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
    }
    .topic-task {
      margin-top: 4px;
      white-space: pre-wrap;
      color: #1e293b;
      font-size: 13px;
    }
    .message {
      display: grid;
      grid-template-columns: 36px minmax(0, 1fr);
      gap: 10px;
      align-items: start;
      max-width: 92%;
    }
    .message.user {
      margin-left: auto;
      grid-template-columns: minmax(0, 1fr) 36px;
    }
    .message.user .avatar { grid-column: 2; }
    .message.user .bubble { grid-row: 1; grid-column: 1; }
    .avatar {
      display: grid;
      place-items: center;
      width: 36px;
      height: 36px;
      border-radius: 50%;
      background: #111827;
      color: #fff;
      font-weight: 700;
      font-size: 14px;
    }
    .bubble {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
      overflow-wrap: anywhere;
    }
    .user .bubble { background: var(--user); }
    .planner .bubble { background: var(--planner); }
    .agent .bubble { background: var(--agent); }
    .role {
      margin-bottom: 6px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    .content {
      white-space: pre-wrap;
      font-size: 14px;
    }
    .empty {
      margin: 0;
      color: var(--muted);
      font-style: italic;
    }
    @media (max-width: 860px) {
      .layout { display: block; }
      aside {
        position: static;
        height: auto;
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }
      main { padding: 18px; }
      .message { max-width: 100%; }
    }
    """

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Role Message Viewer</title>
  <style>{css}</style>
</head>
<body>
  <div class="layout">
    <aside>
      <h1>Role Message Viewer</h1>
      <p>{html.escape(str(input_path))}</p>
      <p>{html.escape(task_source)}</p>
      <nav>{"".join(nav_items)}</nav>
    </aside>
    <main>
      <p class="summary">Loaded {len(records)} records and extracted {total_messages} User/Planner/Agent messages.</p>
      {"".join(session_blocks)}
    </main>
  </div>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize messages whose content starts with User:, Planner:, or Agent:."
    )
    parser.add_argument("input", type=Path, help="Path to a JSON or JSONL file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output HTML path. Defaults to <input_stem>_chat.html next to input.",
    )
    parser.add_argument(
        "--roles",
        nargs="+",
        default=list(DEFAULT_ROLES),
        help="Role prefixes to extract. Default: User Planner Agent.",
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        help=(
            "Optional persona_tasks JSON/JSONL file. If omitted, the script tries to "
            "find persona_tasks_<timestamp>.jsonl next to persona_train_segmented_<timestamp>.jsonl."
        ),
    )
    parser.add_argument(
        "--no-auto-tasks",
        action="store_true",
        help="Disable automatic discovery of the sibling persona_tasks file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    output_path = args.output
    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_chat.html")
    else:
        output_path = output_path.expanduser().resolve()

    task_path = args.tasks.expanduser().resolve() if args.tasks else None
    if task_path is None and not args.no_auto_tasks:
        task_path = infer_task_path(input_path)
    task_records: list[dict[str, Any]] = []
    if task_path is not None:
        if not task_path.exists():
            raise FileNotFoundError(f"Task file does not exist: {task_path}")
        task_records = load_records(task_path)

    records = load_records(input_path)
    rendered = render_html(input_path, records, args.roles, task_records, task_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
