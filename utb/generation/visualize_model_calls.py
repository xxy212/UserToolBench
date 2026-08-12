                      
"""Render model call JSON/JSONL logs as a standalone HTML report."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_records(input_path: Path) -> list[dict[str, Any]]:
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
            if not isinstance(item, dict):
                raise ValueError(f"Line {line_no} is not a JSON object.")
            records.append(item)
        return records

    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
        return parsed
    raise ValueError("Input must be a JSON object, JSON list of objects, or JSONL objects.")


def text_of(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def compact(text: str, limit: int = 220) -> str:
    one_line = " ".join(text.split())
    if len(one_line) <= limit:
        return one_line
    return one_line[: limit - 1].rstrip() + "..."


def prompt_stats(prompt: Any) -> tuple[int, int]:
    if not isinstance(prompt, list):
        return (0, len(text_of(prompt)))
    chars = 0
    for message in prompt:
        if isinstance(message, dict):
            chars += len(text_of(message.get("content")))
        else:
            chars += len(text_of(message))
    return (len(prompt), chars)


def summarize_prompt(prompt: Any) -> str:
    if isinstance(prompt, list) and prompt:
        last = prompt[-1]
        if isinstance(last, dict):
            role = last.get("role", "message")
            content = text_of(last.get("content"))
            return f"{role}: {compact(content)}"
        return compact(text_of(last))
    return compact(text_of(prompt))


def render_text(text: Any) -> str:
    return html.escape(text_of(text))


def render_prompt(prompt: Any) -> str:
    if not isinstance(prompt, list):
        return f'<pre class="text-block">{render_text(prompt)}</pre>'

    parts: list[str] = []
    for idx, message in enumerate(prompt, start=1):
        if isinstance(message, dict):
            role = str(message.get("role", "message"))
            content = message.get("content", "")
        else:
            role = "message"
            content = message
        parts.append(
            '<article class="prompt-message">'
            f'<div class="prompt-role">#{idx} {html.escape(role)}</div>'
            f'<pre class="text-block">{render_text(content)}</pre>'
            "</article>"
        )
    return "".join(parts)


def format_seconds(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.2f}s"
    return "-"


def build_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    roles: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    call_names: Counter[str] = Counter()
    topics: Counter[str] = Counter()
    latency_by_role: dict[str, list[float]] = defaultdict(list)
    latencies: list[float] = []

    for record in records:
        role = str(record.get("role", "unknown"))
        status = str(record.get("status", "unknown"))
        call_name = str(record.get("call_name", "unknown"))
        topic = f"P{record.get('persona_id', '?')}/T{record.get('topic_id', '?')}"
        roles[role] += 1
        statuses[status] += 1
        call_names[call_name] += 1
        topics[topic] += 1
        latency = record.get("latency_seconds")
        if isinstance(latency, (int, float)):
            latencies.append(float(latency))
            latency_by_role[role].append(float(latency))

    return {
        "roles": roles,
        "statuses": statuses,
        "call_names": call_names,
        "topics": topics,
        "latencies": latencies,
        "latency_by_role": latency_by_role,
    }


def render_counter(title: str, counter: Counter[str], limit: int | None = None) -> str:
    items = counter.most_common(limit)
    if not items:
        body = '<p class="empty">No data</p>'
    else:
        body = "".join(
            f'<div class="bar-row"><span>{html.escape(str(name))}</span>'
            f'<strong>{count}</strong></div>'
            for name, count in items
        )
    return f'<section class="stat-panel"><h2>{html.escape(title)}</h2>{body}</section>'


def render_html(input_path: Path, records: list[dict[str, Any]]) -> str:
    stats = build_stats(records)
    latencies = stats["latencies"]
    total_latency = sum(latencies)
    avg_latency = total_latency / len(latencies) if latencies else 0.0
    max_latency = max(latencies) if latencies else 0.0

    unique_roles = sorted(str(role) for role in stats["roles"])
    unique_statuses = sorted(str(status) for status in stats["statuses"])
    role_options = "".join(
        f'<option value="{html.escape(role)}">{html.escape(role)}</option>'
        for role in unique_roles
    )
    status_options = "".join(
        f'<option value="{html.escape(status)}">{html.escape(status)}</option>'
        for status in unique_statuses
    )

    nav_items: list[str] = []
    cards: list[str] = []
    for idx, record in enumerate(records, start=1):
        call_id = record.get("call_id", idx)
        role = str(record.get("role", "unknown"))
        status = str(record.get("status", "unknown"))
        call_name = str(record.get("call_name", "unknown"))
        persona_id = record.get("persona_id", "?")
        topic_id = record.get("topic_id", "?")
        latency = record.get("latency_seconds")
        prompt = record.get("prompt", [])
        output = record.get("output", "")
        prompt_count, prompt_chars = prompt_stats(prompt)
        output_chars = len(text_of(output))
        prompt_summary = summarize_prompt(prompt)
        output_summary = compact(text_of(output))
        card_id = f"call-{html.escape(str(call_id))}"
        search_text = " ".join(
            [
                str(call_id),
                role,
                status,
                call_name,
                str(persona_id),
                str(topic_id),
                prompt_summary,
                output_summary,
            ]
        ).lower()

        nav_items.append(
            f'<a href="#{card_id}" data-role="{html.escape(role)}" '
            f'data-status="{html.escape(status)}">'
            f'#{html.escape(str(call_id))} {html.escape(role)}'
            f'<span>{html.escape(call_name)} · P{html.escape(str(persona_id))}/T{html.escape(str(topic_id))}</span>'
            "</a>"
        )

        meta_items = {
            "status": status,
            "role": role,
            "call_name": call_name,
            "persona/topic": f"{persona_id}/{topic_id}",
            "logged_at": record.get("logged_at", ""),
            "latency": format_seconds(latency),
            "max_tokens": record.get("max_tokens", ""),
            "prompt": f"{prompt_count} messages, {prompt_chars:,} chars",
            "output": f"{output_chars:,} chars",
        }
        meta_html = "".join(
            '<div class="meta-item">'
            f'<span>{html.escape(str(key))}</span>'
            f'<strong>{html.escape(str(value))}</strong>'
            "</div>"
            for key, value in meta_items.items()
        )

        status_class = "ok" if status == "success" else "bad"
        cards.append(
            f'<section id="{card_id}" class="call-card {html.escape(role)} {status_class}" '
            f'data-role="{html.escape(role)}" data-status="{html.escape(status)}" '
            f'data-search="{html.escape(search_text)}">'
            '<header class="call-header">'
            f'<div><h2>Call #{html.escape(str(call_id))}</h2>'
            f'<p>{html.escape(call_name)} · {html.escape(role)} · P{html.escape(str(persona_id))}/T{html.escape(str(topic_id))}</p></div>'
            f'<span class="status {status_class}">{html.escape(status)}</span>'
            "</header>"
            f'<div class="meta-grid">{meta_html}</div>'
            '<div class="preview-grid">'
            f'<div><h3>Prompt summary</h3><p>{html.escape(prompt_summary)}</p></div>'
            f'<div><h3>Output summary</h3><p>{html.escape(output_summary)}</p></div>'
            "</div>"
            '<details class="prompt-details"><summary>Full prompt</summary>'
            f"{render_prompt(prompt)}</details>"
            '<section class="output-section"><h3>Output</h3>'
            f'<pre class="text-block">{render_text(output)}</pre></section>'
            "</section>"
        )

    latency_by_role_rows = []
    for role, values in sorted(stats["latency_by_role"].items()):
        total = sum(values)
        avg = total / len(values) if values else 0.0
        latency_by_role_rows.append(
            f'<tr><td>{html.escape(role)}</td><td>{len(values)}</td>'
            f"<td>{total:.2f}s</td><td>{avg:.2f}s</td><td>{max(values):.2f}s</td></tr>"
        )
    latency_table = (
        '<section class="stat-panel wide"><h2>Latency by role</h2>'
        '<table><thead><tr><th>role</th><th>calls</th><th>total</th><th>avg</th><th>max</th></tr></thead>'
        f'<tbody>{"".join(latency_by_role_rows)}</tbody></table></section>'
    )

    css = """
    :root {
      color-scheme: light;
      --bg: #f5f7fa;
      --panel: #ffffff;
      --text: #1f2937;
      --muted: #667085;
      --line: #d8dee8;
      --accent: #2563eb;
      --success: #15803d;
      --danger: #b42318;
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
      grid-template-columns: minmax(240px, 340px) minmax(0, 1fr);
      min-height: 100vh;
    }
    aside {
      position: sticky;
      top: 0;
      height: 100vh;
      overflow: auto;
      border-right: 1px solid var(--line);
      background: #fff;
      padding: 18px;
    }
    aside h1 { margin: 0 0 8px; font-size: 20px; line-height: 1.2; }
    aside p {
      margin: 0 0 14px;
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    .filters {
      display: grid;
      gap: 8px;
      margin-bottom: 14px;
    }
    input, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--text);
      font: inherit;
      font-size: 13px;
      padding: 9px 10px;
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
      width: min(1280px, 100%);
      margin: 0 auto;
      padding: 28px;
    }
    .summary {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }
    .summary-card, .stat-panel, .call-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: var(--shadow);
    }
    .summary-card {
      padding: 14px;
    }
    .summary-card span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }
    .summary-card strong {
      display: block;
      margin-top: 4px;
      font-size: 22px;
    }
    .stats {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }
    .stat-panel {
      padding: 14px;
      box-shadow: none;
    }
    .stat-panel.wide {
      grid-column: 1 / -1;
    }
    .stat-panel h2 {
      margin: 0 0 10px;
      font-size: 15px;
    }
    .bar-row {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      padding: 6px 0;
      border-top: 1px solid #eef2f7;
      font-size: 13px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      padding: 8px;
      border-top: 1px solid #eef2f7;
      text-align: left;
    }
    .call-list {
      display: grid;
      gap: 18px;
    }
    .call-card {
      padding: 18px;
      scroll-margin-top: 20px;
    }
    .call-card.bad {
      border-color: #fda29b;
    }
    .call-header {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: start;
      margin-bottom: 12px;
    }
    .call-header h2 {
      margin: 0;
      font-size: 21px;
    }
    .call-header p {
      margin: 3px 0 0;
      color: var(--muted);
      font-size: 13px;
    }
    .status {
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
      white-space: nowrap;
    }
    .status.ok {
      background: #dcfce7;
      color: var(--success);
    }
    .status.bad {
      background: #fee4e2;
      color: var(--danger);
    }
    .meta-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
      margin-bottom: 12px;
    }
    .meta-item {
      border: 1px solid #eef2f7;
      border-radius: 8px;
      padding: 8px 10px;
      background: #fafbfc;
      min-width: 0;
    }
    .meta-item span {
      display: block;
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
    }
    .meta-item strong {
      display: block;
      margin-top: 2px;
      overflow-wrap: anywhere;
      font-size: 13px;
      font-weight: 650;
    }
    .preview-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 12px;
    }
    .preview-grid > div {
      border: 1px solid #eef2f7;
      border-radius: 8px;
      background: #f8fafc;
      padding: 10px 12px;
    }
    h3 {
      margin: 0 0 6px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
    }
    .preview-grid p {
      margin: 0;
      font-size: 13px;
      overflow-wrap: anywhere;
    }
    details {
      margin-top: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fcfcfd;
    }
    summary {
      cursor: pointer;
      padding: 10px 12px;
      color: var(--accent);
      font-size: 13px;
      font-weight: 800;
    }
    .prompt-message {
      border-top: 1px solid #eef2f7;
      padding: 10px 12px;
    }
    .prompt-role {
      margin-bottom: 6px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
    }
    .output-section {
      margin-top: 12px;
    }
    .text-block {
      margin: 0;
      max-height: 560px;
      overflow: auto;
      border: 1px solid #eef2f7;
      border-radius: 8px;
      background: #0f172a;
      color: #e5e7eb;
      padding: 12px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font: 12px/1.5 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    }
    .hidden { display: none !important; }
    .empty {
      margin: 0;
      color: var(--muted);
      font-size: 13px;
      font-style: italic;
    }
    @media (max-width: 980px) {
      .layout { display: block; }
      aside {
        position: static;
        height: auto;
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }
      main { padding: 18px; }
      .summary, .stats, .meta-grid, .preview-grid {
        grid-template-columns: 1fr;
      }
    }
    """

    js = """
    const roleFilter = document.getElementById('roleFilter');
    const statusFilter = document.getElementById('statusFilter');
    const searchBox = document.getElementById('searchBox');
    const visibleCount = document.getElementById('visibleCount');
    const cards = Array.from(document.querySelectorAll('.call-card'));
    const navLinks = Array.from(document.querySelectorAll('nav a'));

    function applyFilters() {
      const role = roleFilter.value;
      const status = statusFilter.value;
      const query = searchBox.value.trim().toLowerCase();
      let count = 0;
      cards.forEach((card) => {
        const matchesRole = !role || card.dataset.role === role;
        const matchesStatus = !status || card.dataset.status === status;
        const matchesQuery = !query || card.dataset.search.includes(query);
        const visible = matchesRole && matchesStatus && matchesQuery;
        card.classList.toggle('hidden', !visible);
        if (visible) count += 1;
      });
      navLinks.forEach((link) => {
        const target = document.querySelector(link.getAttribute('href'));
        link.classList.toggle('hidden', !target || target.classList.contains('hidden'));
      });
      visibleCount.textContent = String(count);
    }

    roleFilter.addEventListener('change', applyFilters);
    statusFilter.addEventListener('change', applyFilters);
    searchBox.addEventListener('input', applyFilters);
    applyFilters();
    """

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Model Call Log Viewer</title>
  <style>{css}</style>
</head>
<body>
  <div class="layout">
    <aside>
      <h1>Model Call Log Viewer</h1>
      <p>{html.escape(str(input_path))}</p>
      <div class="filters">
        <input id="searchBox" type="search" placeholder="Search calls, prompts, outputs">
        <select id="roleFilter"><option value="">All roles</option>{role_options}</select>
        <select id="statusFilter"><option value="">All statuses</option>{status_options}</select>
      </div>
      <nav>{"".join(nav_items)}</nav>
    </aside>
    <main>
      <section class="summary">
        <div class="summary-card"><span>Total calls</span><strong id="visibleCount">{len(records)}</strong></div>
        <div class="summary-card"><span>Total latency</span><strong>{total_latency:.2f}s</strong></div>
        <div class="summary-card"><span>Average latency</span><strong>{avg_latency:.2f}s</strong></div>
        <div class="summary-card"><span>Max latency</span><strong>{max_latency:.2f}s</strong></div>
      </section>
      <section class="stats">
        {render_counter("Roles", stats["roles"])}
        {render_counter("Statuses", stats["statuses"])}
        {render_counter("Call names", stats["call_names"], 12)}
        {render_counter("Persona / topics", stats["topics"], 20)}
        {latency_table}
      </section>
      <section class="call-list">{"".join(cards)}</section>
    </main>
  </div>
  <script>{js}</script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize model call JSON/JSONL logs.")
    parser.add_argument("input", type=Path, help="Path to model_calls JSON or JSONL file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output HTML path. Defaults to <input_stem>_log.html next to input.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    output_path = args.output
    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_log.html")
    else:
        output_path = output_path.expanduser().resolve()

    records = load_records(input_path)
    rendered = render_html(input_path, records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
