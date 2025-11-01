import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import json_repair

FLAG = False
GEN_INPUT_DUMP_DIR = "test_output"


@dataclass
class PlanStep:
    module: str
    description: str


@dataclass
class Plan:
    steps: List[PlanStep]


def extract_template_and_instruction(text: str) -> Tuple[str, str]:
    """
    Prefer extracting <template>...</template> and <user_input>...</user_input>.
    Fallback to fenced code block for template and use remaining text as instruction.
    """
    tag = lambda name: re.search(
        rf"<\s*{name}\s*>([\s\S]*?)<\s*/\s*{name}\s*>", text, flags=re.IGNORECASE
    )
    mt = tag("template")
    mi = tag("user_input")
    template = (mt.group(1) or "").strip()
    instruction = (mi.group(1) or "").strip()
    return template, instruction


def join_text_messages(messages: List[Any]) -> str:
    return "\n\n".join(m.content for m in messages if getattr(m, "content", ""))


def decode_plan_from_json_like(content: str) -> Plan:
    plan_decoded_obj = json_repair.repair_json(content, return_objects=True) or {
        "steps": []
    }
    steps: List[PlanStep] = []
    for s in plan_decoded_obj.get("steps", []) or []:
        module = (s.get("module") or "").strip()
        desc = (s.get("description") or "").strip()
        if module:
            steps.append(PlanStep(module=module, description=desc))
    return Plan(steps=steps)


def build_plan_overview(plan: Optional[Plan]) -> Tuple[str, str]:
    plan_lines: List[str] = []
    plan_json_lines: List[str] = []
    if plan and plan.steps:
        for step in plan.steps:
            plan_lines.append(f"- {step.module}: {step.description}")
            plan_json_lines.append(
                '  {"module": "'
                + step.module.replace('"', "'")
                + '", "required_h2_from": "'
                + step.description.replace('"', "'")
                + '"}'
            )
    plan_block = "\n".join(plan_lines) if plan_lines else "(none)"
    plan_json_block = (
        "[\n" + ",\n".join(plan_json_lines) + "\n]" if plan_json_lines else "[]"
    )
    return plan_block, plan_json_block


def build_module_queries_block(step_queries: Dict[str, str]) -> str:
    lines: List[str] = []
    for module, query in step_queries.items():
        lines.append(f"- {module}: {query}")
    return "\n".join(lines)


def dump_generation_messages(
    messages: List[Any],
    *,
    out_dir: Optional[str] = None,
    filename_prefix: str = "generation_input",
    chat_id: Optional[str] = None,
    agent: Optional[str] = None,
) -> Optional[str]:
    """
    Dump generation input messages to a human-readable text file.
    Returns the written file path or None if disabled.
    """
    global FLAG
    if not FLAG:
        return None

    target_dir = (
        out_dir
        or os.getenv("GEN_INPUT_DUMP_DIR", "test_output").strip()
        or "test_output"
    )
    os.makedirs(target_dir, exist_ok=True)

    # Compose filename
    parts: List[str] = [filename_prefix]
    if agent:
        parts.append(agent)
    if chat_id:
        parts.append(chat_id)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    parts.append(ts)
    filename = "-".join(p for p in parts if p) + ".txt"
    path = os.path.join(target_dir, filename)

    # Render content
    lines: List[str] = []
    lines.append(f"# Generation Input ({agent or 'unknown agent'})")
    if chat_id:
        lines.append(f"chat_id: {chat_id}")
    lines.append("")

    for idx, m in enumerate(messages):
        source = getattr(m, "source", "unknown")
        content = getattr(m, "content", "") or ""
        lines.append(f"=== message[{idx}] source: {source} ===")
        lines.append(content)
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return path
