from __future__ import annotations
from typing import List, Dict, Any, Optional
import json, re, requests
from dataclasses import dataclass
from .config import CFG
from .prompts import load_prompts

_JSON_BLOCK = re.compile(r"\{.*\}", re.S)

@dataclass
class PlanItem:
    qid: str
    type: str
    question: str
    targets: List[str]
    must_cite: bool

@dataclass
class Plan:
    refusal: bool
    reason: str
    tasks: List[PlanItem]
    follow_ups: List[str]

def _ollama_complete(prompt: str, model: str, base_url: str, timeout: int = 120) -> str:
    r = requests.post(
        f"{base_url.rstrip('/')}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.2}},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json().get("response", "")

def _extract_json(s: str) -> Dict[str, Any]:
    m = _JSON_BLOCK.search(s.strip())
    if not m:
        raise ValueError("Planner did not return JSON.")
    return json.loads(m.group(0))

def _coerce_plan(js: Dict[str, Any]) -> Plan:
    refusal = bool(js.get("refusal", False))
    reason = str(js.get("reason") or "")
    tasks = []
    for t in js.get("tasks", []):
        tasks.append(PlanItem(
            qid=str(t.get("qid") or f"Q{len(tasks)+1}"),
            type=str(t.get("type") or "factual"),
            question=str(t.get("question") or "").strip(),
            targets=[str(x) for x in (t.get("targets") or [])],
            must_cite=bool(t.get("must_cite", True)),
        ))
    follow_ups = [str(x) for x in js.get("follow_ups", [])]
    return Plan(refusal=refusal, reason=reason, tasks=tasks, follow_ups=follow_ups)

class DecomposerPlanner:
    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None):
        self.model = model or str(CFG.ollama_planner)
        self.base_url = str(base_url or CFG.ollama_base_url)
        self.prompts = load_prompts()

    def plan(self, user_text: str, summary: str = "", history: str = "") -> Plan:
        ptxt = self.prompts.get("planner") or ""
        prompt = (
            ptxt
            + "\n\nUser input:\n"
            + user_text.strip()
            + ("\n\nConversation summary:\n" + summary.strip() if summary else "")
            + ("\n\nRecent history:\n" + history.strip() if history else "")
            + "\n\nReturn ONLY strict JSON per schema."
        )
        out = _ollama_complete(prompt, self.model, self.base_url)
        js = _extract_json(out)
        plan = _coerce_plan(js)

        # hard guard: if tasks > max or empty questions, trim
        plan.tasks = [t for t in plan.tasks if t.question][: CFG.max_plan_tasks]
        return plan
