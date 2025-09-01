
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class SummaryBuffer:
    max_chars: int = 4000
    summary: str = ""
    buffer: List[Dict[str, str]] = field(default_factory=list)

    def add_turn(self, role: str, content: str):
        self.buffer.append({"role": role, "content": content})

    def as_context(self) -> str:
        parts = []
        if self.summary:
            parts.append(f"[Conversation Summary]\n{self.summary}")
        if self.buffer:
            tail = []
            for m in self.buffer[-8:]:
                tag = "User" if m["role"] == "user" else "Assistant"
                tail.append(f"{tag}: {m['content']}")
            parts.append("\n".join(tail))
        return "\n\n".join(parts)

    def maybe_summarize(self, summarizer_callable):
        # approx character budget
        total = sum(len(m["content"]) for m in self.buffer) + len(self.summary)
        if total <= self.max_chars:
            return
        # summarize buffer and reset it
        convo = "\n".join(f"{m['role']}: {m['content']}" for m in self.buffer)
        prompt = f"Summarize the following chat briefly for future context. Keep only salient facts and decisions.\n\n{convo}\n\nSummary:"
        summary = summarizer_callable(prompt)
        self.summary = (self.summary + "\n" + summary).strip() if self.summary else summary
        self.buffer.clear()
