# src/agent/config.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os

# Resolve project ROOT = <repo> (two levels up from this file)
ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    v = os.environ.get(name)
    return Path(v) if v else default


@dataclass(frozen=True)
class AgentConfig:
    # ---- core paths ----
    root: Path = ROOT
    base_model_dir: Path = field(default_factory=lambda: _env_path(
        "TITLE17_BASE_MODEL_DIR",
        ROOT / "models" / "Qwen2.5-1.5B-Instruct"
    ))
    adapter_dir: Path = field(default_factory=lambda: _env_path(
        "TITLE17_ADAPTER_DIR",
        ROOT / "outputs" / "lora_hf" / "title17" / "adapter"
    ))
    reranker_dir: Path = field(default_factory=lambda: _env_path(
        "TITLE17_RERANKER_DIR",
        ROOT / "outputs" / "reranker" / "title17"
    ))
    graph_dir: Path = field(default_factory=lambda: _env_path(
        "TITLE17_GRAPH_DIR",
        ROOT / "outputs" / "graph" / "graph"
    ))
    chunks_dir: Path = field(default_factory=lambda: _env_path(
        "TITLE17_CHUNKS_DIR",
        ROOT / "data" / "chunks"
    ))
    # Where text prompt templates live
    prompts_dir: Path = field(default_factory=lambda: _env_path(
        "TITLE17_PROMPTS_DIR",
        ROOT / "configs" / "prompts"
    ))

    # ---- sessions & logging ----
    sessions_dir: Path = field(default_factory=lambda: _env_path(
        "TITLE17_SESSIONS_DIR",
        ROOT / "outputs" / "sessions"
    ))
    logs_dir: Path = field(default_factory=lambda: _env_path(
        "TITLE17_LOGS_DIR",
        ROOT / "outputs" / "logs"
    ))
    sqlite_path: Path = field(default_factory=lambda: _env_path(
        "TITLE17_SQLITE_PATH",
        ROOT / "outputs" / "logs" / "agent.sqlite"
    ))

    # ---- retrieval knobs ----
    k_nodes: int = int(os.environ.get("TITLE17_K_NODES", 40))
    k_final_nodes: int = int(os.environ.get("TITLE17_K_FINAL_NODES", 6))
    k_each_node: int = int(os.environ.get("TITLE17_K_EACH_NODE", 12))
    k_final_chunks: int = int(os.environ.get("TITLE17_K_FINAL_CHUNKS", 6))

    # ---- generation knobs ----
    max_new_tokens: int = int(os.environ.get("TITLE17_MAX_NEW_TOKENS", 320))
    temperature: float = float(os.environ.get("TITLE17_TEMPERATURE", 0.1))

    # ---- tool LLMs (local only) ----
    ollama_summarizer: str = os.environ.get("TITLE17_OLLAMA_SUMMARIZER", "llama3.2:latest")

    # Optional local orchestration LLMs / settings (kept here to avoid breaking later)
    ollama_base_url: Path | str = "http://localhost:11434"
    ollama_planner: str = "llama3.2:latest"      # or mistral:instruct
    ollama_polisher: str = "llama3.2:latest"     # friendly finalizer
    max_plan_tasks: int = 6

    def ensure_dirs(self) -> None:
        for d in [self.sessions_dir, self.logs_dir, self.prompts_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)


# Singleton
CFG = AgentConfig()
CFG.ensure_dirs()
