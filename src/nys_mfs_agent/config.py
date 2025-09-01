
from dataclasses import dataclass
from pathlib import Path
import os

DOC_ID = "New_York_State_Workers_Compensation_Medical_Fee"

ROOT = Path("D:/IIT BBS/Job Resources/Business Optima/pdf-agent")

RUN_DIR      = ROOT / "data" / "runs" / DOC_ID
CHUNKS_DIR   = RUN_DIR / "chunks"
GRAPH_DIR    = RUN_DIR / "graph" / "graph"
RERANKER_DIR = ROOT / "outputs" / "reranker" / DOC_ID
SESSIONS_DIR = RUN_DIR / "sessions"
LOGS_DIR     = RUN_DIR / "logs"

CORE_MODEL_DIR = Path("D:/IIT BBS/Job Resources/Business Optima/pdf-agent/outputs/lora_hf/New_York_State_Workers_Compensation_Medical_Fee_merged")   # merged model dir
ADAPTER_DIR = None                                 # using merged core; no adapter

PROMPTS_DIR  = ROOT / "configs" / "prompts" / "nys_mfs"

SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class Config:
    doc_id: str = DOC_ID
    core_model_dir: Path = CORE_MODEL_DIR
    adapter_dir: Path | None = ADAPTER_DIR
    chunks_dir: Path = CHUNKS_DIR
    graph_dir: Path = GRAPH_DIR
    reranker_dir: Path = RERANKER_DIR
    sessions_dir: Path = SESSIONS_DIR
    logs_dir: Path = LOGS_DIR
    prompts_dir: Path = PROMPTS_DIR

CFG = Config()
