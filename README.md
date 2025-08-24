  # **TITLE 17 COPILOT**

*Local, guardrailed RAG & PEFT fine-tuning for U.S. Copyright Law (17 U.S.C.)*

> A focused research assistant for **Title 17** (U.S. Copyright Law).
> Everything runs **locally**: ingestion, chunking + embeddings, SFT data gen, **LoRA/PEFT** fine-tuning of Qwen2.5-1.5B, retrieval with **BM25 + CE reranker**, an **agent** orchestrating multiple LLMs via **Ollama**, streaming answers with **citations**, session memory, and a **Gradio** chat UI. Dockerized for one-command spin-ups.
---
<p align="center">
  <img src="assets/Screenshot 2025-08-24 182531.png" style="width: 100%;">
</p>

---

## Table of Contents

* [Project Summary](#project-summary)
* [Flowchart](#flowchart)
* [Data & Preprocessing](#data--preprocessing)
* [SFT Data Generation](#sft-data-generation)
* [Fine-Tuning the Core LLM (PEFT/LoRA)](#fine-tuning-the-core-llm-peftlora)
* [Retrieval Stack](#retrieval-stack)
* [Agent Architecture (multi-LLM, guardrails, memory, logging)](#agent-architecture-multi-llm-guardrails-memory-logging)
* [Prompts](#prompts)
* [Gradio UI](#gradio-ui)
* [Run Locally (venv)](#run-locally-venv)
* [Models (download & placement)](#models-download--placement)
* [Run with Docker](#run-with-docker)
* [Configuration](#configuration)
* [Troubleshooting](#troubleshooting)
* [Why it’s fast, what to scale next](#why-its-fast-what-to-scale-next)
* [License](#license)

---

## **Project Summary**

I worked end-to-end on **Title 17 (U.S. Code)**, using a clean, documented pipeline. Below is a succinct step-by-step recap of the notebooks and what they produced. (Exact file names may differ; the stages and outputs match.)

1. **Ingestion & PDF prep**

   * **Source**: `title17.pdf` (U.S. Copyright Law).
   * **Parsed** text, preserved **section headings** & **page numbers**.
   * Stored parsed text blocks with metadata (heading\_path, page\_start/page\_end).

2. **Chunking**

   * Built **hierarchical chunks** aligned to sections/subsections.
   * Persisted to `data/chunks/` (each JSON contains `text`, `pages`, `node_id`, `chunk_id`, `section`, `heading_path`).

3. **Embeddings (CPU)**

   * Model: **BAAI/bge-base-en-v1.5** (`SentenceTransformer` on CPU).
   * Normalized embeddings; stored in local vector store (Chroma or plain files) together with chunk metadata.

4. **BM25 baseline retriever**

   * Indexed chunk texts in a BM25 store for fast lexical retrieval.
   * Built **hybrid** retrieval: BM25 → candidate nodes → refine with embeddings/reranker.

5. **Reranker training (Cross-Encoder)**

   * Trained a **CE reranker** on automatically generated positives/negatives (from your notebook).
   * Saved to `outputs/reranker/title17/`.

6. **RAG retrieval sanity check**

   * Issued sample queries (e.g., “What does §107 say about fair use?”).
   * Inspected top hits and distances. Confirmed sections and pages make sense.

7. **SFT data generation (40 sections)**

   * Why: created **supervised fine-tuning (SFT)** pairs to nudge the LLM towards Title 17 style & answers.
   * How: auto-drafted Q/A pairs per section; one prompt per major concept; ensured **answer framing** with citations and page hints.
   * Data looks like **instruction** (question), **input** (context if used), **output** (target answer).
   * Saved raw SFT JSONL to `outputs/sft/title17/raw/`.

8. **Alpaca schema + splits**

   * Converted to **Alpaca**-style (instruction/input/output).
   * Created train/dev/test splits (counts saved alongside; see `outputs/sft/title17/`).

9. **Core LLM setup**

   * Base: **Qwen2.5-1.5B-Instruct** (local).
   * Tokenizer padded token set to EOS for stability.

10. **PEFT/LoRA fine-tuning**

    * Trained **LoRA adapter** on SFT (keeps footprint small; no full-model finetune).
    * **Outputs**: adapter in `outputs/lora_hf/title17/adapter`.
    * I also experimented with a merged model (base + adapter weights); final agent **uses adapter-only** model for best behavior.

11. **Closed-book evaluation**

    * Asked queries **without** retrieval—gauge model recall.
    * Conclusion: merged model sometimes produced undesired responses; **adapter-only** gave more controlled & concise answers.

12. **RAG evaluation (Ragas)**

    * Attempted **faithfulness / answer relevancy / context precision / recall**.
    * Due to timeouts/parsing on local judge LLM, i deferred completing this to a future separate notebook.
    * I did verify retrieval quality & ensured citations map to the right sections.

13. **Agent build, streaming, and UI**

    * Built `src/agent` and `src/ui`: modular, **multi-LLM** orchestration, **guardrails**, **memory**, **SQLite logging**, **Gradio** UI with session management & streaming.
---

## **Flowchart**
```pgsql
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                              TITLE 17 PIPELINE                                              │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

[ Ingestion & Prep ]
┌───────────────┐      ┌──────────────────────────────┐      ┌───────────────────────────────────────────────┐
│ Title17.pdf   ├─────▶│ Parse & Metadata             ├─────▶│ Chunking (hierarchical, section-aware)       │
│ (source)      │      │ (headings, page numbers)     │      │ (section ids, page spans, clean text)        │
└───────────────┘      └──────────────────────────────┘      └───────────────────────────────────────────────┘
                                                                     │
                                                                     │
                                           ┌─────────────────────────┴─────────────────────────┐
                                           │                                                   │
                                   ┌──────────────────────────┐                        ┌───────────────────────┐
                                   │ Embeddings (BGE-base)    │                        │ BM25 Index            │
                                   │ vectors for each chunk   │                        │ lexical index         │
                                   └──────────────┬───────────┘                        └───────────┬──────────┘
                                                  │                                             │
                                                  └───────────────┬─────────────────────────────┘
                                                                  │
                                                        ┌─────────▼───────────┐
                                                        │ Hybrid Retrieval    │
                                                        │ BM25 → CE reranker  │
                                                        │ (Top-K contexts)    │
                                                        └─────────┬───────────┘
                                                                  │
                                                                  │
       [ SFT & Training ]                                         │                        [ Multi-LLM Agent ]
┌────────────────────────────┐    ┌──────────────────────────┐    │    ┌───────────────────────────────┐
│ SFT Generation (~40 sect.) │───▶│ Alpaca conversion +      │    │    │ Core Prompt Build            │
│ Q/A pairs per section      │    │ train/dev/test splits    │    │    │ (system + style + citations  │
└──────────────┬─────────────┘    └──────────────┬───────────┘    │    │ + retrieved contexts)        │
               │                                 │                │    └───────────────┬──────────────┘
               │                                 ▼                │                    │
               │                    ┌──────────────────────────┐  │           ┌────────▼─────────┐
               │                    │ PEFT/LoRA Fine-tune      │  │           │ Core LLM         │
               │                    │ Qwen2.5-1.5B (CPU)       │  │           │ (Qwen + Adapter) │
               │                    └───────────┬──────────────┘  │           └────────┬─────────┘
               │                                │                 │                    │ (token stream)
               │                    ┌───────────▼──────────────┐  │           ┌────────▼──────────────┐
               │                    │ Adapter stored (LoRA)    │  │           │ Raw Answer (stream)   │
               │                    │ outputs/lora_hf/title17  │  │           └────────┬──────────────┘
               │                    └──────────────────────────┘  │                    │
               │                                                  │           ┌────────▼──────────────┐
               │                                                  │           │ Polisher LLM (Ollama) │
               │                                                  │           │ Human-friendly +       │
               │                                                  │           │ formatting + citations │
               │                                                  │           └────────┬──────────────┘
               │                                                  │                    │
               │                                                  │                    ▼
               │                 ┌─────────────────────────────────────────────────────────────────────┐
               │                 │ Planner LLM (Ollama)                                               │
               └────────────────▶│ Splits heavy/compound user asks into solvable sub-queries;         │
                                 │ routes each sub-query through retrieval → core LLM; aggregates.    │
                                 └─────────────────────────────────────────────────────────────────────┘

[ UI & Logging ]
┌──────────────────────────────┐         ┌───────────────────────────┐          ┌───────────────────────────┐
│ Gradio UI (streaming chat)   │ ──────▶ │ Session JSON transcripts  │ ◀──────── │ SQLite Logs               │
│ intro message + send button  │         │ outputs/sessions/*.json   │          │ conversations/messages/   │
│ load previous sessions       │         └───────────────────────────┘          │ events (agent.sqlite)     │
└──────────────────────────────┘                                                       └─────────────────────┘

Legend:
- Retrieval path:  Chunking → Embeddings/BM25 → Hybrid Retrieval → Core Prompt → Core LLM → Raw Answer → Polisher → UI
- Training path:   SFT Gen → Alpaca + Splits → PEFT/LoRA Fine-tune → Adapter stored (used by Core LLM at inference)
- Planner path:    Planner LLM can split heavy inputs and loop back into Retrieval/Core for each sub-task
```

---

## **Data & Preprocessing**

* **Document**: Title 17 of the U.S. Code (copyright).
  *Place the source PDF under `data/` if you want to re-ingest; the repo works with the pre-chunked data under `data/chunks/`.*

* **Parsing**: extract text while preserving:

  * `heading_path` (e.g., `Title > Chapter > Section`),
  * `page_start` / `page_end`.

* **Chunking**: hierarchical, section-aware chunks.
  Each chunk JSON contains:
  `{"text", "pages": [start, ...], "node_id", "chunk_id", "section", "heading_path"}`.

---

## **SFT Data Generation**

* **Goal**: teach the model to answer **in a Title 17-specific style**, with page-reference habits and careful phrasing.
* **Coverage**: \~**40 sections** sampled for breadth; auto-generated Q/A pairs using seeds per section.
* **Format**: Alpaca-style triplets:

  ```json
  {
    "instruction": "What does §107 say about fair use? End with [pp. 40–41].",
    "input": "",
    "output": "Answer:\n... (concise, cites pages)\n\nCitations:\n- [pp. 40–41]"
  }
  ```
* **Splits**: saved to `outputs/sft/title17/` (train/dev/test).
  *Exact counts live in the split files; we didn’t hardcode them in this README.*

---

## **Fine-Tuning the Core LLM (PEFT/LoRA)**

* **Base**: `models/Qwen2.5-1.5B-Instruct/`
* **Why PEFT/LoRA**:

  * Tiny parameter footprint, fast to train on CPU, no full-model rewrite.
  * Swap adapters per domain; share the same base model.
* **Artifacts**:

  * **Adapter** (final choice for inference): `outputs/lora_hf/title17/adapter/`
  * **Merged** (optional; tested but not used in production agent).
* **Hyperparameters**: I used a **sane LoRA config for Qwen 1.5B** and CPU-friendly training setup.
  *For exact values, see the generated `adapter/config.json` in the above folder.*
* **Why adapter-only in the agent**: produced **more controlled** answers; merged model occasionally drifted.

---

## **Retrieval Stack**

* **Embeddings**: `BAAI/bge-base-en-v1.5` (SentenceTransformer; CPU).
* **BM25**: fast lexical recall of candidate **nodes**.
* **CE Reranker**: re-scores candidates using a fine-tuned cross-encoder (your `outputs/reranker/title17/`).
* **Knobs** (see `CFG` in `src/agent/config.py`):

  * `k_nodes=40` → `k_final_nodes=6` → `k_each_node=12` → `k_final_chunks=6`
* **Sanity checks**:  I verified that queries like “§107 fair use” pull the correct slices, with page ranges.

---

## **Evaluation (closed-book & retrieval sanity; ragas deferred)**

* **Closed-book eval**: I queried the model **without** retrieval.

* Takeaway: adapter-only was **more stable** than merged. Results are concise but textbook-ish (hence our Output Polisher).
* **RAG sanity**: I printed top chunks, distances, and pages; verified correct sections.
* **Ragas**: set up with local judge LLM & offline embeddings; encountered **timeouts/parsing** on our environment. Deferred to a clean, future notebook.

---

## **Agent Architecture (multi-LLM, guardrails, memory, logging)**

The agent is organized and **fully local**. Key pieces:

* **Core LLM** (inference): Qwen2.5-1.5B + **LoRA adapter** (Title 17-tuned).
  Loaded via `src/agent/llms.py` → `load_core_llm()`; streaming generator `stream_generate`.

* **Planner LLM** (Ollama): breaks **heavy / multi-part** user prompts into manageable **sub-tasks** the Core can answer one by one.
  Model: `llama3.2:latest` (configurable via `CFG.ollama_planner`).

* **Polisher LLM** (Ollama): turns Core LLM’s raw answer into a **human-friendly** response that respects style rules & includes citations.
  Model: `llama3.2:latest` (configurable via `CFG.ollama_polisher`).

* **Guardrails**:
  `src/agent/policy.py::guard_title17_scope()` checks **scope** using a Title-17 lexicon/regex.
  Out-of-scope → polite refusal. These refusals are **logged** and **visible in the UI** now.

* **Retrieval**:
  `src/agent/retriever.py::HierBM25CEReranker` does BM25 → CE re-ranking.

* **Prompts**:
  Stored in `configs/prompts/` and loaded by `src/agent/prompts.py`.
  I use `system.txt`, `style_rules.txt`, `answer_with_citations.txt` (Core);
  `planner.txt` (Planner), `polisher.txt` (Polisher).

* **Memory**:

  * **Per-session JSON** transcripts in `outputs/sessions/sess-*.json`.
  * **Optional** LangChain `ConversationSummaryBufferMemory` with local Ollama to keep a rolling summary for context (doesn’t alter Title 17 identity).

* **Logging**:
  SQLite db at `outputs/logs/agent.sqlite` with three tables:

  * `conversations(id, created_ts)`,
  * `messages(id, session_id, ts, role, content)`,
  * `events(id, ts, event, payload)`.

* **Streaming**:
  The orchestrator (`src/agent/orchestrator.py::Title17Agent`) yields token events as it generates, so the UI renders tokens live.

### **Multi-LLM flow (conceptual)**

1. **Guardrail** checks user input.
2. **Planner (Ollama)** optionally splits a heavy prompt into **N tasks**.
3. For each task:

   * Retrieve chunks (BM25 → CE),
   * Build Core prompt,
   * **Core LLM** streams the answer for that sub-task.
4. Aggregate sub-answers.
5. **Polisher (Ollama)** rewrites into a single, helpful, **human-readable** response with required **citations**.
6. Return to UI; log to SQLite + session JSON.

---

## **Prompts**

* `configs/prompts/system.txt` – your assistant identity (Title 17, no off-topic work).
* `configs/prompts/style_rules.txt` – tone, formatting, citations, page range style.
* `configs/prompts/answer_with_citations.txt` – answer scaffold with explicit “Answer:” and “Citations:” sections.
* `configs/prompts/planner.txt` – how to split heavy user inputs into multiple sub-tasks.
* `configs/prompts/polisher.txt` – how to rewrite raw core outputs into a friendly final answer (preserve citations).

> You can tweak these without touching code. The orchestrator consumes them at runtime.

---

## **Gradio UI**

* File: `src/ui/chat.py` → `start_app()`
* Features:

  * **Streaming** response tokens,
  * **Welcome message** on first render,
  * **Send button** and Enter key,
  * **Session IDs** (auto-generated) shown in the header,
  * **Session picker** to load previous transcripts (from JSON files),
  * **Export** transcript button,
  * Guardrail refusals appear to the user (not just logs).

Run directly:

```bash
python -m src.ui.chat
# Default on http://127.0.0.1:7860
```

---

## **Run Locally (venv)**

1. **Clone**

   ```bash
   git clone <your-repo-url>.git
   cd pdf-agent
   ```

2. **Create venv + install**

   ```bash
   python -m venv .venv
   . .venv/bin/activate           # Windows: .venv\Scripts\activate
   pip install --upgrade pip

   # Either:
   pip install -r requirements.txt
   # Or install main deps manually (matches Dockerfile):
   pip install \
     "torch==2.4.0" \
     "transformers>=4.43,<5" \
     "accelerate>=0.31" \
     "peft>=0.11" \
     "sentence-transformers>=2.2.2" \
     "datasets>=2.19" \
     "rank-bm25>=0.2.2" \
     "langchain>=0.3.2" \
     "langchain-community==0.3.2" \
     "langchain-ollama>=0.1.0" \
     "gradio>=4.44,<5" \
     "pydantic>=2.7" \
     "requests>=2.31" \
     "numpy>=1.26"
   ```

3. **Install Ollama & models** (for Planner/Polisher; Core runs via Transformers)

   ```bash
   # https://ollama.com/download
   ollama serve   # keep running in a terminal
   ollama pull llama3.2:latest
   # (optional) ollama pull mistral:instruct
   ```

4. **Place models/data**

   * Base LLM: `models/Qwen2.5-1.5B-Instruct/`
   * LoRA adapter: `outputs/lora_hf/title17/adapter/`
   * Reranker: `outputs/reranker/title17/`
   * Chunks: `data/chunks/`
   * Prompts: `configs/prompts/`

5. **Run the UI**

   ```bash
   python -m src.ui.chat
   ```

---

## **Models (download & placement)**

* **Qwen2.5-1.5B-Instruct** (Hugging Face). Put under:

  ```
  models/Qwen2.5-1.5B-Instruct/
  ```
* **SentenceTransformer** `BAAI/bge-base-en-v1.5` is fetched on demand and cached (HuggingFace cache).
* **Ollama**: `llama3.2:latest` (Planner & Polisher).
* **LoRA Adapter**: produced by your fine-tune notebook:

  ```
  outputs/lora_hf/title17/adapter/   # used by the agent
  ```

> I intentionally **use the adapter-only** path in production (better behavior than merged in our tests).

---

## **Run with Docker**

### Build & compose

* Ensure **Docker Desktop** is running (Windows/macOS) or Docker Engine (Linux).

```bash
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml up
```

* App UI: `http://localhost:7860`
* Ollama API: `http://localhost:11434`

**Tip (ports busy):**

```bash
# Change the external port without editing code:
GRADIO_SERVER_PORT=7870 docker compose -f docker/docker-compose.yml up
# (compose file can map "7870:7860" if you want a stable external port)
```

### **Context size (important)**

I shipped a `.dockerignore` so your 13+ GB local models/outputs don’t get copied into the image.
If you want the container to see your host models/data, **use volumes** (already present in `docker/docker-compose.yml`):

```yaml
volumes:
  - ../models:/app/models:ro
  - ../data:/app/data:ro
  - ../configs/prompts:/app/configs/prompts:ro
  - app_outputs:/app/outputs
  - hf_cache:/root/.cache/huggingface
```

---

## **Configuration**

Edit **`src/agent/config.py`** or set environment variables:

| Env Var                     | Purpose           | Default                           |
| --------------------------- | ----------------- | --------------------------------- |
| `TITLE17_BASE_MODEL_DIR`    | Base LLM path     | `models/Qwen2.5-1.5B-Instruct`    |
| `TITLE17_ADAPTER_DIR`       | LoRA adapter      | `outputs/lora_hf/title17/adapter` |
| `TITLE17_RERANKER_DIR`      | CE reranker       | `outputs/reranker/title17`        |
| `TITLE17_CHUNKS_DIR`        | Chunked data      | `data/chunks`                     |
| `TITLE17_PROMPTS_DIR`       | Prompts dir       | `configs/prompts`                 |
| `TITLE17_SESSIONS_DIR`      | Session JSONs     | `outputs/sessions`                |
| `TITLE17_SQLITE_PATH`       | SQLite DB path    | `outputs/logs/agent.sqlite`       |
| `TITLE17_K_*`               | Retrieval knobs   | `40/6/12/6`                       |
| `TITLE17_MAX_NEW_TOKENS`    | Generation tokens | `320`                             |
| `TITLE17_TEMPERATURE`       | Core temp         | `0.1`                             |
| `TITLE17_OLLAMA_SUMMARIZER` | LC memory LLM     | `llama3.2:latest`                 |

---

## **Troubleshooting**

* **Gradio port in use**: set `GRADIO_SERVER_PORT` or pass `server_port=` to `launch()`.
* **Ollama not reachable**: ensure `ollama serve` is running on `localhost:11434`, and models are pulled.
* **Slow streaming** (CPU): reduce `max_new_tokens` and retrieval depth; or switch to a GPU image later.
* **Ragas errors**: I deferred Ragas to a clean notebook due to local judge timeouts; the RAG pipeline itself is healthy.

---

## **Why it’s fast, what to scale next**

* **Fast** because:

  * LoRA adapter inference (tiny overhead),
  * Focused retrieval (BM25 → CE rerank, limited to top chunks),
  * Streaming tokenization with `do_sample=False` & low temperature,
  * Small, local **planner/polisher** models (llama3.2) offloaded to **Ollama**.

* **Scale up** (if you have more compute/time):

  * Train on **more sections** (cover entire Title 17) → better coverage & style,
  * Increase SFT epochs/batch size → better closed-book recall,
  * Use **stronger reranker** or multi-stage reranking → higher precision contexts,
  * Move to a larger base model (e.g., 7B) + LoRA → richer synthesis,
  * Finish **Ragas** eval with a reliable local judge and stable embeddings.
---

## **Examples**

### **Retrieval sanity (Python)**

```python
from sentence_transformers import SentenceTransformer
st = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")

def embed_query(q: str):
    return st.encode(["query: " + q.strip()], normalize_embeddings=True)[0].tolist()

res = coll.query(query_embeddings=[embed_query("What does §107 say about fair use?")],
                 n_results=8, include=["documents","metadatas","distances"])

for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
    print(f"[dist={dist:.3f} | p.{meta.get('page_start')}] {meta.get('heading_path')}")
    print(doc[:260], "…")
```

### **Agent smoke test (Notebook)**

```python
import uuid
from src.agent.orchestrator import Title17Agent

async def run_once(q: str):
    agent = Title17Agent()
    sid = f"sess-{uuid.uuid4().hex[:8]}"
    final = []
    async for ev in agent.achat_stream(sid, q):
        if ev["type"] == "token":
            print(ev["text"], end="")
            final.append(ev["text"])
        elif ev["type"] == "final":
            print("\n\n[FINAL]\n", ev["text"])
            print("[CITATIONS]", ev.get("citations", []))
    return sid, "".join(final)

await run_once("What does §107 say about fair use? End with [pp. 40–41].")
```

### **Launch UI**

```bash
python -m src.ui.chat
```

---

## **License**

This repository ships code and prompts for building a **local** Title 17 assistant.
You are responsible for lawfulness of usage and respecting the licenses of upstream models and data.

---

### **Final notes**

* This project demonstrates a **complete** domain RAG + PEFT fine-tune loop: from raw PDF to agentic, multi-LLM UI with citations and guardrails.
* The agent stays **in character**: if the input isn’t clearly about **Title 17**, it politely declines—visible both in the UI and the logs.
* Despite running on CPU, the system streams responses quickly thanks to a compact base model + adapter, tight retrieval, and lean prompts.
