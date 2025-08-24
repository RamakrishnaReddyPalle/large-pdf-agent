# src/ui/chat.py

from __future__ import annotations
import os, socket, atexit, uuid, json
from pathlib import Path
from typing import List, Dict, Any, Optional

import gradio as gr

from ..agent.orchestrator import Title17Agent
from ..agent.config import CFG
from ..agent.logger import EventLogger

# ------------------------
# Internal server lifecycle
# ------------------------

_RUNNING_DEMO: Optional[gr.Blocks] = None  # track the last launched demo so we can close it

def _port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False

def find_free_port(preferred: Optional[int] = None, start: int = 7860, end: int = 7900) -> int:
    if preferred is not None and _port_is_free(preferred):
        return preferred
    for p in range(start, end + 1):
        if _port_is_free(p):
            return p
    raise OSError(f"No free port found in {start}-{end}")

def close_running():
    """Close any previously launched Gradio demos (best-effort)."""
    global _RUNNING_DEMO
    try:
        gr.close_all()  # supported by recent Gradio versions
    except Exception:
        pass
    if _RUNNING_DEMO is not None:
        try:
            _RUNNING_DEMO.close()
        except Exception:
            pass
        _RUNNING_DEMO = None

atexit.register(close_running)

def launch_app(
    demo: gr.Blocks,
    server_name: str = "127.0.0.1",
    preferred_port: Optional[int] = None,
    share: bool = False,
):
    """
    Safe launcher:
      - closes any previously running demos,
      - unsets GRADIO_SERVER_PORT,
      - finds a free port (respecting preferred_port if free),
      - launches with prevent_thread_lock=True (notebook-friendly),
      - if launch fails due to port, retries with a different free port.
    """
    global _RUNNING_DEMO

    close_running()
    os.environ.pop("GRADIO_SERVER_PORT", None)  # avoid pinning to a busy port

    port = find_free_port(preferred=preferred_port)
    try:
        ret = demo.launch(
            server_name=server_name,
            server_port=port,
            share=share,
            prevent_thread_lock=True,  # keeps notebook cell from blocking
        )
    except OSError:
        # rare race: something grabbed the port between check and launch — retry
        port = find_free_port(preferred=None)
        ret = demo.launch(
            server_name=server_name,
            server_port=port,
            share=share,
            prevent_thread_lock=True,
        )

    _RUNNING_DEMO = demo
    return ret

def start_and_launch(
    server_name: str = "127.0.0.1",
    preferred_port: Optional[int] = None,
    share: bool = False,
):
    """Convenience: build the UI and launch it safely in one call."""
    demo = start_app()
    launch_app(demo, server_name=server_name, preferred_port=preferred_port, share=share)
    return demo

# ------------------------
# UI wiring
# ------------------------

def _list_sessions() -> List[str]:
    CFG.sessions_dir.mkdir(parents=True, exist_ok=True)
    return sorted(p.stem for p in CFG.sessions_dir.glob("sess-*.json"))

def _load_session_messages(session_id: str) -> List[Dict[str, Any]]:
    fp = CFG.sessions_dir / f"{session_id}.json"
    if not fp.exists():
        return []
    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
        msgs = data.get("messages", [])
        out: List[Dict[str, Any]] = []
        for m in msgs:
            role = m.get("role") or "user"
            content = m.get("content") or ""
            out.append({"role": role, "content": content})
        return out
    except Exception:
        return []

def start_app() -> gr.Blocks:
    """
    Build the Gradio UI (no launch here). Use launch_app() or start_and_launch() to run it safely.
    """
    agent = Title17Agent()
    elog = EventLogger()

    def _new_session_id() -> str:
        return f"sess-{uuid.uuid4().hex[:8]}"

    def _intro_message() -> Dict[str, str]:
        return {
            "role": "assistant",
            "content": (
                "Hi — I’m the **Title 17 Assistant** (U.S. Copyright Law). "
                "Ask about sections like §107 (fair use), §108 (libraries), §110 (classroom), or §114 (sound recordings). "
                "Out-of-scope requests will be politely declined."
            ),
        }

    with gr.Blocks(css="""
    #title {font-size: 20px; font-weight: 600; margin-bottom: 8px}
    .small {font-size: 12px; opacity: 0.8}
    """) as demo:
        gr.Markdown(
            "<div id='title'>Title 17 Assistant · Local, Guardrailed RAG</div>"
            "<div class='small'>Focus: U.S. Copyright Law (Title 17). Out-of-scope requests are refused.</div>"
        )

        # --- left column: sessions panel ---
        with gr.Row():
            with gr.Column(scale=1, min_width=260):
                sid_state = gr.State(value=_new_session_id())
                sessions_dd = gr.Dropdown(
                    choices=_list_sessions(),
                    label="Sessions",
                    value=None,
                    allow_custom_value=False,
                )
                btn_refresh = gr.Button("↻ Refresh", variant="secondary")
                btn_load = gr.Button("📂 Load", variant="secondary")
                btn_new = gr.Button("🔄 New Session", variant="secondary")
                out_file = gr.File(label="Session JSON", visible=False)
                btn_export = gr.Button("⬇️ Export Transcript", variant="secondary")

            # --- right column: chat ---
            with gr.Column(scale=3):
                chat_state = gr.State(value=[_intro_message()])  # list of {role, content}
                chatbot = gr.Chatbot(
                    value=chat_state.value,
                    type="messages",
                    height=520,
                    show_copy_button=True,
                    bubble_full_width=False,
                )
                with gr.Row():
                    txt = gr.Textbox(
                        placeholder="Ask about Title 17… e.g., “What does §107 say about fair use?”",
                        lines=2,
                        autofocus=True,
                    )
                    btn_send = gr.Button("Send", variant="primary")

        # ------------- handlers -------------

        def _refresh_sessions():
            return gr.update(choices=_list_sessions())

        btn_refresh.click(_refresh_sessions, outputs=[sessions_dd])

        def _load_session(selected: str):
            if not selected:
                return gr.update(), chat_state.value
            sid_state.value = selected
            msgs = _load_session_messages(selected)
            if not msgs:
                msgs = [_intro_message()]
            return gr.update(value=selected), msgs

        btn_load.click(_load_session, inputs=[sessions_dd], outputs=[sid_state, chatbot])

        def _new_session():
            new_sid = _new_session_id()
            # reset chat area with intro message
            return new_sid, [_intro_message()], gr.update(value=None, visible=False)

        btn_new.click(_new_session, outputs=[sid_state, chatbot, out_file])

        def _export(sid: str):
            fp = CFG.sessions_dir / f"{sid}.json"
            if not fp.exists():
                return gr.update(value=None, visible=False)
            return gr.update(value=str(fp), visible=True)

        btn_export.click(_export, inputs=[sid_state], outputs=[out_file])

        # streaming send
        def _sync_stream(user_message: str, session_id: str):
            import threading, queue, asyncio
            q: "queue.Queue[str|None]" = queue.Queue()
            STOP = None

            async def runner():
                async for ev in agent.achat_stream(session_id, user_message):
                    if ev["type"] == "token":
                        q.put(ev["text"])
                q.put(STOP)

            def run_loop():
                asyncio.run(runner())

            t = threading.Thread(target=run_loop, daemon=True)
            t.start()

            acc = ""
            while True:
                chunk = q.get()
                if chunk is STOP:
                    break
                acc += chunk
                yield acc  # incremental assistant message

        def _on_send(user_text: str, history: List[Dict[str, str]], sid: str):
            if not user_text or not user_text.strip():
                return gr.update(), history  # no-op

            # echo user message to UI
            updated = history + [{"role": "user", "content": user_text}]

            # stream assistant message
            stream = _sync_stream(user_text, sid)
            for partial in stream:
                yield gr.update(value=updated + [{"role": "assistant", "content": partial}]), updated

        # wire both textbox Enter and Send button
        txt.submit(_on_send, inputs=[txt, chatbot, sid_state], outputs=[chatbot, chatbot])
        btn_send.click(_on_send, inputs=[txt, chatbot, sid_state], outputs=[chatbot, chatbot])

    return demo
