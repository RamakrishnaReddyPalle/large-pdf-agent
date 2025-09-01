# src/nys_mfs_agent/ui.py
from __future__ import annotations
import os, socket, atexit, uuid, json, threading, queue, asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path

import gradio as gr

from .chat_service import ChatService
from .session_store import list_sessions, load_session, create_session, export_path
from .config import CFG

_RUNNING_DEMO: Optional[gr.Blocks] = None

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
    global _RUNNING_DEMO
    try:
        gr.close_all()
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
    global _RUNNING_DEMO
    close_running()
    os.environ.pop("GRADIO_SERVER_PORT", None)
    port = find_free_port(preferred=preferred_port)
    try:
        ret = demo.launch(server_name=server_name, server_port=port, share=share, prevent_thread_lock=True)
    except OSError:
        port = find_free_port(preferred=None)
        ret = demo.launch(server_name=server_name, server_port=port, share=share, prevent_thread_lock=True)
    _RUNNING_DEMO = demo
    return ret

def _intro_message() -> Dict[str, str]:
    return {
        "role": "assistant",
        "content": "Hi — I’m the **New York MFS Assistant**. Ask about sections and ground rules. Out-of-scope requests will be refused.",
    }

def _list_sessions() -> List[str]:
    return list_sessions()

def _load_session_messages(session_id: str) -> List[Dict[str, Any]]:
    data = load_session(session_id)
    msgs = data.get("messages", [])
    if not msgs:
        msgs = [_intro_message()]
    return msgs

def start_app() -> gr.Blocks:
    svc = ChatService()

    with gr.Blocks(css="""
    #title {font-size: 20px; font-weight: 600; margin-bottom: 8px}
    .small {font-size: 12px; opacity: 0.8}
    """) as demo:
        gr.Markdown(
            "<div id='title'>NY Workers’ Comp Medical Fee Schedule · Local Guardrailed RAG</div>"
            "<div class='small'>Scope-limited to the OFFICIAL NYS Workers’ Comp Medical Fee Schedule.</div>"
        )

        with gr.Row():
            # Sessions column
            with gr.Column(scale=1, min_width=260):
                sid_state = gr.State(value=create_session(intro_assistant=_intro_message()["content"]))
                sessions_dd = gr.Dropdown(choices=_list_sessions(), label="Sessions", value=None, allow_custom_value=False)
                btn_refresh = gr.Button("↻ Refresh", variant="secondary")
                btn_load = gr.Button("📂 Load", variant="secondary")
                btn_new = gr.Button("🔄 New Session", variant="secondary")
                out_file = gr.File(label="Session JSON", visible=False)
                btn_export = gr.Button("⬇️ Export Transcript", variant="secondary")

            # Chat column
            with gr.Column(scale=3):
                chat_state = gr.State(value=[_intro_message()])
                chatbot = gr.Chatbot(
                    value=chat_state.value,
                    type="messages",
                    height=520,
                    show_copy_button=True,
                    bubble_full_width=False,
                    render_markdown=True,
                )
                with gr.Row():
                    txt = gr.Textbox(
                        placeholder="Ask about billing rules, E/M docs, Physical Medicine RVU cap, etc.",
                        lines=2,
                        autofocus=True,
                    )
                    btn_send = gr.Button("Send", variant="primary")

        # ------------- handlers -------------

        btn_refresh.click(lambda: gr.update(choices=_list_sessions()), outputs=[sessions_dd])

        def _load_session(selected: str):
            if not selected:
                return gr.update(), chat_state.value
            msgs = _load_session_messages(selected)
            return gr.update(value=selected), selected, msgs

        btn_load.click(_load_session, inputs=[sessions_dd], outputs=[sessions_dd, sid_state, chatbot])

        def _new_session():
            sid = create_session(intro_assistant=_intro_message()["content"])
            return sid, [_intro_message()], gr.update(value=None, visible=False)

        btn_new.click(_new_session, outputs=[sid_state, chatbot, out_file])

        def _export(sid: str):
            p = export_path(sid)
            return gr.update(value=p, visible=bool(p))

        btn_export.click(_export, inputs=[sid_state], outputs=[out_file])

        # --- streaming helpers ---
        def _sync_stream(user_message: str, session_id: str):
            q: "queue.Queue[object]" = queue.Queue()
            STOP = object()

            async def runner():
                async for ev in svc.stream_answer(session_id, user_message):
                    # 'ev' might be a string token or a dict like {"type":"final","text":"..."}
                    q.put(ev)
                q.put(STOP)

            def run_loop():
                asyncio.run(runner())

            t = threading.Thread(target=run_loop, daemon=True)
            t.start()
            return q, STOP

        # NOTE: single output to Chatbot + clear Textbox during stream
        def _on_send(user_text: str, history: List[Dict[str, str]], sid: str):
            if not user_text or not user_text.strip():
                return gr.update(value=history), gr.update()  # no-op

            # 1) show user msg + empty assistant bubble immediately; clear input
            base = history + [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": ""},  # placeholder for streaming
            ]
            yield gr.update(value=base), gr.update(value="")

            # 2) start background stream of tokens
            q, STOP = _sync_stream(user_text, sid)

            acc = ""
            final_shown = False
            while True:
                ev = q.get()
                if ev is STOP:
                    break

                # Dict 'final' event → replace last assistant bubble with the cleaned answer
                if isinstance(ev, dict) and ev.get("type") == "final":
                    final_text = (ev.get("text") or "").strip()
                    live = base[:-1] + [{"role": "assistant", "content": final_text}]
                    yield gr.update(value=live), gr.update(value="")
                    final_shown = True
                    # keep draining until STOP, but don't update further
                    continue

                # Regular token (only if we haven't shown the final yet)
                if isinstance(ev, str) and not final_shown:
                    acc += ev
                    live = base[:-1] + [{"role": "assistant", "content": acc}]
                    yield gr.update(value=live), gr.update(value="")

        # wire: Chatbot + Textbox (CLEAR)
        txt.submit(_on_send, inputs=[txt, chatbot, sid_state], outputs=[chatbot, txt])
        btn_send.click(_on_send, inputs=[txt, chatbot, sid_state], outputs=[chatbot, txt])

    return demo
