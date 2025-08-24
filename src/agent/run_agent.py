from __future__ import annotations
import asyncio, sys, uuid
from .orchestrator import Title17Agent

async def main():
    agent = Title17Agent()
    session_id = f"sess-{uuid.uuid4().hex[:8]}"

    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    if not question:
        question = "Summarize § 114 performance rights caveat. End with [pp. 67–88]."

    print(f"[session] {session_id}")
    print(f"[user] {question}\n")
    async for ev in agent.achat_stream(session_id, question):
        if ev["type"] == "token":
            print(ev["text"], end="", flush=True)
        elif ev["type"] == "final":
            print("\n\n[FINAL]")
            print(ev["text"])
            print("\n[CITATIONS]")
            for c in ev.get("citations", []):
                print(c)
        elif ev["type"] == "error":
            print("\n[ERROR]", ev["text"])

if __name__ == "__main__":
    asyncio.run(main())
