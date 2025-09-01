from __future__ import annotations
import argparse, asyncio, sys
from .multi_orchestrator import MultiAgent
from .logger import SessionLogger

async def _amain(q: str):
    agent = MultiAgent()
    log = SessionLogger()
    print("User:", q)
    print("Assistant:", end=" ", flush=True)
    async for tok in agent.ask(q, logger=log):
        print(tok, end="", flush=True)
    print()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("question", nargs="+", help="Question to ask the agent")
    args = p.parse_args()
    q = " ".join(args.question)
    asyncio.run(_amain(q))

if __name__ == "__main__":
    main()
