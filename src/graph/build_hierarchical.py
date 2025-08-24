# src/graph/build_hierarchical.py
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from collections import defaultdict

TOC_LINE = re.compile(r"^\s*\|.*\|\s*$")
SECTION_LINE = re.compile(r"^\s*§+\s*([0-9A-Za-z\-]+)[^\n]*", re.UNICODE)

def clean_text(t: str) -> str:
    if not isinstance(t, str):
        return ""
    # drop obvious TOC blocks (many pipes)
    lines = []
    for ln in t.splitlines():
        if TOC_LINE.match(ln):
            continue
        # drop ultra-short page artifacts
        if len(ln.strip()) <= 1:
            continue
        lines.append(ln)
    return "\n".join(lines).strip()

def guess_section(obj: dict) -> str | None:
    # 1) prefer explicit field if present
    sec = obj.get("section")
    if isinstance(sec, str) and sec.strip():
        return sec.strip()

    # 2) try to parse a § line from text
    txt = obj.get("text") or obj.get("content") or ""
    if not isinstance(txt, str):
        return None
    for ln in txt.splitlines():
        m = SECTION_LINE.match(ln)
        if m:
            # use the full line as node name for readability
            return ln.strip()

    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_node_chars", type=int, default=400)  # drop tiny nodes
    args = ap.parse_args()

    chunks_dir = Path(args.chunks_dir)
    out_dir    = Path(args.out_dir) / "graph"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all chunks
    chunks = []
    for fp in sorted(chunks_dir.glob("*.jsonl")):
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                cid  = obj.get("id") or obj.get("chunk_id")
                text = clean_text(obj.get("text") or obj.get("content") or "")
                if not text:
                    continue
                # stash essentials
                chunks.append({"id": cid, "text": text, "obj": obj})

    # Group by section
    groups: dict[str, dict] = defaultdict(lambda: {"chunk_ids": [], "texts": []})
    misc_key = "MISC (no § heading detected)"
    for ch in chunks:
        name = guess_section(ch["obj"])
        if not name:
            name = misc_key
        groups[name]["chunk_ids"].append(ch["id"])
        groups[name]["texts"].append(ch["text"])

    # Build nodes (drop ultra tiny)
    nodes = []
    root_children = []
    for i, (name, data) in enumerate(groups.items(), 1):
        text = "\n\n".join(data["texts"]).strip()
        if len(text) < args.min_node_chars:
            # fold very small fragments into MISC
            if name != misc_key:
                groups[misc_key]["chunk_ids"].extend(data["chunk_ids"])
                groups[misc_key]["texts"].append(text)
            continue

    # Rebuild after folding
    nodes = []
    root_children = []
    for i, (name, data) in enumerate(groups.items(), 1):
        text = "\n\n".join(t for t in data["texts"] if t).strip()
        if not text:
            continue
        node_id = f"SEC-{i:05d}"
        nodes.append({
            "id": node_id,
            "name": name,
            "level": 1,
            "parent": "ROOT",
            "children": [],
            "chunk_ids": data["chunk_ids"],
        })
        root_children.append(node_id)

    # If absolutely nothing grouped, make a single ROOT bucket
    if not nodes and chunks:
        node_id = "SEC-00001"
        all_text = "\n\n".join([c["text"] for c in chunks])
        nodes = [{
            "id": node_id,
            "name": "ALL",
            "level": 1,
            "parent": "ROOT",
            "children": [],
            "chunk_ids": [c["id"] for c in chunks],
        }]
        root_children = [node_id]
        node_texts = [{"node_id": node_id, "name": "ALL", "level": 1, "text": all_text}]
    else:
        # write concatenated texts per node
        node_texts = []
        for n in nodes:
            sect_texts = []
            for cid in n["chunk_ids"]:
                # you can skip looking up each chunk again: we already have text in groups
                pass
            # We kept concatenated text in groups; recompute quickly:
            # Build fast lookup from node name back to joined text:
            pass  # will fill below

        # Build a quick map name->text from groups
        name_to_text = {name: "\n\n".join(v["texts"]).strip() for name, v in groups.items()}
        node_texts = [{
            "node_id": n["id"],
            "name": n["name"],
            "level": n["level"],
            "text": name_to_text.get(n["name"], "")
        } for n in nodes]

    root = {
        "id": "ROOT",
        "name": "ROOT",
        "level": 0,
        "parent": None,
        "children": root_children,
        "chunk_ids": [],
    }

    hier = {
        "n_nodes": 1 + len(nodes),
        "n_chunks": len(chunks),
        "nodes": [root] + nodes,
    }

    # Write files
    (out_dir / "hierarchy.json").write_text(json.dumps(hier, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out_dir / "node_texts.jsonl", "w", encoding="utf-8") as f:
        for r in node_texts:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[OK] built hierarchy: {len(nodes)+1} nodes from {len(chunks)} chunks")
    print(f"[OK] wrote: {out_dir / 'hierarchy.json'}")
    print(f"[OK] wrote: {out_dir / 'node_texts.jsonl'}")

if __name__ == "__main__":
    main()
