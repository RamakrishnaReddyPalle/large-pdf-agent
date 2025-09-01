import re

def rewrite_query(q: str) -> str:
    s = (q or "").strip()
    s = re.sub(r'\bE\s*&\s*M\b', 'Evaluation and Management', s, flags=re.I)
    s = re.sub(r'\bE\/M\b', 'Evaluation and Management', s, flags=re.I)
    s = re.sub(r'\bRVU\b', 'relative value unit', s, flags=re.I)
    s = re.sub(r'\bcap\b', 'limit', s, flags=re.I)
    return s
