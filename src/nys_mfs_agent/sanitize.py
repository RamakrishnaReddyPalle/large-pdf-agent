import re

_JSON_EDGE = re.compile(r'^\s*[{[]|[}\]]\s*$')
_CODE_FENCE = re.compile(r'^\s*```')
_QUOTED_LINE = re.compile(r'^\s*["“]')

def clean_text(s: str) -> str:
    lines = (s or "").splitlines()
    out = []
    for ln in lines:
        if _CODE_FENCE.match(ln): 
            continue
        if _JSON_EDGE.match(ln):
            continue
        ln = ln.strip()
        if _QUOTED_LINE.match(ln) and ln.endswith(('"',"”")):
            ln = ln.strip(' "”')
        out.append(ln)
    txt = "\n".join(out).strip()
    txt = txt.replace(',.', '.').replace('.,', '.')
    txt = re.sub(r'\s{2,}', ' ', txt)
    return txt
