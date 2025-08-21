# src/ingest/pdf_to_markdown.py
from __future__ import annotations
import argparse, json, io, os
from pathlib import Path
from typing import Dict, Any, Optional

# --- Prefer Docling (CPU, tables, OCR optional) ---
_DOC_AVAILABLE = False
try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    _DOC_AVAILABLE = True
except Exception:
    _DOC_AVAILABLE = False

# --- PyMuPDF + optional Tesseract OCR fallback ---
import fitz  # PyMuPDF
try:
    import pytesseract
    from PIL import Image
    _TESS = True
except Exception:
    _TESS = False

def _extract_toc_and_pages(pdf_path: Path, pages_jsonl: Path, toc_json: Path,
                           do_ocr: bool = False, dpi: int = 300, ocr_lang: str = "eng",
                           min_chars_no_ocr: int = 60) -> None:
    doc = fitz.open(str(pdf_path))
    # TOC (bookmarks)
    toc = doc.get_toc(simple=True)  # [level, title, page]
    toc_json.write_text(json.dumps(
        [{"level": int(l), "title": t.strip(), "page": int(p)} for (l, t, p) in toc],
        ensure_ascii=False, indent=2
    ), encoding="utf-8")
    # Pages text (+ optional OCR)
    pages_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with pages_jsonl.open("w", encoding="utf-8") as f:
        for i in range(len(doc)):
            page = doc.load_page(i)
            text = page.get_text("text") or ""
            rec: Dict[str, Any] = {"page": i + 1, "text": text}
            if do_ocr and len(text.strip()) < min_chars_no_ocr:
                if not _TESS:
                    rec["note"] = "OCR requested, but pytesseract/PIL not available"
                else:
                    pix = page.get_pixmap(dpi=dpi)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    try:
                        ocr_text = pytesseract.image_to_string(img, lang=ocr_lang)
                        if ocr_text.strip():
                            rec["ocr_used"] = True
                            rec["text"] = (text + "\n" + ocr_text).strip() if text else ocr_text
                    except Exception as e:
                        rec["ocr_error"] = str(e)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def _docling_markdown(pdf_path: Path) -> str:
    # Configure Docling to do table structure + accurate mode (still CPU-friendly)
    pipeline = PdfPipelineOptions(do_table_structure=True)
    pipeline.table_structure_options.mode = TableFormerMode.ACCURATE
    # Limit threads if desired: os.environ.setdefault("OMP_NUM_THREADS", "4")

    conv = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline)}
    )
    res = conv.convert(str(pdf_path))
    # You can also inspect res.document for per-block provenance if needed later
    return res.document.export_to_markdown()

def _pymupdf_markdown(pdf_path: Path) -> str:
    # Very simple “preserve paragraphs” extractor as a last resort
    doc = fitz.open(str(pdf_path))
    parts = []
    for i in range(len(doc)):
        page = doc.load_page(i)
        txt = page.get_text("markdown") or page.get_text("text") or ""
        parts.append(txt.strip())
    return "\n\n---\n\n".join(parts)

def convert_pdf_to_markdown(pdf_path: Path, out_dir: Path,
                            ocr: bool = False, dpi: int = 300, ocr_lang: str = "eng") -> Dict[str, Optional[Path]]:
    pdf_path = Path(pdf_path); out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Convert to Markdown (Docling → PyMuPDF fallback)
    if _DOC_AVAILABLE:
        md_text = _docling_markdown(pdf_path)
    else:
        md_text = _pymupdf_markdown(pdf_path)

    md_path = out_dir / f"{pdf_path.stem}.md"
    md_path.write_text(md_text, encoding="utf-8")

    # 2) Extract TOC & per-page text (with optional OCR on low-text pages)
    toc_json = out_dir / f"{pdf_path.stem}.toc.json"
    pages_jsonl = out_dir / f"{pdf_path.stem}.pages.jsonl"
    _extract_toc_and_pages(pdf_path, pages_jsonl, toc_json, do_ocr=ocr, dpi=dpi, ocr_lang=ocr_lang)

    return {
        "markdown": md_path,
        "toc_json": toc_json,
        "pages_jsonl": pages_jsonl,
        "images_dir": None,  # Docling can export figures separately if we need that later
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out_dir", default="data/md")
    ap.add_argument("--ocr", action="store_true")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--ocr_lang", default="eng")
    a = ap.parse_args()
    paths = convert_pdf_to_markdown(Path(a.pdf), Path(a.out_dir), ocr=a.ocr, dpi=a.dpi, ocr_lang=a.ocr_lang)
    for k, v in paths.items(): print(f"{k}: {v}")

if __name__ == "__main__":
    main()
