import json
import os
import shutil
from io import BytesIO

import docx
import pandas as pd
import pypdf


PDF_NATIVE_MIN_CHARS = int(os.getenv("PDF_NATIVE_MIN_CHARS", "80"))
PDF_OCR_DPI = int(os.getenv("PDF_OCR_DPI", "200"))
PDF_OCR_LANG = os.getenv("PDF_OCR_LANG", "eng+fra")
PDF_OCR_MAX_PAGES = int(os.getenv("PDF_OCR_MAX_PAGES", "0")) or None


def _normalize_text(text: str) -> str:
    lines = [line.strip() for line in (text or "").splitlines()]
    return "\n".join(line for line in lines if line)


def _extract_pdf_with_pypdf(file_path: str) -> str:
    pages = []
    with open(file_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            page_text = _normalize_text(page_text)
            if page_text:
                pages.append(f"[Page {page_number}]\n{page_text}")
    return "\n\n".join(pages)


def _extract_pdf_with_pymupdf(file_path: str) -> str:
    try:
        import fitz
    except ImportError:
        return ""

    pages = []
    with fitz.open(file_path) as doc:
        for page_number, page in enumerate(doc, start=1):
            page_text = _normalize_text(page.get_text("text"))
            if page_text:
                pages.append(f"[Page {page_number}]\n{page_text}")
    return "\n\n".join(pages)


def _extract_pdf_with_ocr(file_path: str) -> str:
    if not shutil.which("tesseract"):
        raise RuntimeError("Tesseract binary is not installed or not available in PATH.")

    try:
        import fitz
        import pytesseract
        from PIL import Image
    except ImportError as e:
        raise RuntimeError(
            "OCR dependencies are missing. Install pytesseract and pillow."
        ) from e

    pages = []
    zoom = PDF_OCR_DPI / 72
    matrix = fitz.Matrix(zoom, zoom)

    with fitz.open(file_path) as doc:
        page_limit = min(len(doc), PDF_OCR_MAX_PAGES) if PDF_OCR_MAX_PAGES else len(doc)
        for page_index in range(page_limit):
            page = doc[page_index]
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image = Image.open(BytesIO(pixmap.tobytes("png")))
            page_text = pytesseract.image_to_string(image, lang=PDF_OCR_LANG)
            page_text = _normalize_text(page_text)
            if page_text:
                pages.append(f"[Page {page_index + 1} OCR]\n{page_text}")

    return "\n\n".join(pages)


def extract_pdf_text(file_path: str) -> str:
    """
    Extract text from PDFs using layered fallbacks:
    1. pypdf for standard digital PDFs
    2. PyMuPDF for PDFs where pypdf misses layout/text
    3. OCR via Tesseract for scanned/image-only PDFs
    """
    attempts = []

    for extractor in (_extract_pdf_with_pypdf, _extract_pdf_with_pymupdf):
        try:
            text = _normalize_text(extractor(file_path))
            if text:
                attempts.append(text)
            if len(text) >= PDF_NATIVE_MIN_CHARS:
                return text
        except Exception as e:
            print(f"PDF native extraction warning ({extractor.__name__}): {e}")

    best_native = max(attempts, key=len, default="")

    try:
        ocr_text = _normalize_text(_extract_pdf_with_ocr(file_path))
        if ocr_text and not best_native:
            return ocr_text
        if ocr_text and len(ocr_text) >= max(PDF_NATIVE_MIN_CHARS, len(best_native) * 2):
            return ocr_text
    except Exception as e:
        if not best_native:
            print(f"PDF OCR fallback unavailable: {e}")

    return best_native


def extract_text(file_path: str, filename: str) -> str:
    ext = filename.lower().split(".")[-1]
    text = ""

    try:
        if ext == "txt":
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        elif ext == "pdf":
            text = extract_pdf_text(file_path)
        elif ext == "docx":
            doc = docx.Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
        elif ext in ["csv", "xlsx"]:
            df = pd.read_csv(file_path) if ext == "csv" else pd.read_excel(file_path)
            text = f"Dataset Columns: {list(df.columns)}\nSample Data:\n{df.head(5).to_csv(index=False)}"
        elif ext == "json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                text = json.dumps(data, indent=2)
    except Exception as e:
        print(f"Error extracting text from {filename}: {e}")
        text = ""

    return _normalize_text(text)
