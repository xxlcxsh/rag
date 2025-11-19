from pathlib import Path
from typing import List, Dict, Optional
import pdfplumber
from docx import Document
from striprtf.striprtf import rtf_to_text
import numpy as np
import faiss
import fitz
import torch
import easyocr
from sentence_transformers import SentenceTransformer, util
import re

RAW_DIR = Path(r"/kaggle/input/documents")
EXTRACTED_DIR = Path(r"/kaggle/working/test2")
VECTOR_STORE_PATH = Path(r"/kaggle/working/test/vector_store")
RAW_DIR.mkdir(parents=True, exist_ok=True)
EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

def extract_pdf(path: Path) -> Optional[str]:
    try:
        with pdfplumber.open(path) as pdf:
            pieces = [page.extract_text() for page in pdf.pages if page.extract_text()]
        return "\n\n".join(pieces) or None
    except:
        return None
    
def extract_pdf_ocr(path: Path) -> Optional[str]:
    try:
        reader = easyocr.Reader(['ru'], gpu=torch.cuda.is_available())
        doc = fitz.open(path)
        text_parts = []
        
        for page in doc:
            pix = page.get_pixmap(dpi=150)
            img_data = pix.tobytes("png")
            
            results = reader.readtext(img_data, paragraph=True)
            page_text = "\n".join([result[1] for result in results])
            if page_text.strip():
                text_parts.append(page_text)
        doc.close()
        return "\n\n".join(text_parts).strip() or None
    except Exception as e:
        print(f"EasyOCR PDF error ({path}): {e}")
        return None

def extract_docx(path: Path) -> Optional[str]:
    try:
        doc = Document(str(path))
        parts = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(parts) or None
    except:
        return None

def extract_rtf(path: Path) -> Optional[str]:
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
        text = rtf_to_text(content)
        return text.strip() or None
    except:
        return None

def extract_text(path: Path) -> Optional[str]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        t = extract_pdf(path)
        if not t:
            t = extract_pdf_ocr(path)
        return t
    if ext == ".docx" or ext == ".doc":
        return extract_docx(path)
    if ext == ".rtf":
        return extract_rtf(path)
    print(path)
    return None

def process_text():
    file_paths = list(RAW_DIR.rglob("*.doc"))
    for path in file_paths:
        print(f"Обработка документа {path}")
        text = extract_text(path)
        print(f"Извлечение документа {path} завершено")
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        allowed_re = r'[^0-9A-Za-zА-Яа-яёЁ\n\.\,\:\;\—\-\–\(\)\[\]\{\}°%±≤≥µΩ×\+\*/=≤≥≤≈™®©\|№§\s]'
        cleaned = re.sub(allowed_re, '', text)
        cleaned = re.sub(r'[ \t]+\n', '\n', cleaned)
        text = re.sub(r'\n[ \t]+', '\n', cleaned)
        if not text:
            continue

        out_file = EXTRACTED_DIR / f"{path.stem}.txt"
        out_file.write_text(text, encoding="utf-8", errors="ignore")

    return None
