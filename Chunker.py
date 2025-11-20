from pathlib import Path
import pickle
from typing import List, Dict, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, util
import zipfile
import nltk
from nltk.tokenize import sent_tokenize

RAW_DIR = Path("/kaggle/input/cleany-docs")
VECTOR_STORE_PATH = Path("/kaggle/working/vector_store")
RAW_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cuda" if embedding_model.device.type=="cuda" else "cpu")

BIG_CHUNK_SIZE = 3000
SMALL_CHUNK_SIZE = 600

nltk.download("punkt")

def split_into_sentences(text: str) -> list[str]:
    return sent_tokenize(text)


def recursive_split(text: str, size: int) -> List[str]:
    sentences = split_into_sentences(text)
    chunks = []
    buf = ""
    for s in sentences:
        if len(buf) + len(s) + 1 <= size:
            buf = (buf + " " + s).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = s
    if buf:
        chunks.append(buf)
    return chunks
    
# ===================== Основной пайплайн =====================

def process_all_hierarchical():
    file_paths = list(RAW_DIR.rglob("*.txt"))

    big_chunks = []
    small_chunks = []
    big_payloads = []
    small_payloads = []

    big_id_counter = 1
    small_id_counter = 1

    for path in file_paths:
        text = path.read_text(encoding="utf-8", errors="ignore")

        # 1. Делаем большие сегменты
        bigs = recursive_split(text, BIG_CHUNK_SIZE)
        for i, big_text in enumerate(bigs):
            big_chunks.append(big_text)
            big_payloads.append({
                "id": big_id_counter,
                "text": big_text,
                "source": path.name
            })

            # 2. Делаем маленькие сегменты внутри большого
            smalls = recursive_split(big_text, SMALL_CHUNK_SIZE)
            for j, small_text in enumerate(smalls):
                small_chunks.append(small_text)
                small_payloads.append({
                    "id": small_id_counter,
                    "parent_id": big_id_counter,
                    "text": small_text,
                    "source": path.name
                })
                small_id_counter += 1

            big_id_counter += 1

    return big_chunks, big_payloads, small_chunks, small_payloads

# ===================== Векторизация и FAISS =====================

def build_hierarchical_index(big_chunks, big_payloads, small_chunks, small_payloads):
    # 1. Эмбеддинги больших сегментов
    big_embs = embedding_model.encode(big_chunks, convert_to_numpy=True)
    big_embs = big_embs / np.linalg.norm(big_embs, axis=1, keepdims=True)
    big_embs = big_embs.astype(np.float16)
    dim = big_embs.shape[1]
    index_big = faiss.IndexFlatIP(dim)
    index_big.add(big_embs)

    # 2. Эмбеддинги маленьких сегментов
    small_embs = embedding_model.encode(small_chunks, convert_to_numpy=True)
    small_embs = small_embs / np.linalg.norm(small_embs, axis=1, keepdims=True)
    small_embs = small_embs.astype(np.float16)

    index_small = faiss.IndexFlatIP(dim)
    index_small.add(small_embs)

    # Сохраняем все индексы и payloads
    faiss.write_index(index_big, str(VECTOR_STORE_PATH / "faiss_big.index"))
    faiss.write_index(index_small, str(VECTOR_STORE_PATH / "faiss_small.index"))

    with open(VECTOR_STORE_PATH / "big_payloads.pkl", "wb") as f:
        pickle.dump(big_payloads, f)
    with open(VECTOR_STORE_PATH / "small_payloads.pkl", "wb") as f:
        pickle.dump(small_payloads, f)

    np.save(VECTOR_STORE_PATH / "big_embs.npy", big_embs)
    np.save(VECTOR_STORE_PATH / "small_embs.npy", small_embs)

    print("Hierarchical FAISS indices saved.")

def make_zip_archive(zip_path: Path, folder: Path):
    with zipfile.ZipFile(zip_path, "w") as zf:
        for file in folder.rglob("*"):
            zf.write(file, arcname=file.relative_to(folder))
    print(f"ZIP archive created at {zip_path}")
