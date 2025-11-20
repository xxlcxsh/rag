from pathlib import Path
from typing import List, Dict, Optional
import re
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, util

# ===================== Параметры =====================
RAW_DIR = Path("/kaggle/input/cleany-docs")
VECTOR_STORE_PATH = Path("/kaggle/working/vector_store")
VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

BIG_CHUNK_SIZE = 1600   # большие сегменты
SMALL_CHUNK_SIZE = 200  # маленькие сегменты

# ===================== Функции для разбиения =====================
def recursive_split(text: str, size: int) -> List[str]:
    """Разбивает текст на чанки размером до `size` по предложениям"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
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
def process_all_hierarchical() -> tuple:
    """Создаёт big и small чанки с payloads"""
    file_paths = list(RAW_DIR.rglob("*.txt"))
    
    big_chunks = []
    big_payloads = []
    small_chunks = []
    small_payloads = []

    big_counter = 1
    small_counter = 1

    for path in file_paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        text = re.sub(r'\s+', ' ', text).strip()
        if not text:
            continue

        # --- big chunks ---
        bigs = recursive_split(text, BIG_CHUNK_SIZE)
        for i, big_text in enumerate(bigs):
            big_chunks.append(big_text)
            big_payloads.append({
                "id": big_counter,
                "source": path.name,
                "chunk": i,
                "text": big_text
            })
            # --- small chunks внутри big ---
            smalls = recursive_split(big_text, SMALL_CHUNK_SIZE)
            for j, small_text in enumerate(smalls):
                small_chunks.append(small_text)
                small_payloads.append({
                    "id": small_counter,
                    "parent_id": big_counter,
                    "source": path.name,
                    "chunk": j,
                    "text": small_text
                })
                small_counter += 1

            big_counter += 1

    print(f"Total big chunks: {len(big_chunks)}")
    print(f"Total small chunks: {len(small_chunks)}")
    return big_chunks, big_payloads, small_chunks, small_payloads

# ===================== Векторизация и FAISS =====================
def build_hierarchical_index(big_chunks: List[str], big_payloads: List[Dict],
                             small_chunks: List[str], small_payloads: List[Dict]):

    # --- big embeddings ---
    big_embs = embedding_model.encode(big_chunks, convert_to_numpy=True).astype(np.float16)
    small_embs = embedding_model.encode(small_chunks, convert_to_numpy=True).astype(np.float16)

    # Нормализация для cosine similarity
    big_embs = big_embs / np.linalg.norm(big_embs, axis=1, keepdims=True)
    small_embs = small_embs / np.linalg.norm(small_embs, axis=1, keepdims=True)

    # --- FAISS индексы (FlatIP) ---
    dim = big_embs.shape[1]
    index_big = faiss.IndexFlatIP(dim)
    index_big.add(big_embs.astype(np.float32))  # FAISS требует float32
    faiss.write_index(index_big, str(VECTOR_STORE_PATH / "faiss_big.index"))


    index_small = faiss.IndexFlatIP(dim)
    index_small.add(small_embs.astype(np.float32))
    faiss.write_index(index_small, str(VECTOR_STORE_PATH / "faiss_small.index"))

    # --- Сохраняем NumPy массивы эмбеддингов ---
    np.save(VECTOR_STORE_PATH / "big_embs.npy", big_embs)
    np.save(VECTOR_STORE_PATH / "small_embs.npy", small_embs)

    # --- Сохраняем payloads в JSON ---
    with open(VECTOR_STORE_PATH / "big_payloads.json", "w", encoding="utf-8") as f:
        json.dump(big_payloads, f, ensure_ascii=False)
    with open(VECTOR_STORE_PATH / "small_payloads.json", "w", encoding="utf-8") as f:
        json.dump(small_payloads, f, ensure_ascii=False)

    print("Hierarchical FAISS indices and embeddings saved.")

# ===================== RUN =====================
if __name__ == "__main__":
    big_chunks, big_payloads, small_chunks, small_payloads = process_all_hierarchical()
    build_hierarchical_index(big_chunks, big_payloads, small_chunks, small_payloads)
    print("Pipeline completed. Ready for offline RAG search.")
