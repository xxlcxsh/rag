import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer,CrossEncoder
from pathlib import Path

VECTOR_STORE_PATH = Path("data/vector_store")
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

class FAISSStore:
    def __init__(self, vector_store_path: Path = VECTOR_STORE_PATH):
        self.vector_store_path = vector_store_path
        self.index = None
        self.embeddings = None
        self.payloads = None
        self.ids = None

    def load_embds(self):
        self.embeddings = np.load(self.vector_store_path / "embeddings.npy")
        self.index = faiss.read_index(str(self.vector_store_path / "faiss.index"))
        with open(self.vector_store_path / "payloads.pkl", "rb") as f:
            self.payloads = pickle.load(f)
        # Генерируем ids как последовательность индексов
        self.ids = list(range(len(self.payloads)))
        return self

    def search(self, query_emb: np.ndarray, top_k: int = 3):
        scores, indices = self.index.search(query_emb[np.newaxis, :], top_k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            results.append({
                "score": float(score),
                "payload": self.payloads[idx]
            })
        return results
class Reranker:
    def __init__(self, model_path: str):
        self.model = CrossEncoder(model_path, local_files_only=False)

    def rerank(self, query: str, candidates: list, top_n: int = 3):
        pairs = [(query, c['payload']['text']) for c in candidates]
        scores = self.model.predict(pairs)
        ranked_results = [c for _, c in sorted(zip(scores, candidates), reverse=True)]
        return ranked_results[:top_n]


def get_context(query: str, store, emb_model, reranker: Reranker, top_faiss: int = 20, top_final: int = 3):
    _, faiss_results = store.search(emb_model.encode([query])[0], top_k=top_faiss)
    final_results = reranker.rerank(query, faiss_results, top_n=top_final)
    context = "\n\n".join(doc['payload']['text'] for doc in final_results)
    return context, final_results
