import os
import csv
import pandas as pd
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
from faiss_store import FAISSStore, Reranker, get_context
from llm import load_llm, generate_answer 

INPUT_CSV = "input.csv"
OUTPUT_CSV = "output.csv"
LLM_MODEL = "qwen2.5-0.5b-instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VECTOR_STORE_PATH = Path("vector_store")

SYSTEM_PROMPT = (
    "Используя только предоставленный контекст, дай краткий и точный ответ на вопрос пользователя. "
    "Не придумывай информацию, если информация в контексте отсутствует, "
    "выдай типовое сообщение: 'Информация в предоставленных документах отсутствует.'. "
    "Ответ должен быть коротким, насколько это возможно, но при этом информативным"
)

def main():
    if not os.path.exists(INPUT_CSV):
        print("ERROR: input.csv not found")
        return

    df = pd.read_csv(INPUT_CSV)
    store = FAISSStore(VECTOR_STORE_PATH).load_embds()
    emb_model = SentenceTransformer("models/all-mpnet-base-v2", local_files_only=True)

    tokenizer, model = load_llm(LLM_MODEL)

    results = []
    for _, row in df.iterrows():
        qid = row["id"]
        question = str(row["question"])
        print(f"Processing id={qid}...")

        context, retrieved_docs = get_context(question, store, emb_model)
        answer = generate_answer(tokenizer, model, question, context)

        results.append({
            "id": qid,
            "answer": answer,
            "documents": ';'.join([doc['payload']['source'] for doc in retrieved_docs])
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_MINIMAL)

if __name__ == "__main__":
    main()
