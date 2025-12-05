"""
Exemplo CLI de RAG:
- Carrega index FAISS e metadata
- Faz embed da query, busca top_k documentos
- Monta prompt concatenando os trechos recuperados
- Chama Hugging Face Inference API para gerar a resposta final (RAG)
"""
import os, json, argparse
from pathlib import Path
import numpy as np
import faiss
from src.utils.embedding_client import EmbeddingClient
from src.utils.hf_generate import hf_generate

INDEX_DIR = Path("data/index")
META_FILE = INDEX_DIR / "metadata.json"
INDEX_FILE = INDEX_DIR / "index.faiss"

def load_index():
    if not INDEX_FILE.exists() or not META_FILE.exists():
        raise FileNotFoundError("Index não encontrado. Rode scripts/build_index.py primeiro.")
    index = faiss.read_index(str(INDEX_FILE))
    meta = json.loads(META_FILE.read_text(encoding="utf-8"))
    return index, meta

def retrieve(query, top_k=5, emb_client=None):
    emb_client = emb_client or EmbeddingClient()
    q_emb = emb_client.embed_query(query)
    # normalizar    
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-10)
    index, meta = load_index()
    D, I = index.search(np.array([q_emb], dtype='float32'), top_k)
    hits = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        hits.append({
            "score": float(score),
            "doc": meta["docs"][idx],
            "filename": meta["filenames"][idx]
        })
    return hits

def build_prompt(question, hits):
    # Prompt simples: instrução + contextos + pergunta
    context = "\n\n---\n\n".join([f"[Fonte: {h['filename']}]\n{h['doc']}" for h in hits])
    prompt = (
        "Você é um assistente que responde com base nos contextos fornecidos.\n"
        "Utilize apenas as informações dos contextos. Se não houver resposta nos contextos, "
        "responda honestamente que não há informação suficiente.\n\n"
        f"CONTEXTOS:\n{context}\n\nPERGUNTA: {question}\n\nRESPOSTA:"
    )
    return prompt

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--q", "--query", dest="query", required=True)
    p.add_argument("--topk", dest="topk", type=int, default=int(os.getenv("TOP_K", 5)))
    p.add_argument("--hf-model", dest="hf_model", default=os.getenv("HF_GEN_MODEL", None))
    args = p.parse_args()

    emb_client = EmbeddingClient()
    hits = retrieve(args.query, top_k=args.topk, emb_client=emb_client)
    print("=== Recuperados ===")
    for h in hits:
        print(f"{h['filename']} (score={h['score']:.4f})")
    prompt = build_prompt(args.query, hits)
    # chama Hugging Face Inference API
    print("\n=== Prompt (resumido) ===")
    print(prompt[:1000] + ("...\n[truncado]" if len(prompt)>1000 else ""))
    print("\nGerando resposta com Hugging Face...")
    resp = hf_generate(prompt, model=args.hf_model)
    print("\n=== Resposta gerada ===")
    print(resp)

if __name__ == "__main__":
    main()
