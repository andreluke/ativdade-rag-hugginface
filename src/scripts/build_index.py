"""
Script que:
- Lê os .txt de data/docs/
- Calcula embeddings com sentence-transformers
- Cria index FAISS e salva (index.faiss + metadata.json)
"""
import os, json
from pathlib import Path
import numpy as np
import faiss
from src.utils.embedding_client import EmbeddingClient

DATA_DIR = Path("data/docs")
OUT_DIR = Path("data/index")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_docs(data_dir):
    docs = []
    filenames = []
    for p in sorted(data_dir.glob("*.txt")):
        text = p.read_text(encoding="utf-8").strip()
        if text:
            docs.append(text)
            filenames.append(str(p.name))
    return docs, filenames

def main():
    docs, filenames = load_docs(DATA_DIR)
    if not docs:
        print("Nenhum documento encontrado em data/docs/*.txt")
        return

    emb_client = EmbeddingClient()
    embeddings = emb_client.embed_documents(docs)  # shape (n_docs, dim)
    dim = embeddings.shape[1]

    # cria index faiss (L2 - usar inner product com normalização se quiser cosine)
    index = faiss.IndexFlatIP(dim)  # we'll normalize to use cosine similarity
    # normalizar embeddings para usar IP como cosine
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    emb_norm = embeddings / norms
    index.add(emb_norm.astype('float32')) # type: ignore

    faiss.write_index(index, str(OUT_DIR / "index.faiss"))

    # salvar metadados (docs, filenames)
    meta = {
        "filenames": filenames,
        "docs": docs,
        "dim": dim
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Index salvo em {OUT_DIR}. entradas: {len(docs)} dim={dim}")

if __name__ == "__main__":
    main()
