"""
Funções para gerar embeddings (local com sentence-transformers) e abstração.
"""
import os
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingClient:
    def __init__(self, model_name=None):
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        # carrega modelo localmente (rápido e leve)
        print(f"[EmbeddingClient] carregando modelo: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)

    def embed_documents(self, docs: list[str]) -> np.ndarray:
        """Retorna array (n_docs, dim) de embeddings."""
        embeddings = self.model.encode(docs, show_progress_bar=True, convert_to_numpy=True)
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode([query], convert_to_numpy=True)[0]
