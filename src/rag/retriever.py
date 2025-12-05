import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List
from rag.chunking import simple_chunk_text

CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

EMBEDDINGS_PATH = os.path.join(CACHE_DIR, 'embeddings.npy')
META_PATH = os.path.join(CACHE_DIR, 'metadata.json')
INDEX_PATH = os.path.join(CACHE_DIR, 'vector_index.faiss')


class Retriever:
    def __init__(self, embed_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.embedder = SentenceTransformer(embed_model_name)
        self.index = None
        self.metadata = []

    def build_index_if_needed(self, data_path: str):
        if os.path.exists(INDEX_PATH):
            print('Carregando índice existente...')
            self._load_index()
            return

        print('Construindo índice a partir de:', data_path)
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()

        chunks = simple_chunk_text(text)
        embeddings = self.embedder.encode(chunks, show_progress_bar=True, convert_to_numpy=True)

        self.metadata = [{'text': c} for c in chunks]
        np.save(EMBEDDINGS_PATH, embeddings)
        with open(META_PATH, 'w', encoding='utf-8') as mf:
            json.dump(self.metadata, mf, ensure_ascii=False, indent=2)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings) # type: ignore
        faiss.write_index(index, INDEX_PATH)

        self.index = index
        print('Índice construído e salvo.')

    def _load_index(self):
        self.index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, 'r', encoding='utf-8') as mf:
            self.metadata = json.load(mf)

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        D, I = self.index.search(q_emb, top_k) # type: ignore
        return [self.metadata[idx]['text'] for idx in I[0] if idx < len(self.metadata)]