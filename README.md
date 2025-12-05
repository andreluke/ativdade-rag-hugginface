# RAG + Hugging Face — Exemplo em Python

Este projeto demonstra um pipeline simples de RAG:

1. Calcular embeddings dos documentos (sentence-transformers).
2. Indexar com FAISS.
3. Recuperar top-k documentos para uma query.
4. Montar prompt com os documentos e chamar a Hugging Face Inference API para geração da resposta (RAG).

> Conceitos de embeddings, TF-IDF e similaridade usados aqui seguem a Aula 3 (BoW/TF-IDF → Embeddings) do material "PLN - Aula 3.pdf". :contentReference[oaicite:1]{index=1}

## Pré-requisitos

- Python 3.9+
- pip
- Chave Hugging Face (se quiser usar a Inference API): <https://huggingface.co/settings/tokens>

## Instalação

```bash
git clone https://github.com/andreluke/ativdade-rag-hugginface
cd ativdade-rag-hugginface
python -m venv .venv
source .venv/bin/activate   # ou .venv\Scripts\activate no Windows
pip install -r requirements.txt
cp .env.example .env

