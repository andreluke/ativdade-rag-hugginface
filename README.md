# RAG + Hugging Face — Exemplo em Python

Este projeto demonstra um pipeline simples de RAG:

1. Calcular embeddings dos documentos (sentence-transformers).
2. Indexar com FAISS.
3. Recuperar top-k documentos para uma query.
4. Montar prompt com os documentos e chamar a Hugging Face Inference API para geração da resposta (RAG).

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
```

## Modo de uso

Comece rodando a build para atualizar os docs

```bash
python -m src.scripts.build_index
```

E envie sua pergunta

```bash
python -m src.scripts.query_rag --q "O que diz o documento sobre gatos?"
```
