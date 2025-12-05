# LLM + RAG — Recuperação-Aumentada (RAG) com Hugging Face

Este repositório demonstra uma integração prática entre uma LLM (modelo de linguagem) e um pipeline RAG (Retrieval-Augmented Generation). O objetivo é enriquecer a geração com contexto recuperado de uma base de conhecimento local usando embeddings e busca vetorial (FAISS).

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/transformers)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-green.svg)](https://github.com/facebookresearch/faiss)

## 🎯 Visão Geral

- **RAG**: indexação de documentos → recuperação por similaridade → geração condicionada ao contexto recuperado.
- **Uso típico**: responder perguntas sobre uma base de conhecimento específica, sumarização com contexto, e aplicações de QA em domínio fechado.
- **Arquitetura**: embeddings (sentence-transformers) + índice FAISS + LLM (Transformers) para geração.

## ✅ Funcionalidades Principais

- **Indexação de documentos**: chunking de textos e cálculo de embeddings.
- **Busca semântica eficiente**: FAISS para recuperação rápida de contexto.
- **Geração condicionada**: LLM que recebe o contexto recuperado para respostas mais precisas.
- **Cache de embeddings e índice**: evita recomputação em execuções subsequentes.
- **Facilidade de extensão**: trocar modelos, bases de conhecimento e parâmetros de busca.

## 🗂️ Estrutura do Projeto

`src/`
- `main.py` — ponto de entrada para construir índice e executar exemplos
- `rag/chunking.py` — funções para segmentar documentos
- `rag/retriever.py` — construir/consultar índice FAISS
- `llm/model.py` — wrapper para carregar e inferir com a LLM
- `utils/preprocessing.py` — pré-processamento de texto

`data/`
- `dsm_material.txt` — base de conhecimento de exemplo (substitua pelo seu corpus)

`requirements.txt` — dependências do projeto

## 🚀 Instalação e Execução Rápida

### Pré-requisitos

- Python 3.8+
- Conexão com internet (para baixar modelos na primeira execução)
- 4GB+ de RAM recomendado (varia conforme o modelo)

### Passos

1. Clone o repositório e entre na pasta:

```powershell
git clone https://github.com/andreluke/ativdade-rag-hugginface atividade1
cd atividade1
```

2. Crie e ative um ambiente virtual (PowerShell):

```powershell
python -m venv .venv
.\venv\Scripts\Activate.ps1
```

3. Instale dependências:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

4. Execute o exemplo (constrói índice e demonstra RAG):

```powershell
python src/main.py
```

Na primeira execução, o índice será construído e os modelos serão baixados. Em execuções subsequentes o cache é reutilizado.

## ⚙️ Como Funciona (Fluxo RAG)

1. Documentos em `data/` são divididos em chunks.
2. Cada chunk recebe um embedding via `sentence-transformers`.
3. Embeddings são indexados com FAISS.
4. Para uma consulta, calculamos o embedding da pergunta e recuperamos os N contextos mais relevantes.
5. A LLM gera a resposta condicionada pelo contexto recuperado.

## 🧩 Trocar Modelos

Edite `src/main.py` ou `llm/model.py` para alterar os modelos usados:

```python
# exemplo
model_name = "sentence-transformers/all-MiniLM-L6-v2"  # embeddings
llm_model = "distilgpt2"  # ou outro modelo compatível de geração
```

Recomenda-se usar um modelo de geração que aceite prompts com contexto e que caiba nos recursos disponíveis.

## 📌 Boas práticas

- Mantenha o corpus atualizado e normalize texto antes da indexação.
- Use chunking com sobreposição para preservar contexto.
- Avalie trade-offs entre tamanho do modelo e latência.

## Contribuições

- Fork → branch → PR. Abra issues para sugestões e bugs.

## Licença

Conteúdo de exemplo para fins educacionais.
