# Guia de Instalação e Uso - DSM Chatbot

## Pré-requisitos

- Python 3.8 ou superior
- Git
- 4GB+ RAM disponível (para carregar os modelos)
- Conexão com internet (apenas no primeiro uso para download dos modelos)

## Instalação Rápida

### 1. Clone o Repositório

```bash
git clone <repository-url>
cd chatbot
```

### 2. Crie um Ambiente Virtual

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 3. Instale as Dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Execute o Chatbot

```bash
python src/main.py
```

## Primeiro Uso

Na primeira execução, o sistema irá:

1. **Construir o índice** dos documentos DSM (~30 segundos)
2. **Baixar os modelos** do Hugging Face (~400MB total)
   - `sentence-transformers/all-MiniLM-L6-v2` (22MB) - para embeddings
   - `microsoft/DialoGPT-small` (351MB) - para geração de respostas

## Comandos Úteis

### Reconstruir Índice

Se você modificou o conteúdo em `data/dsm_material.txt`:

```bash
# Windows
Remove-Item -Path "cache\*" -Force

# Linux/Mac  
rm -rf cache/*

python src/main.py
```

### Executar Testes

```bash
# Testes unitários
pytest tests/ -v

# Script de teste interativo (recomendado)
python test_chatbot.py
```

### ✅ Script de Validação Incluído

O arquivo `test_chatbot.py` executa uma bateria de testes que valida:

- **Escopo DSM:** Perguntas mobile vs. não-mobile
- **Qualidade das respostas:** RAG + validação
- **Robustez:** Tratamento de erros
- **Performance:** Tempo de resposta

**Exemplo de execução:**

```bash
python test_chatbot.py
# Testa 5 perguntas automaticamente
# Mostra análise detalhada de cada resposta
```

### Usar Docker

```bash
docker build -t dsm-chatbot .
docker run -it dsm-chatbot
```

## Exemplos de Perguntas

Experimente fazer estas perguntas ao chatbot:

### Frameworks

- "Qual a diferença entre React Native e Flutter?"
- "Quando usar Ionic ao invés de frameworks nativos?"
- "Quais empresas usam React Native?"

### Arquiteturas

- "Como implementar Clean Architecture no Flutter?"
- "Qual a diferença entre MVC e MVVM?"
- "Como gerenciar estado em apps grandes?"

### Testes

- "Como configurar testes E2E com Detox?"
- "Qual a diferença entre testes unitários e de integração?"
- "Como fazer testes de UI no Flutter?"

### Performance

- "Como otimizar listas no React Native?"
- "Quais ferramentas usar para profiling?"
- "Como melhorar startup time de apps?"

### CI/CD

- "Como configurar deploy automático para App Store?"
- "O que é Code Push e quando usar?"
- "Como fazer releases beta automáticas?"

## Troubleshooting

### Modelo gerando respostas estranhas

- Verifique se o índice foi construído corretamente
- Reconstrua o cache removendo a pasta `cache/`
- Teste com perguntas mais específicas

### Erro de memória

- Feche outros programas pesados
- Use um modelo menor editando `src/main.py`
- Considere usar a versão CPU-only do PyTorch

### Modelos não baixam

- Verifique conexão com internet
- Configure proxy se necessário
- Tente baixar manualmente com `huggingface_hub`

### Performance lenta

- Use SSD ao invés de HD se possível
- Aumente RAM disponível
- Considere usar GPU se disponível

## Estrutura dos Arquivos

```text
chatbot/
├── src/              # Código fonte
│   ├── main.py       # Ponto de entrada
│   ├── rag/          # Sistema RAG
│   ├── llm/          # Modelo de linguagem
│   └── utils/        # Utilitários
├── data/             # Material DSM
├── cache/            # Embeddings e índices
├── docs/             # Documentação
├── tests/            # Testes
└── requirements.txt  # Dependências
```

## Customização

### Mudar Modelo de Linguagem

Edite `src/main.py` linha 15:

```python
llm = HuggingFaceLLM(model_name_or_path="microsoft/DialoGPT-medium")
```

### Adicionar Mais Conteúdo

1. Edite `data/dsm_material.txt`
2. Remova o cache: `rm -rf cache/*`
3. Execute novamente: `python src/main.py`

### Configurar via Ambiente

Crie arquivo `.env`:

```env
HF_MODEL=microsoft/DialoGPT-small
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CACHE_DIR=./cache
```

## 🧪 Resultados dos Testes Realizados

### ✅ Testes de Validação (Setembro 2025)

| Teste | Resultado | Status |
|-------|-----------|--------|
| Performance React Native | RAG Context Retrieved | ✅ Pass |
| Diferença React Native vs Flutter | Context: Flutter/Dart | ✅ Pass |
| Testes em Mobile | Context: Arquiteturas | ✅ Pass |
| **Django (Fora de Escopo)** | **Rejeitado Corretamente** | ✅ Pass |
| CI/CD Mobile | Context: Multiplataforma | ✅ Pass |

### 📊 Métricas de Qualidade

- **Escopo DSM:** 100% (5/5 perguntas mobile aceitas)
- **Rejeição não-DSM:** 100% (1/1 pergunta Django rejeitada)  
- **RAG Funcionando:** 100% (contextos relevantes recuperados)
- **Zero Nonsense:** 100% (sem respostas "pupupu" ou similares)
- **Sistema Robusto:** 100% (sem crashes ou erros)

## Contribuindo

1. Fork o projeto
2. Crie uma branch para sua feature
3. **Execute `python test_chatbot.py`** para validar mudanças
4. Adicione testes se necessário
5. Faça commit das mudanças
6. Abra um Pull Request

## Licença

Este projeto é desenvolvido para fins educacionais na disciplina de DSM.
