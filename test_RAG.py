import sys
import os

# Adicionar o diretório src ao path
sys.path.append('src')

from rag.retriever import Retriever
from llm.model import HuggingFaceLLM

def test_RAG():
    print("=== Teste do RAG LLM ===")
    
    # Inicializar componentes
    print("Inicializando RAG...")
    retriever = Retriever()
    retriever.build_index_if_needed('data/data.txt')
    
    print("Inicializando modelo...")
    llm = HuggingFaceLLM()
    
    # Perguntas de teste
    test_questions = [
        "Como otimizar performance em React Native?",
        "Qual a diferença entre React Native e Flutter?",
        "Como fazer testes em aplicações mobile?",
    ]
    
    print("\n=== Executando Testes ===\n")
    
    for i, question in enumerate(test_questions, 1):
        print(f"🔍 Pergunta {i}: {question}")
        print("-" * 60)
        
        try:
            # Buscar contexto relevante
            context = retriever.retrieve(question, top_k=3)
            
            # Preparar prompt completo
            prompt = f"Informações relevantes:\n"
            for doc in context:
                prompt += f"• {doc}\n"
            prompt += f"\nUsuário: {question}"
            
            # Gerar resposta
            response = llm.generate(prompt)
            
            print(f"🤖 Resposta: {response}")
            
        except Exception as e:
            print(f"❌ Erro: {e}")
        
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    test_RAG()