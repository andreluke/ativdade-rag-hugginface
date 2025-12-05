"""Ponto de entrada simples para o chatbot RAG."""
import os
from rag.retriever import Retriever
from llm.model import HuggingFaceLLM

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'dsm_material.txt')


def main():
    print("DSM Chatbot - RAG + Hugging Face")
    retriever = Retriever()
    retriever.build_index_if_needed(DATA_PATH)

    llm = HuggingFaceLLM(model_name="microsoft/DialoGPT-small")

    history = []
    while True:
        user = input("Você: ")
        if user.strip().lower() in ("sair", "exit", "quit"):
            print("Encerrando...")
            break

        contexts = retriever.retrieve(user, top_k=3)
        
        # Construir prompt simples
        prompt = f"Conversa sobre desenvolvimento mobile:\n\n"
        for h in history[-3:]:  # Últimas 3 interações
            prompt += f"Usuário: {h['user']}\nBot: {h['bot']}\n\n"
        
        if contexts:
            prompt += "Informações relevantes:\n"
            for ctx in contexts:
                prompt += f"• {ctx}\n"
            prompt += "\n"
        
        prompt += f"Usuário: {user}\nBot:"
        
        response = llm.generate(prompt)

        print("Bot:", response)
        history.append({"user": user, "bot": response})


if __name__ == "__main__":
    main()