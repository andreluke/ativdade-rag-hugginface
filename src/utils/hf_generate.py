"""
Gerador que pode usar a Hugging Face Inference API para geração.
Também é possível adaptar para usar transformers localmente se desejar.
"""
import os
import requests
from typing import Optional

HF_API_TOKEN = os.getenv("HF_API_TOKEN", None)
DEFAULT_MODEL = os.getenv("HF_GEN_MODEL", "google/flan-t5-base")

HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"} if HF_API_TOKEN else {}

def hf_generate(prompt: str, model: Optional[str]=None, max_length: int = 256) -> str:
    model = model or DEFAULT_MODEL
    if not HF_API_TOKEN:
        raise RuntimeError("HF_API_TOKEN não encontrado no ambiente. Configure sua chave no .env")

    url = f"https://api-inference.huggingface.co/models/{model}"
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_length, "return_full_text": False},
    }
    resp = requests.post(url, headers=HEADERS, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    # a resposta pode ser lista ou dict, tratar os dois casos
    if isinstance(data, list) and len(data)>0 and "generated_text" in data[0]:
        return data[0]["generated_text"]
    if isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"]
    # fallback: stringify
    return str(data)
