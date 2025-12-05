"""Funções para chunking de texto."""

def simple_chunk_text(text: str, max_words: int = 150):
    text = text.replace('\r\n', '\n')
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    chunks = []
    for p in paragraphs:
        words = p.split()
        if len(words) <= max_words:
            chunks.append(p)
        else:
            for i in range(0, len(words), max_words):
                slice_words = words[i:i+max_words]
                chunks.append(' '.join(slice_words))
    return chunks