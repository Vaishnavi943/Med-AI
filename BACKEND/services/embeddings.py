import numpy as np
from sentence_transformers import SentenceTransformer

# Load the model once when the module is imported
# This is a small model, perfectly suited for running locally without a GPU
model = SentenceTransformer("all-MiniLM-L6-v2")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Splits text into chunks of `chunk_size` characters with an overlap.
    A simple character-based chunking strategy for PDF text.
    """
    if not text:
        return []
        
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)

    return chunks


def get_embeddings(chunks: list[str]) -> np.ndarray:
    """
    Generates embeddings for a list of text chunks.
    Returns a numpy array of shape (len(chunks), embedding_dim).
    """
    if not chunks:
        return np.array([])
        
    embeddings = model.encode(chunks)
    return np.array(embeddings)
