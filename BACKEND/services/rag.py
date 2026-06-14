import os
import numpy as np
import google.generativeai as genai
from services.embeddings import chunk_text, get_embeddings, model as embedding_model
from config.settings import SYSTEM_PROMPT_PDF

# Initialize Gemini client
# Ensure GEMINI_API_KEY is in your .env file
try:
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        gemini_client_configured = True
    else:
        gemini_client_configured = False
except Exception as e:
    print(f"Warning: Failed to configure Gemini. Is GEMINI_API_KEY set? {e}")
    gemini_client_configured = False

# In-memory document store
# Structure: { "session_id": { "chunks": ["text1", ...], "embeddings": np.array([...]) } }
DOCUMENT_STORE = {}


def cosine_similarity(query_emb: np.ndarray, doc_embs: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a query embedding and a list of document embeddings.
    """
    if doc_embs.size == 0:
        return np.array([])
        
    dot_product = np.dot(doc_embs, query_emb)
    query_norm = np.linalg.norm(query_emb)
    doc_norms = np.linalg.norm(doc_embs, axis=1)
    
    # Avoid division by zero
    denoms = query_norm * doc_norms
    denoms[denoms == 0] = 1e-10
    
    return dot_product / denoms


def store_document(session_id: str, text: str):
    """
    Chunks the text, calculates embeddings, and stores them in the in-memory store.
    """
    chunks = chunk_text(text)
    if not chunks:
        DOCUMENT_STORE[session_id] = {"chunks": [], "embeddings": np.array([])}
        return

    embeddings = get_embeddings(chunks)
    DOCUMENT_STORE[session_id] = {
        "chunks": chunks,
        "embeddings": embeddings
    }


def query_document(session_id: str, question: str, top_k: int = 3) -> str:
    """
    Searches the stored document for relevant chunks and asks Groq to answer.
    """
    if session_id not in DOCUMENT_STORE:
        return "Error: Document session not found. Please upload the PDF again."

    store_data = DOCUMENT_STORE[session_id]
    chunks = store_data.get("chunks", [])
    doc_embeddings = store_data.get("embeddings", np.array([]))

    if not chunks or doc_embeddings.size == 0:
        return "The uploaded document contains no readable text."

    if not gemini_client_configured:
        return "Error: Gemini API is not configured. Please set GEMINI_API_KEY."

    # Embed the question
    query_emb = embedding_model.encode(question)

    # Calculate similarities
    similarities = cosine_similarity(query_emb, doc_embeddings)

    # Get top K most similar chunks
    # Handle the case where we have fewer chunks than top_k
    k = min(top_k, len(similarities))
    top_indices = np.argsort(similarities)[-k:][::-1]

    # Combine top chunks into context
    context = "\n\n---\n\n".join([chunks[i] for i in top_indices])

    # Construct the prompt for Gemini
    prompt = f"{SYSTEM_PROMPT_PDF}\n\nContext Information:\n{context}\n\nQuestion: {question}"

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(temperature=0.1)
        )
        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return "Sorry, I encountered an error while trying to generate the answer."
