from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from config.settings import EMBEDDING_MODEL 

model = SentenceTransformer(EMBEDDING_MODEL)

def embed_text(texts):
    return model.encode(texts, show_progress_bar=True)

def build_faiss_index(embeddings):
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])  # inner product = cosine
    index.add(embeddings)
    return index

def search_similar(query_embedding, index, top_k=3):
    query = np.array([query_embedding])
    faiss.normalize_L2(query)
    distances, indices = index.search(query, top_k)
    return indices[0], distances[0]