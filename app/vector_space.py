from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from config.settings import EMBEDDING_MODEL 

model = SentenceTransformer(EMBEDDING_MODEL)

def embed_text(texts):
    return model.encode(texts, show_progress_bar=True)

def build_fass_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  # Using L2 distance
    index.add(embeddings)
    return index

def search_similar(query_embeddings, index, top_k = 3):
    distances, indices = index.search(np.array([query_embeddings]), top_k)
    return distances[0], indices[0]