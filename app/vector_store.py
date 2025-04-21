from sentence_transformers import SentenceTransformer
from .keyword_weighting import extract_keywords_only
import faiss
import numpy as np
from config.settings import EMBEDDING_MODEL 

model = SentenceTransformer(EMBEDDING_MODEL)

def embed_text(texts):
    return model.encode(texts, show_progress_bar=True)

def build_faiss_index(embeddings):
    # normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])  # inner product = cosine
    index.add(embeddings)
    return index

def search_similar(query_embedding, index, job_descriptions, resume_text, top_k=3, alpha=0.7):
    """
    Optimized search using resume keywords to filter job descriptions.
    Step 1: Extract keywords from resume.
    Step 2: Filter jobs that contain at least one keyword.
    Step 3: Apply combined keyword weight and semantic similarity.
    """
    # extract resume keywords
    resume_keywords = extract_keywords_only(resume_text)

    # filter job descriptions
    filtered_indices = []
    for idx, desc in enumerate(job_descriptions):
        desc_lower = desc.lower()
        if any(kw in desc_lower for kw in resume_keywords):
            filtered_indices.append(idx)

    if not filtered_indices:
        return [], []

    # compute combined scores
    query = np.array([query_embedding])
    faiss.normalize_L2(query)
    
    semantic_distances, semantic_indices = index.search(query, len(job_descriptions))
    semantic_score_dict = {idx: score for idx, score in zip(semantic_indices[0], semantic_distances[0])}

    scores = []
    for idx in filtered_indices:
        desc = job_descriptions[idx].lower()
        match_count = sum(1 for kw in resume_keywords if kw in desc)
        keyword_score = match_count / len(resume_keywords)

        sem_score = semantic_score_dict.get(idx, 0.0)
        combined_score = alpha * sem_score + (1 - alpha) * keyword_score
        scores.append((idx, combined_score))

    # sort by combined score
    sorted_results = sorted(scores, key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in sorted_results[:top_k]]
    top_scores = [score for _, score in sorted_results[:top_k]]
    return top_indices, top_scores