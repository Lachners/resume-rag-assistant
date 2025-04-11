
import os
import pickle
import faiss
from app.resume_parser import extract_text_from_pdf
from app.vector_space import embed_texts, search_similar


def test_similarity(resume_path: str, top_k: int = 5):
    if not os.path.exists(resume_path):
        print(f"❌ File not found: {resume_path}")
        return

    print("📄 Extracting resume text...")
    resume_text = extract_text_from_pdf(resume_path)

    print("📐 Embedding resume...")
    resume_vec = embed_texts([resume_text])[0]

    print("📦 Loading FAISS index and job metadata...")
    index = faiss.read_index("data/embeddings/index.faiss")
    with open("data/embeddings/jobs.pkl", "rb") as f:
        jobs = pickle.load(f)

    print("🔍 Running similarity search...")
    indices, distances = search_similar(resume_vec, index, top_k)

    print("\n🔗 Top Matching Jobs:")
    for i, idx in enumerate(indices):
        job_id, title, description = jobs[idx]
        print(f"\n#{i+1}: {title}")
        print(f"Job ID: {job_id}")
        print(f"Distance: {distances[i]:.4f}")
        print(f"Description Snippet: {description[:300]}...\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_similarity.py path/to/resume.pdf")
    else:
        test_similarity(sys.argv[1])