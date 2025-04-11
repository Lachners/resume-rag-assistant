# main.py

import sys
import os
from app.resume_parser import extract_text_from_pdf
from app.vector_space import embed_text, search_similar
from app.rag_agent import generate_resume_feedback
from app import db

import pickle
import faiss


def main(resume_path: str, top_k: int = 3):
    # Step 1: Extract resume text from PDF
    if not os.path.exists(resume_path):
        print(f"❌ File not found: {resume_path}")
        return

    print("📄 Extracting text from resume...")
    resume_text = extract_text_from_pdf(resume_path)

    # Step 2: Embed resume
    print("🔍 Embedding resume...")
    resume_embedding = embed_text([resume_text])[0]

    # Step 3: Load FAISS index and job metadata
    print("📦 Loading FAISS index and job metadata...")
    index = faiss.read_index("data/embeddings/index.faiss")
    with open("data/embeddings/jobs.pkl", "rb") as f:
        jobs = pickle.load(f)

    # Step 4: Search top matches
    print("🧠 Searching for best matching jobs...")
    indices, distances = search_similar(resume_embedding, index, top_k=top_k)

    # Step 5: Generate feedback for each match
    for rank, idx in enumerate(indices, 1):
        job_id, title, description = jobs[idx]
        print(f"\n🔹 Match #{rank}: {title}")
        print(f"   ID: {job_id} | Similarity Score: {distances[rank-1]:.4f}")

        print("📝 Generating feedback...")
        feedback = generate_resume_feedback(resume_text, title, description)
        print("🎯 AI Feedback:\n")
        print(feedback)
        print("\n" + "-" * 50)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py path/to/resume.pdf")
    else:
        resume_pdf_path = sys.argv[1]
        main(resume_pdf_path)
