from app.resume_parser import extract_text_from_pdf
from app.vector_store import embed_text, search_similar
from app.rag_agent import generate_resume_feedback
import pickle
import faiss

resume_path = "path/to/resume.pdf"
resume_text = extract_text_from_pdf(resume_path)
resume_embedding = embed_text([resume_text])[0]

# Load FAISS index and job metadata
index = faiss.read_index("data/embeddings/index.faiss")
with open("data/embeddings/jobs.pkl", "rb") as f:
    jobs = pickle.load(f)

# Find top matches
indices, _ = search_similar(resume_embedding, index, top_k=3)

for idx in indices:
    job_id, title, description = jobs[idx]
    feedback = generate_resume_feedback(resume_text, title, description)
    print(f"\n🧠 Feedback for job: {title}\n{feedback}")