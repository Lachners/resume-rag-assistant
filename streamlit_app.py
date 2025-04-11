import streamlit as st
import faiss
import pickle
import tempfile
import os

from app.resume_parser import extract_text_from_pdf
from app.vector_store import embed_text, search_similar
from app.rag_agent import generate_resume_feedback

# Set up page
st.set_page_config(page_title="AI Resume Assistant", layout="wide")
st.title("📄 Resume Feedback Chatbot")

# Load FAISS index and job metadata once
@st.cache_resource
def load_index_and_jobs():
    index = faiss.read_index("data/embeddings/index.faiss")
    with open("data/embeddings/jobs.pkl", "rb") as f:
        jobs = pickle.load(f)
    return index, jobs

index, jobs = load_index_and_jobs()

# Sidebar – Resume upload
st.sidebar.title("Upload your resume")
uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

# Process and analyze resume
if uploaded_file:
    with st.spinner("Extracting resume text..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            resume_path = tmp.name

        resume_text = extract_text_from_pdf(resume_path)
        st.sidebar.success("✅ Resume text extracted.")
        st.subheader("📋 Extracted Resume Text")
        st.text_area("This is your resume content:", resume_text, height=300)

        # Embed and search
        with st.spinner("🔍 Matching your profile with job roles..."):
            resume_embedding = embed_text([resume_text])[0]
            indices, _ = search_similar(resume_embedding, index, top_k=3)

        st.subheader("🧠 Top Matching Job Feedback")

        # Show results
        for rank, idx in enumerate(indices, start=1):
            job_id, title, description = jobs[idx]
            with st.expander(f"Match #{rank}: {title}"):
                with st.spinner("💬 Generating feedback..."):
                    feedback = generate_resume_feedback(resume_text, title, description)
                st.markdown(f"### Feedback for: **{title}**")
                st.markdown(feedback)

        # Cleanup
        os.remove(resume_path)
