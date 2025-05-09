import streamlit as st
import faiss
import pickle
import tempfile
import os
from dotenv import load_dotenv


load_dotenv()

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL")

from app.resume_parser import extract_text_from_pdf
from app.vector_store import embed_text, search_similar
from app.rag_agent import generate_resume_feedback

# Set up page
st.set_page_config(page_title="AI Resume Assistant", layout="wide")
st.title("📄 Resume Feedback Chatbot")
st.subheader("Upload your resume, get it contrasted with job descriptions from our database, and receive AI-generated feedback.")

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
            job_descriptions = [job[2] for job in jobs]  # Extract job description texts from loaded job tuples
            indices, _ = search_similar(resume_embedding, index, job_descriptions, resume_text, top_k=3)


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

st.info("This project aims to provide job-search-related feedback by analyzying your resume and contrasting it with job descriptions selected via a hybrid score, which uses keyword filtering and semantic similarity to generate role-specific feedback using a local LLM - DeepSeek-r1 (7b) in this case." \
" The retrieved descriptions are stored within a PostgreSQL database, heavily based on the following dataset: \n"
"\n\n" \
"Arsh Koneru. (2024). LinkedIn Job Postings (2023 - 2024) [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/9200871 \n" 
"\n\n" \
"Prompt engineering constraints were used to avoid LLM hallucination, therefore enforcing citation of resume content for the responses. In addition to that, it uses a secondary agent (Bespoke Minicheck) to fact-check and revise any ungrounded claims. \n The system scores and explains how fit the candidate is for each role, with a final overview." \
"\n\n" \
"**Disclaimer:** This is a project prototype. The feedback provided is generated by an AI model and should not be considered as professional career advice, therefore, hallucinations or potential errors are possible. Always consult with a human expert for personalized guidance.")
