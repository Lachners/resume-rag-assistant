# Resume-Feedback RAG Application

This is a personal project I built during the month of April 2025, the main idea being here an AI system capable of providing insightful feedback to our own resume locally,  
respecting therefore our privacy in the process and aiming to help potential candidates to find roles that may fit them well, based on their CV.

The app takes the uploaded resume (in PDF), extracts its text, embeds it, and compares it to job descriptions pulled from a local database (based on the Kaggle dataset: LinkedIn Job Postings (2023–2024) by Arsh Koneru).  
It then provides AI-generated feedback about how well the resume aligns with specific job postings using a local LLM and mentions the areas towards further growth that could be found.

---

## How It Works – A quick overview of how everything comes together:

### 1. Resume Upload and Parsing

    User uploads a PDF resume via the Streamlit interface.  
    The backend uses pdfminer.six to extract text from the PDF cleanly.

### 2. Resume Embedding

    The text from the resume is embedded using sentence-transformers (all-mpnet-base-v2).  
    These embeddings are ready to be compared to job descriptions.

### 3. Job Description Database

    The app uses a local PostgreSQL database populated with job postings from the Kaggle dataset mentioned above.  
    Each job description is preprocessed and embedded once during setup.

### 4. Semantic Similarity Matching (FAISS)

    FAISS is used to perform efficient similarity searches between the resume and job descriptions.  
    The most relevant matches are retrieved using cosine similarity combined with a keyword pairing.

### 5. RAG + Feedback Generation

    A local LLM (via Ollama) takes the top N most similar job descriptions and the resume content.  
    A prompt guides the model to generate feedback based strictly on the actual content and according to the needs of this task.  
    A separate fact-checking model (bespoke-minicheck:7b) helps catch potential LLM hallucinations and ensures factual consistency.

---

## Ressources and Libraries:

- **Frontend**: Streamlit  
- **LLM**: Ollama – `deepseek-r1:7b`, `bespoke-minicheck:7b`  
- **Parsing**: `pdfminer.six`  
- **Embeddings**: `sentence-transformers` – `all-mpnet-base-v2`  
- **Database**: PostgreSQL  
- **Dataset**: Arsh Koneru. (2024). *LinkedIn Job Postings (2023 - 2024)* [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/9200871  
- **Deployment**: Docker, Render (In Process)

---



Local Setup:

1. **Clone the Repo**:
2. Install dependencies:
   "pip install -r requirements.txt"
3. **Pull the Ollama LLMs**:
   "ollama pull deepseek-r1:7b"
   "ollama pull bespoke-minicheck:7b"
   Note: `all-mpnet-base-v2` is pulled automatically when needed by `sentence-transformers`
5. **Run the App**:
   "streamlit run streamlit_app.py"




