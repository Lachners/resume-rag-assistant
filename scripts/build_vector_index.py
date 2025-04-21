import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import numpy as np
import pickle
import faiss
from app import db, vector_store



# Make sure directories exist
os.makedirs("data/embeddings", exist_ok=True)

# Step 1: Load job descriptions from DB
jobs = db.fetch_job_descriptions()  # or 12600
descriptions = [desc for _, _, desc in jobs]

# Step 2: Embed descriptions
print("📦 Embedding job descriptions...")
embeddings = vector_store.embed_text(descriptions)

# Step 3: Build FAISS index and save
print("💾 Saving FAISS index and job metadata...")
index = vector_store.build_faiss_index(np.array(embeddings))
faiss.write_index(index, "data/embeddings/index.faiss")

with open("data/embeddings/jobs.pkl", "wb") as f:
    pickle.dump(jobs, f)

print("✅ Done! You can now run main.py")