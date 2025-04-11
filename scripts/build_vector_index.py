import os
import numpy as np
import pickle
import faiss

from app import db, vector_space
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Make sure directories exist
os.makedirs("data/embeddings", exist_ok=True)

# Step 1: Load job descriptions from DB
jobs = db.fetch_job_descriptions(limit=10000)  # or 12600
descriptions = [desc for _, _, desc in jobs]

# Step 2: Embed descriptions
print("📦 Embedding job descriptions...")
embeddings = vector_space.embed_texts(descriptions)

# Step 3: Build FAISS index and save
print("💾 Saving FAISS index and job metadata...")
index = vector_space.build_faiss_index(np.array(embeddings))
faiss.write_index(index, "data/embeddings/index.faiss")

with open("data/embeddings/jobs.pkl", "wb") as f:
    pickle.dump(jobs, f)

print("✅ Done! You can now run main.py")