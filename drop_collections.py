from hashlib import md5
import os
import faiss
import numpy as np
from dotenv import load_dotenv
from together import Together
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import asyncio
from urllib.parse import urljoin
import google.generativeai as genai

# ================== Environment Variables ==================
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGO_URI = os.getenv("MongoDB")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# =================== Connections ============================
# Configure Gemini (Google Generative AI)
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# MongoDB Connection
mongo_client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
db = mongo_client["userembeddings"]
collection = db["embeddings"]

client = Together(api_key=TOGETHER_API_KEY)

def drop_and_create_collections():
    """Drops collections individually and recreates them."""
    global db, collection, faiss_index
    
    # Get all collection names
    collections = db.list_collection_names()
    
    # Drop each collection
    for coll in collections:
        db[coll].drop()
        print(f"Collection '{coll}' dropped.")

    # Reinitialize collections
    collection = db["embeddings"]
    
    # Reinitialize FAISS index
    faiss_index = faiss.IndexFlatL2(384)
    
    print("All collections dropped and 'embeddings' collection recreated successfully.")


# def compute_file_hash(file):
#     """Computes MD5 hash of the file content."""
#     hasher = md5()
#     for chunk in iter(lambda: file.read(4096), b""):
#         hasher.update(chunk)
#     file.seek(0)  # Reset file pointer
#     return hasher.hexdigest()

# def update_vector_db_with_hash(texts, file):
#     """Updates the vector database with text embeddings and includes a file hash."""
#     if not texts:
#         return
    
#     file_hash = compute_file_hash(file)
#     embeddings = embedding_model.encode(texts).tolist()
#     documents = [{"file_hash": file_hash, "text": text, "embedding": emb} for text, emb in zip(texts, embeddings)]
#     collection.insert_many(documents)
#     faiss_index.add(np.array(embeddings, dtype="float32"))
    
#     print("Vector database updated with file hash.")

drop_and_create_collections()
