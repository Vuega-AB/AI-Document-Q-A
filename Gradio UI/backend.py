import os
import requests
from bs4 import BeautifulSoup
import aiohttp
import asyncio
import PyPDF2
import faiss
import time # Make sure time is imported
import numpy as np
from sentence_transformers import SentenceTransformer
from langdetect import detect
import json
from dotenv import load_dotenv
from io import BytesIO
from together import Together
import re
from pymongo import MongoClient, server_api
import logging
from openai import OpenAI
import sys
import httpx
from urllib.parse import urljoin
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
import dropbox
import hashlib
import io
import tempfile
import bcrypt
import gradio
from datetime import datetime # Make sure datetime is imported

# --- Environment Variables & Initializations ---
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGO_URI = os.getenv("MongoDB") 
if not MONGO_URI:
    MONGO_URI = os.getenv("MONGO_URI")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DROPBOX_REFRESH_TOKEN = os.getenv("DROPBOX_REFRESH_TOKEN")
DROPBOX_APP_KEY = os.getenv("DROPBOX_APP_KEY")
DROPBOX_APP_SECRET = os.getenv("DROPBOX_APP_SECRET")

CONFIG_FILENAME = "app_config_main.json"
INDEX_FILE_DROPBOX = "/faiss_index.index"
TEXT_FILE_DROPBOX = "/text_store.json"
TOKEN_FILE = "dropbox_token.json"
MONGO_DB_NAME = "IntelLawDB_Gradio"
FAISS_COLLECTION_NAME = "faiss_index_store"
TEXT_STORE_COLLECTION_NAME = "text_content_store"
ADMIN_USERS_COLLECTION_NAME = "admin_users" # User credentials and status

# --- Global Variables for Backend State ---
gemini_model_genai = None
together_client = None
openai_client = None
dbx = None
mongo_client_instance = None
mongo_db_obj = None
auth_mongo_db_obj = None
embedding_model = None
faiss_index = None
text_store = []
BACKEND_INITIAL_LOAD_MSG = "Backend not initialized."

AVAILABLE_MODELS_DICT = {
    "gemini-1.5-flash-latest": {"price": "Custom", "type": "gemini", "name": "Gemini 1.5 Flash"},
    "openai-gpt-4o": {"price": "Custom", "type": "openai", "name": "OpenAI GPT-4o"},
    "meta-llama/Llama-3-70B-Instruct-hf": {"price": "$0.90", "type": "together", "name": "Llama3 70B Instruct (HF)"},
    "meta-llama/Llama-3-8B-Instruct-hf": {"price": "$0.20", "type": "together", "name": "Llama3 8B Instruct (HF)"},
    "microsoft/WizardLM-2-8x22B": {"price": "$1.80", "type": "together", "name": "WizardLM-2 8x22B"},
    "mistralai/Mixtral-8x22B-Instruct-v0.1": {"price": "$1.20", "type": "together", "name": "Mixtral 8x22B Instruct"},
    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO": {"price": "$0.60", "type": "together", "name": "Hermes-2 Mixtral DPO"},
}
AVAILABLE_MODELS_NAMES = [details['name'] for details in AVAILABLE_MODELS_DICT.values()]
MODEL_NAME_TO_ID_MAP = {details['name']: model_id for model_id, details in AVAILABLE_MODELS_DICT.items()}
MODEL_ID_TO_NAME_MAP = {v: k for k, v in MODEL_NAME_TO_ID_MAP.items()}

# --- Password Hashing Functions ---
def hash_password(password: str) -> bytes:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def verify_password(plain_password: str, hashed_password_bytes: bytes) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password_bytes)

# --- MongoDB User Authentication Functions ---
def get_auth_db():
    global auth_mongo_db_obj, mongo_client_instance
    if auth_mongo_db_obj is None:
        if mongo_client_instance is None:
            initialize_mongodb_client() # Ensures client is initialized
        if mongo_client_instance:
            # Use the same DB instance for auth, just different collections
            auth_mongo_db_obj = mongo_client_instance[MONGO_DB_NAME] 
    return auth_mongo_db_obj

def create_admin_user_if_not_exists(email, plain_password, role="admin"):
    db = get_auth_db()
    if db is None:
        print("Error: MongoDB for auth not available. Cannot create/update admin user.")
        return False
    users_collection = db[ADMIN_USERS_COLLECTION_NAME]
    
    user = users_collection.find_one({"email": email})
    if user:
        print(f"User {email} already exists.")
        if user.get("status") != "active" or user.get("role") != "admin":
            users_collection.update_one(
                {"email": email},
                {"$set": {"status": "active", "role": "admin", "updated_at": time.time()}}
            )
            print(f"Updated user {email} to ensure admin role and active status.")
        return True
    
    hashed_pass = hash_password(plain_password)
    try:
        users_collection.insert_one({
            "email": email,
            "password": hashed_pass,
            "role": role,
            "status": "active",
            "created_at": time.time(), # Storing as float (Unix timestamp)
            "updated_at": time.time()
        })
        print(f"Admin user {email} created successfully with active status.")
        return True
    except Exception as e:
        print(f"Error creating admin user {email}: {e}")
        return False

def create_user(email, plain_password):
    db = get_auth_db()
    if db is None:
        return False, "Database error, please try again later."
    
    users_collection = db[ADMIN_USERS_COLLECTION_NAME]
    if users_collection.find_one({"email": email}):
        return False, "Email address already registered."
    
    hashed_pass = hash_password(plain_password)
    try:
        users_collection.insert_one({
            "email": email,
            "password": hashed_pass,
            "role": "user",
            "status": "pending", # New users start as pending
            "created_at": time.time(), # Storing as float (Unix timestamp)
            "updated_at": time.time()
            # Consider adding 'full_name': email.split('@')[0] here if desired
        })
        print(f"User {email} registered with pending status.")
        return True, "Registration successful! Your account is pending admin approval."
    except Exception as e:
        print(f"Error creating user {email}: {e}")
        return False, "An error occurred during registration."

def get_user_by_email(email):
    db = get_auth_db()
    if db is None:
        print("Error: MongoDB for auth not available. Cannot get user.")
        return None
    users_collection = db[ADMIN_USERS_COLLECTION_NAME]
    return users_collection.find_one({"email": email})

def get_all_users_from_db():
    """Fetches all users from the database for admin display."""
    db = get_auth_db()
    if db is None:
        print("ERROR: MongoDB for auth not available in backend.get_all_users_from_db")
        return []
    
    users_collection = db[ADMIN_USERS_COLLECTION_NAME]
    try:
        users_cursor = users_collection.find({})
        users_list = []
        for user_doc in users_cursor:
            user_doc['_id'] = str(user_doc['_id'])
            
            if 'password' in user_doc: # Never send password hash to frontend
                del user_doc['password']
            
            # Convert 'created_at' from float timestamp to datetime object for consistent processing
            if 'created_at' in user_doc and isinstance(user_doc['created_at'], (int, float)):
                user_doc['created_at'] = datetime.fromtimestamp(user_doc['created_at'])
            
            # Convert 'updated_at' if it exists and is a float timestamp
            if 'updated_at' in user_doc and isinstance(user_doc['updated_at'], (int, float)):
                user_doc['updated_at'] = datetime.fromtimestamp(user_doc['updated_at'])

            # Convert 'last_login_at' if it exists (assuming it might be stored as float)
            if 'last_login_at' in user_doc and isinstance(user_doc['last_login_at'], (int, float)):
                user_doc['last_login_at'] = datetime.fromtimestamp(user_doc['last_login_at'])
            
            # Ensure 'full_name' for display, derive from email if not present
            if 'full_name' not in user_doc and 'email' in user_doc:
                user_doc['full_name'] = user_doc['email'].split('@')[0]
            elif 'full_name' not in user_doc:
                user_doc['full_name'] = "N/A"
                
            users_list.append(user_doc)
        return users_list
    except Exception as e:
        print(f"Error fetching all users: {e}")
        return []

def update_user_status_in_db(user_email, new_status):
    """Updates the status of a user in the database."""
    db = get_auth_db()
    if db is None:
        print("ERROR: MongoDB for auth not available in backend.update_user_status_in_db")
        return False, "Database not connected."
    
    users_collection = db[ADMIN_USERS_COLLECTION_NAME]
    
    allowed_statuses = ["active", "pending", "suspended", "deactivated"]
    if new_status not in allowed_statuses:
        return False, f"Invalid status '{new_status}'. Allowed statuses are: {', '.join(allowed_statuses)}."

    try:
        # Use float timestamp for updated_at, consistent with created_at
        current_time_for_update = time.time() 

        result = users_collection.update_one(
            {"email": user_email},
            {"$set": {"status": new_status, "updated_at": current_time_for_update}}
        )
        if result.matched_count == 0:
            return False, f"User with email '{user_email}' not found."
        if result.modified_count == 0:
            # Check if the status was already the new_status
            current_user = users_collection.find_one({"email": user_email})
            if current_user and current_user.get("status") == new_status:
                 return True, f"User status for '{user_email}' was already '{new_status}'. No change made but considered success." # Treat as success
            return False, f"User status for '{user_email}' could not be updated (already '{new_status}' or other issue)."
        return True, f"User '{user_email}' status updated to '{new_status}'."
    except Exception as e:
        print(f"Error updating user status for {user_email}: {e}")
        return False, "An error occurred while updating user status."

# --- Dropbox Functions ---
# ... (existing Dropbox functions: load_access_token, save_access_token, get_dropbox_access_token, get_valid_access_token, initialize_dropbox_client) ...
def load_access_token():
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "r") as file: data = json.load(file); return data.get("access_token"), data.get("expires_at")
    return None, None

def save_access_token(access_token, expires_in):
    expires_at = int(time.time()) + expires_in - 300
    with open(TOKEN_FILE, "w") as file: json.dump({"access_token": access_token, "expires_at": expires_at}, file)

def get_dropbox_access_token():
    if not (DROPBOX_APP_KEY and DROPBOX_APP_SECRET and DROPBOX_REFRESH_TOKEN):
        print("Error: Dropbox credentials (APP_KEY, APP_SECRET, REFRESH_TOKEN) not fully configured.")
        return None
    response = requests.post("https://api.dropbox.com/oauth2/token", data={"grant_type": "refresh_token", "refresh_token": DROPBOX_REFRESH_TOKEN}, auth=(DROPBOX_APP_KEY, DROPBOX_APP_SECRET))
    if response.status_code == 200:
        data = response.json()
        save_access_token(data["access_token"], data.get("expires_in", 14400))
        return data["access_token"]
    else:
        print(f"Failed to refresh Dropbox token: {response.text}")
        return None

def get_valid_access_token():
    access_token, expires_at = load_access_token()
    if access_token and expires_at and int(time.time()) < expires_at: return access_token
    return get_dropbox_access_token()

def initialize_dropbox_client():
    global dbx
    if not (DROPBOX_REFRESH_TOKEN and DROPBOX_APP_KEY and DROPBOX_APP_SECRET):
        print("Warning: Dropbox credentials not fully set. Dropbox features disabled.")
        dbx = None; return
    try:
        access_token = get_valid_access_token()
        if access_token:
            dbx = dropbox.Dropbox(access_token)
            print("Dropbox client initialized successfully.")
        else:
            print("Failed to obtain Dropbox access token. Dropbox client not initialized.")
            dbx = None
    except Exception as e:
        print(f"Error connecting to Dropbox: {e}")
        dbx = None

# --- MongoDB Functions (for app data) ---
def initialize_mongodb_client():
    global mongo_client_instance, mongo_db_obj, auth_mongo_db_obj
    if MONGO_URI and mongo_client_instance is None:
        try:
            mongo_client_instance = MongoClient(MONGO_URI, server_api=server_api.ServerApi('1'))
            mongo_client_instance.admin.command('ping') # Verify connection
            mongo_db_obj = mongo_client_instance[MONGO_DB_NAME] # For app data (FAISS, text_store)
            auth_mongo_db_obj = mongo_client_instance[MONGO_DB_NAME] # For auth data (admin_users)
                                                                  # Using same DB, but conceptually could be different
            print("MongoDB client initialized successfully.")
        except Exception as e:
            print(f"MongoDB connection failed: {e}")
            mongo_client_instance = None; mongo_db_obj = None; auth_mongo_db_obj = None
    elif not MONGO_URI:
        print("Warning: MONGO_URI not set. MongoDB features disabled.")
    # else:
        # print("MongoDB client already initialized or MONGO_URI not set.")


# --- Data Persistence Functions (FAISS, text_store) ---
# ... (existing: save_data_to_selected_db, load_data_from_selected_db) ...
def save_data_to_selected_db(selected_db):
    global faiss_index, text_store, dbx, mongo_db_obj

    if selected_db == "Dropbox" and dbx is None: initialize_dropbox_client()
    if selected_db == "MongoDB" and mongo_db_obj is None: initialize_mongodb_client() 

    index_to_save = faiss_index
    if index_to_save is None or embedding_model is None:
        print("Warning: FAISS index or embedding model is None. Cannot save empty or uninitialized index.")
        if embedding_model and index_to_save is None:
            index_to_save = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
            print("Created new empty FAISS index for saving.")
        else:
            return

    if selected_db == "Dropbox":
        if dbx is None: print("Dropbox not initialized for saving."); return
        try:
            if index_to_save.ntotal == 0:
                print("Skipping FAISS index save to Dropbox as it's empty.")
            else:
                temp_idx_file = "temp_faiss_to_dropbox.index"
                faiss.write_index(index_to_save, temp_idx_file)
                with open(temp_idx_file, "rb") as f: dbx.files_upload(f.read(), INDEX_FILE_DROPBOX, mode=dropbox.files.WriteMode.overwrite)
                os.remove(temp_idx_file)
            
            text_json = json.dumps(text_store, ensure_ascii=False, indent=4).encode('utf-8')
            dbx.files_upload(text_json, TEXT_FILE_DROPBOX, mode=dropbox.files.WriteMode.overwrite)
            print(f"Data saved to Dropbox. Index size: {index_to_save.ntotal}, Text items: {len(text_store)}")
        except Exception as e: print(f"Error saving to Dropbox: {e}")
    elif selected_db == "MongoDB":
        if mongo_db_obj is None: print("MongoDB not initialized for saving app data."); return
        try:
            if index_to_save.ntotal == 0:
                 print("Skipping FAISS index save to MongoDB as it's empty.")
                 mongo_db_obj[FAISS_COLLECTION_NAME].delete_one({"_id": "main_faiss_index"})
            else:
                temp_idx_file = "temp_faiss_to_mongo.idx"
                faiss.write_index(index_to_save, temp_idx_file)
                with open(temp_idx_file, "rb") as f: index_bytes = f.read()
                os.remove(temp_idx_file)
                mongo_db_obj[FAISS_COLLECTION_NAME].update_one({"_id": "main_faiss_index"}, {"$set": {"index_data": index_bytes}}, upsert=True)

            mongo_db_obj[TEXT_STORE_COLLECTION_NAME].delete_many({})
            if text_store: mongo_db_obj[TEXT_STORE_COLLECTION_NAME].insert_many(text_store)
            print(f"Data saved to MongoDB. Index size: {index_to_save.ntotal}, Text items: {len(text_store)}")
        except Exception as e: print(f"Error saving to MongoDB: {e}")


def load_data_from_selected_db(selected_db):
    global faiss_index, text_store, embedding_model, dbx, mongo_db_obj
    if embedding_model is None:
        print("Error: Embedding model not initialized. Cannot load data.")
        return "Embedding model not initialized. Load failed."

    faiss_index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
    text_store = []
    alert_msg = ""

    if selected_db == "Dropbox":
        if dbx is None: initialize_dropbox_client()
        if dbx is None: alert_msg = "Dropbox not initialized. Cannot load."; return alert_msg
        try:
            _, res_index = dbx.files_download(path=INDEX_FILE_DROPBOX)
            temp_idx_file = "temp_faiss_from_dropbox.index"
            with open(temp_idx_file, "wb") as f: f.write(res_index.content)
            loaded_index = faiss.read_index(temp_idx_file)
            if loaded_index.d == faiss_index.d:
                faiss_index = loaded_index
            else:
                alert_msg += f"Warning: Dropbox index dimension mismatch ({loaded_index.d} vs {faiss_index.d}). Not loading index. "
            os.remove(temp_idx_file)

            _, res_text = dbx.files_download(path=TEXT_FILE_DROPBOX)
            text_store = json.loads(res_text.content.decode('utf-8'))
            alert_msg += f"Data loaded from Dropbox. {len(text_store)} text items, index has {faiss_index.ntotal} vectors."
        except dropbox.exceptions.ApiError as e:
            if isinstance(e.error, dropbox.files.DownloadError) and e.error.is_path() and e.error.get_path().is_not_found():
                alert_msg += "No existing data on Dropbox. Initialized empty store."
            else: alert_msg += f"Dropbox API error: {e}"
        except Exception as e: alert_msg += f"Error loading from Dropbox: {e}"

    elif selected_db == "MongoDB":
        if mongo_db_obj is None: initialize_mongodb_client() 
        if mongo_db_obj is None: alert_msg = "MongoDB not initialized for app data. Cannot load."; return alert_msg
        try:
            index_doc = mongo_db_obj[FAISS_COLLECTION_NAME].find_one({"_id": "main_faiss_index"})
            if index_doc and "index_data" in index_doc:
                temp_idx_file = "temp_faiss_from_mongo.idx"
                with open(temp_idx_file, "wb") as f: f.write(index_doc["index_data"])
                loaded_index = faiss.read_index(temp_idx_file)
                if loaded_index.d == faiss_index.d:
                    faiss_index = loaded_index
                else:
                    alert_msg += f"Warning: MongoDB index dimension mismatch ({loaded_index.d} vs {faiss_index.d}). Not loading index. "
                os.remove(temp_idx_file)
            
            text_docs = list(mongo_db_obj[TEXT_STORE_COLLECTION_NAME].find({}))
            text_store = [{k: v for k, v in doc.items() if k != '_id'} for doc in text_docs]
            alert_msg += f"Data loaded from MongoDB. {len(text_store)} text items, index has {faiss_index.ntotal} vectors."
            if not index_doc and not text_docs and not alert_msg: # if no error, and no data
                alert_msg = "No existing data on MongoDB. Initialized empty store."
        except Exception as e: alert_msg += f"Error loading from MongoDB: {e}"
    
    if faiss_index.ntotal > 0 and not text_store:
        alert_msg += " Warning: Index has vectors but text store is empty. Data might be corrupt. Clearing index."
        faiss_index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
    elif not faiss_index.ntotal and text_store: # Index empty but text_store has data
        alert_msg += " Warning: Text store loaded but index is empty/failed to load. Consider re-indexing or checking data integrity."


    print(f"Load attempt for {selected_db}: {alert_msg}")
    return alert_msg if alert_msg else "Data loaded successfully. Store might be empty."

# --- PDF Processing ---
# ... (existing: chunk_text, extract_text_from_pdf_bytes, process_and_add_pdf_core) ...
def chunk_text(text, chunk_size=400, min_chunk_length=20):
    paragraphs = re.split(r'\n{2,}', text); chunks = []
    for para in paragraphs:
        sentences = re.split(r'(?<=[.!?])\s+', para); temp_chunk = ""
        for sentence in sentences:
            if len(temp_chunk) + len(sentence) < chunk_size: temp_chunk += sentence + " "
            else:
                cleaned_chunk = temp_chunk.strip()
                if len(cleaned_chunk) >= min_chunk_length: chunks.append(cleaned_chunk)
                temp_chunk = sentence + " "
        cleaned_chunk = temp_chunk.strip()
        if len(cleaned_chunk) >= min_chunk_length: chunks.append(cleaned_chunk)
    return chunks

def extract_text_from_pdf_bytes(pdf_bytes):
    reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes)); text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        text += page_text + "\n" if page_text else ""
    return text

def process_and_add_pdf_core(pdf_bytes, file_name, selected_db):
    global text_store, faiss_index, embedding_model
    if embedding_model is None: return "Embedding model not initialized.", False
    if faiss_index is None: 
        if embedding_model: faiss_index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
        else: return "FAISS index not initialized and embedding model missing.", False

    file_hash = hashlib.md5(pdf_bytes).hexdigest()
    if any(item.get('file_hash') == file_hash for item in text_store if isinstance(item, dict)):
        return f"File '{file_name}' (hash: {file_hash[:7]}) seems to already exist based on hash.", False
    
    raw_text = extract_text_from_pdf_bytes(pdf_bytes)
    if not raw_text.strip():
        return f"No text could be extracted from '{file_name}'.", False
        
    chunks = chunk_text(raw_text)
    if not chunks:
        return f"Could not break '{file_name}' into processable chunks.", False

    embeddings = embedding_model.encode(chunks)
    embeddings_np = np.array(embeddings).astype("float32")
    if embeddings_np.shape[0] > 0: faiss_index.add(embeddings_np)
    
    for chunk_text_content in chunks:
        text_store.append({"text": chunk_text_content, "file_name": file_name, "file_hash": file_hash})
    return f"Processed and added '{file_name}'. Chunks: {len(chunks)}, Index size: {faiss_index.ntotal}", True

# --- AI Response Generation ---
# ... (existing: generate_response_gemini, generate_response_together_ai, generate_response_openai_api) ...
def generate_response_gemini(prompt, context, temp, top_p, system_prompt):
    if not gemini_model_genai: return "Gemini client not initialized."
    input_parts = [system_prompt + "\nContext: " + context, "Question: " + prompt]
    config = genai.GenerationConfig(max_output_tokens=2048, temperature=temp, top_p=top_p)
    try: response = gemini_model_genai.generate_content(input_parts, generation_config=config); return response.text
    except Exception as e: return f"Gemini Error: {e}"

def generate_response_together_ai(prompt, context, model_id, temp, top_p, system_prompt):
    if not together_client: return "TogetherAI client not initialized."
    try:
        response = together_client.chat.completions.create(
            model=model_id, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Context: {context}\nQuestion: {prompt}"}],
            temperature=temp, top_p=top_p
        )
        return response.choices[0].message.content.strip()
    except Exception as e: return f"TogetherAI Error ({model_id}): {e}"

def generate_response_openai_api(prompt, context, temp, top_p, system_prompt): 
    if not openai_client: return "OpenAI client not initialized."
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Context: {context}\nQuestion: {prompt}"}],
            temperature=temp, top_p=top_p
        )
        return response.choices[0].message.content
    except Exception as e: return f"OpenAI Error: {e}"

# --- RAG ---
# ... (existing: retrieve_context_from_db) ...
def retrieve_context_from_db(query, top_k=5):
    global text_store, faiss_index, embedding_model
    if embedding_model is None: return "Embedding model not initialized for RAG."
    if faiss_index is None or faiss_index.ntotal == 0: return "No documents indexed."
    
    query_embedding = embedding_model.encode([query])
    query_embedding_np = np.array(query_embedding).astype("float32")
    
    if query_embedding_np.shape[1] != faiss_index.d:
        return f"Query embedding dimension ({query_embedding_np.shape[1]}) does not match FAISS index dimension ({faiss_index.d})."

    distances, indices = faiss_index.search(query_embedding_np, top_k)
    
    valid_indices = [i for i in indices[0] if 0 <= i < len(text_store)]
    retrieved_texts = [text_store[idx]["text"] for idx in valid_indices if isinstance(text_store[idx], dict) and "text" in text_store[idx]]
    return "\n\n".join(retrieved_texts) if retrieved_texts else "No relevant context found."


# --- Web Scraping ---
# ... (existing web scraping functions) ...
async def fetch_page_async(url):
    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=30.0, follow_redirects=True)
        response.raise_for_status() 
        return response.text, str(response.url)

async def extract_pdf_links_from_url_async(url):
    try:
        html, base_url_resolved = await fetch_page_async(url)
        soup = BeautifulSoup(html, "html.parser")
        return [urljoin(base_url_resolved, a["href"]) for a in soup.find_all("a", href=True) if ".pdf" in a["href"].lower()]
    except Exception as e:
        print(f"Error scraping {url} for PDF links: {e}")
        return []

async def process_scraped_pdf_links_async(urls):
    all_pdf_links = set()
    tasks = [extract_pdf_links_from_url_async(u) for u in urls]
    for url_group in await asyncio.gather(*tasks):
        all_pdf_links.update(url_group)
    return list(all_pdf_links)

async def download_and_process_scraped_pdf(session, pdf_link, selected_db):
    try:
        async with session.get(pdf_link, timeout=60) as response:
            if response.status == 200:
                pdf_bytes = await response.read()
                filename = os.path.basename(pdf_link)
                status_msg, success = process_and_add_pdf_core(pdf_bytes, filename, selected_db) 
                return status_msg, success, filename
            return f"Failed to download {pdf_link} (status: {response.status})", False, os.path.basename(pdf_link)
    except Exception as e:
        return f"Error processing {pdf_link}: {e}", False, os.path.basename(pdf_link)

async def batch_download_and_process_pdfs(pdf_links, selected_db):
    results = []
    connector = aiohttp.TCPConnector(ssl=False) 
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [download_and_process_scraped_pdf(session, link, selected_db) for link in pdf_links]
        for result in await asyncio.gather(*tasks):
            results.append(result)
    return results

def get_page_items_sync(url, base_url_val, listing_endpoint_val):
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser"); items = set()
        for item in soup.find_all("a"):
            link = item.get("href")
            if link and f"/{listing_endpoint_val}/" in link and link != url and not link.endswith("/rss"):
                if not link.startswith("http"): link = urljoin(base_url_val, link)
                items.add(link)
        return list(items)
    except Exception as e: print(f"Error scraping page items {url}: {e}"); return []

def get_all_page_urls_for_scraping(base_url_val, listing_endpoint_val, pagination_format_val, num_pages_val):
    all_page_urls_to_scrape = set()
    for page_num in range(1, int(num_pages_val) + 1):
        url_path = f"{listing_endpoint_val}/{pagination_format_val}{page_num}"
        url = urljoin(base_url_val + ("/" if not base_url_val.endswith("/") else ""), url_path)
        page_items = get_page_items_sync(url, base_url_val, listing_endpoint_val)
        if not page_items and page_num > 1 : 
            print(f"No items found on page {page_num} ({url}), stopping pagination.")
            break
        all_page_urls_to_scrape.update(page_items)
        if not page_items and page_num == 1: 
             print(f"No items found on the first page ({url}). Check scraper settings.")
             break
    return list(all_page_urls_to_scrape)

# --- UI Callable Backend Functions ---
# ... (existing: get_unique_filenames_from_text_store, _build_file_list_updates, etc.) ...
def get_unique_filenames_from_text_store():
    global text_store
    if not text_store: return []
    unique_files = {} 
    for item in text_store:
        if isinstance(item, dict) and 'file_hash' in item and 'file_name' in item:
            if item["file_hash"] not in unique_files:
                unique_files[item["file_hash"]] = item["file_name"]
    return sorted(list(unique_files.values()))

def _build_file_list_updates():
    filenames = get_unique_filenames_from_text_store()
    md_output = "#### Current Files in Database\n---\n"
    if filenames:
        for f_name in filenames: md_output += f"- {f_name}\n"
    else: md_output += "_No files currently in this database._\n"
    checkbox_group_update = gradio.update(choices=filenames, value=[]) 
    markdown_update = gradio.update(value=md_output)
    return checkbox_group_update, markdown_update

def get_current_file_list_md_backend():
    _ , md_update = _build_file_list_updates()
    return md_update.get('value', "_Error generating file list._")

def update_app_config_backend(models_names, vary_t, temp, vary_p, top_p, sys_prompt, current_app_config_state):
    selected_model_ids = [MODEL_NAME_TO_ID_MAP[name] for name in models_names if name in MODEL_NAME_TO_ID_MAP]
    if len(selected_model_ids) > 3: selected_model_ids = selected_model_ids[:3]
    
    current_app_config_state.update({
        "selected_models": selected_model_ids,
        "vary_temperature": vary_t, "temperature": temp,
        "vary_top_p": vary_p, "top_p": top_p,
        "system_prompt": sys_prompt
    })
    return "Configuration updated.", current_app_config_state

def switch_db_backend(selected_db_val, current_selected_db_state_value): 
    if selected_db_val == current_selected_db_state_value:
        db_status_msg = f"Already using {selected_db_val}. No change."
    else:
        db_status_msg = load_data_from_selected_db(selected_db_val)
    cb_update, md_update = _build_file_list_updates()
    return selected_db_val, db_status_msg, cb_update, md_update

def handle_pdf_upload_backend(files_obj_list, selected_db_from_state):
    if files_obj_list is None:
        cb_update, md_update = _build_file_list_updates()
        return "No files uploaded.", cb_update, md_update
    alerts = []
    any_successful_upload = False
    for file_obj in files_obj_list:
        file_path = file_obj.name 
        file_display_name = os.path.basename(getattr(file_obj, 'orig_name', file_path))
        with open(file_path, 'rb') as f: pdf_bytes = f.read()
        status_msg, success = process_and_add_pdf_core(pdf_bytes, file_display_name, selected_db_from_state)
        alerts.append(status_msg)
        if success: any_successful_upload = True
    if any_successful_upload: save_data_to_selected_db(selected_db_from_state)
    status_summary = "\n".join(alerts)
    cb_update, md_update = _build_file_list_updates()
    return status_summary, cb_update, md_update

def delete_files_backend(filenames_to_delete, selected_db):
    global text_store, faiss_index, embedding_model
    if not filenames_to_delete:
        cb_update, md_update = _build_file_list_updates()
        return "No files selected for deletion.", cb_update, md_update
    if embedding_model is None or faiss_index is None:
        cb_update, md_update = _build_file_list_updates()
        return "Error: Core components not ready. Deletion aborted.", cb_update, md_update
    kept_text_store_entries_with_original_indices = []
    for i, item in enumerate(text_store):
        if isinstance(item, dict) and item.get("file_name") not in filenames_to_delete:
            kept_text_store_entries_with_original_indices.append((i, item))
    if len(kept_text_store_entries_with_original_indices) == len(text_store):
        cb_update, md_update = _build_file_list_updates()
        return "Selected files not found or no changes made.", cb_update, md_update
    new_text_store = [item for _, item in kept_text_store_entries_with_original_indices]
    new_faiss_index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
    if kept_text_store_entries_with_original_indices:
        original_indices_to_keep = [original_idx for original_idx, _ in kept_text_store_entries_with_original_indices]
        valid_original_indices_to_keep = [idx for idx in original_indices_to_keep if idx < faiss_index.ntotal]
        if valid_original_indices_to_keep:
            vectors_to_keep = faiss_index.reconstruct_n(0, faiss_index.ntotal)
            kept_vectors = vectors_to_keep[valid_original_indices_to_keep, :]
            if kept_vectors.shape[0] > 0: new_faiss_index.add(kept_vectors.astype("float32"))
        else: print("Warning: No valid vectors to keep after filtering indices for deletion.")
    text_store = new_text_store
    faiss_index = new_faiss_index
    save_data_to_selected_db(selected_db)
    num_deleted = len(filenames_to_delete)
    status_msg = f"Successfully deleted {num_deleted} file(s) and their associated data. Index rebuilt."
    cb_update, md_update = _build_file_list_updates()
    return status_msg, cb_update, md_update

def apply_uploaded_config_backend(config_file_obj, current_app_config_state_dict):
    if config_file_obj is None:
        return "No config file uploaded.", False, current_app_config_state_dict, *[gradio.update()]*6
    try:
        with open(config_file_obj.name, 'r') as f: new_config = json.load(f)
        current_app_config_state_dict.update(new_config)
        sel_model_ids = current_app_config_state_dict.get("selected_models", [])
        model_names_for_ui = [MODEL_ID_TO_NAME_MAP[mid] for mid in sel_model_ids if mid in MODEL_ID_TO_NAME_MAP]
        return (
            "Configuration loaded successfully from file.", True,
            current_app_config_state_dict,
            gradio.update(value=model_names_for_ui),
            gradio.update(value=current_app_config_state_dict.get("vary_temperature", True)),
            gradio.update(value=current_app_config_state_dict.get("temperature", 0.7)),
            gradio.update(value=current_app_config_state_dict.get("vary_top_p", False)),
            gradio.update(value=current_app_config_state_dict.get("top_p", 0.9)),
            gradio.update(value=current_app_config_state_dict.get("system_prompt", ""))
        )
    except Exception as e:
        error_msg = f"Error loading config: {e}"
        return error_msg, False, current_app_config_state_dict, *[gradio.update()]*6

def generate_config_for_download_backend(current_app_config_state_dict):
    try:
        config_to_download = { # Only saving a subset, not the full model list.
            "vary_temperature": current_app_config_state_dict.get("vary_temperature", True),
            "temperature": current_app_config_state_dict.get("temperature", 0.7),
            "vary_top_p": current_app_config_state_dict.get("vary_top_p", False),
            "top_p": current_app_config_state_dict.get("top_p", 0.9),
            "system_prompt": current_app_config_state_dict.get("system_prompt", "You are a helpful assistant."),
            "selected_models": current_app_config_state_dict.get("selected_models", []) # Saving selected model IDs
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding='utf-8') as tmp_file:
            json.dump(config_to_download, tmp_file, indent=2)
            tmp_file_path = tmp_file.name
        return tmp_file_path, "Config ready for download.", True
    except Exception as e:
        return None, f"Error generating config file for download: {e}", False

def chat_interface_backend(user_input, chat_history_list_messages, selected_db_state_val, app_config_state_dict):
    if not user_input or not user_input.strip():
        return chat_history_list_messages 

    if chat_history_list_messages is None: 
        chat_history_list_messages = []
    chat_history_list_messages.append({"role": "user", "content": user_input})

    context_text = retrieve_context_from_db(user_input)
    
    temp_config = app_config_state_dict['temperature']
    top_p_config = app_config_state_dict['top_p']
    system_prompt_config = app_config_state_dict['system_prompt']
    
    temp_values_to_run = [temp_config]
    if app_config_state_dict['vary_temperature'] and temp_config > 0.01: 
        temp_values_to_run = sorted(list(set([
            round(max(0.01, temp_config * 0.5), 2), temp_config, 
            round(min(1.0, temp_config * 1.5), 2) if temp_config * 1.5 <=1.0 else temp_config])))

    top_p_values_to_run = [top_p_config]
    if app_config_state_dict['vary_top_p'] and top_p_config > 0.01:
        top_p_values_to_run = sorted(list(set([
            round(max(0.01, top_p_config * 0.5), 2), top_p_config,
            round(min(1.0, top_p_config * 1.5),2) if top_p_config * 1.5 <= 1.0 else top_p_config])))
        
    selected_ai_model_ids = app_config_state_dict.get('selected_models', [])
    
    bot_response_content_parts = [] 

    if not selected_ai_model_ids:
        ai_response_text = "System: No AI model selected in configuration."
        bot_response_content_parts.append(ai_response_text)
    else:
        for model_id in selected_ai_model_ids:
            model_detail = AVAILABLE_MODELS_DICT.get(model_id, {})
            model_type = model_detail.get("type")
            model_display_name = model_detail.get("name", model_id)

            for temp_val in temp_values_to_run:
                for top_p_val in top_p_values_to_run:
                    response_content = f"Error generating response for {model_display_name}."
                    if model_type == "gemini":
                        response_content = generate_response_gemini(user_input, context_text, temp_val, top_p_val, system_prompt_config)
                    elif model_type == "together":
                        response_content = generate_response_together_ai(user_input, context_text, model_id, temp_val, top_p_val, system_prompt_config)
                    elif model_type == "openai":
                        response_content = generate_response_openai_api(user_input, context_text, temp_val, top_p_val, system_prompt_config)
                    
                    model_info_str = f"{model_display_name} (T:{temp_val}, P:{top_p_val})"
                    bot_response_content_parts.append(f"--- {model_info_str} ---\n{response_content}")

                    if not app_config_state_dict['vary_top_p']: break 
                if not app_config_state_dict['vary_temperature']: break 
        
    final_bot_response = "\n\n".join(bot_response_content_parts)
    if not final_bot_response:
        final_bot_response = "No responses generated or models configured."

    chat_history_list_messages.append({"role": "assistant", "content": final_bot_response})
    return chat_history_list_messages

def run_scraper_backend(base_url, endpoint, pagination, num_pages, selected_db_val):
    if sys.platform == "win32" and isinstance(asyncio.get_event_loop_policy(), asyncio.WindowsSelectorEventLoopPolicy):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    status_updates = ["Starting scraping..."]
    page_urls_to_scan = get_all_page_urls_for_scraping(base_url, endpoint, pagination, num_pages)
    status_updates.append(f"Found {len(page_urls_to_scan)} site pages to scan for PDF links.")
    any_successful_scrape_process = False
    if page_urls_to_scan:
        pdf_links_found = asyncio.run(process_scraped_pdf_links_async(page_urls_to_scan))
        status_updates.append(f"Found {len(pdf_links_found)} unique PDF links.")
        if pdf_links_found:
            status_updates.append(f"Starting PDF download and processing for {len(pdf_links_found)} links...")
            processing_results = asyncio.run(batch_download_and_process_pdfs(pdf_links_found, selected_db_val))
            success_count = 0; processed_files_messages = []
            for msg, success, fname in processing_results:
                processed_files_messages.append(f"{fname}: {msg} ({'Success' if success else 'Failed'})")
                if success: success_count += 1; any_successful_scrape_process = True
            status_updates.append(f"\n--- PDF Processing Results ---"); status_updates.extend(processed_files_messages)
            status_updates.append(f"\nScraping finished. Processed {success_count} new PDFs out of {len(pdf_links_found)} found.")
    else: status_updates.append("No site pages found to scan based on current settings.")
    if any_successful_scrape_process: save_data_to_selected_db(selected_db_val)
    cb_update, md_update = _build_file_list_updates()
    return "\n".join(status_updates), cb_update, md_update


# --- Backend Initialization ---
def initialize_all_components(default_db="MongoDB"):
    global gemini_model_genai, together_client, openai_client, embedding_model, faiss_index, BACKEND_INITIAL_LOAD_MSG, mongo_client_instance, mongo_db_obj, auth_mongo_db_obj

    print("Initializing backend components...")
    
    if mongo_client_instance is None:
        initialize_mongodb_client()

    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
        try:
            gemini_model_genai = genai.GenerativeModel("gemini-1.5-flash-latest") 
            print("Gemini client configured with gemini-1.5-flash-latest.")
        except Exception as e:
            print(f"Failed to initialize Gemini client with 'gemini-1.5-flash-latest': {e}. Trying 'gemini-pro'.")
            try:
                gemini_model_genai = genai.GenerativeModel("gemini-pro")
                print("Gemini client configured with gemini-pro.")
            except Exception as e_pro:
                 print(f"Failed to initialize Gemini client with 'gemini-pro': {e_pro}. Gemini features may be affected.")
    else: print("Warning: GOOGLE_API_KEY not found. Gemini features will be disabled.")

    if TOGETHER_API_KEY:
        together_client = Together(api_key=TOGETHER_API_KEY)
        print("TogetherAI client configured.")
    else: print("Warning: TOGETHER_API_KEY not found. Together AI features will be disabled.")

    if OPENAI_API_KEY:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        print("OpenAI client configured.")
    else: print("Warning: OPENAI_API_KEY not found. OpenAI features will be disabled.")

    try:
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("SentenceTransformer model loaded.")
    except Exception as e:
        print(f"Error loading SentenceTransformer model: {e}. Backend cannot function fully.")
        BACKEND_INITIAL_LOAD_MSG = "Critical: Embedding model failed. Backend non-functional."
        return # Cannot proceed without embedding model

    initialize_dropbox_client()

    if embedding_model:
        BACKEND_INITIAL_LOAD_MSG = load_data_from_selected_db(default_db)
    else: # Should not happen due to return above, but as a safeguard
        BACKEND_INITIAL_LOAD_MSG = "Critical component (embedding model) failed. Data loading skipped."
    
    print(f"Backend Initial Load Status: {BACKEND_INITIAL_LOAD_MSG}")
    print("Backend initialization complete.")



def get_backend_initial_load_message():
    global BACKEND_INITIAL_LOAD_MSG
    return BACKEND_INITIAL_LOAD_MSG