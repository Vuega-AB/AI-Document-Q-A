import base64
import io
import os
import requests
from bs4 import BeautifulSoup
import aiohttp
import asyncio
import PyPDF2
import faiss
import time
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from dotenv import load_dotenv
from io import BytesIO
from together import Together
import re
from pymongo import MongoClient, server_api
import logging # Keep for general logging if needed elsewhere
from openai import OpenAI
import sys
import httpx
from urllib.parse import urljoin, unquote
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
import dropbox
import hashlib

# --- Environment Variables & Constants ---
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGO_URI = os.getenv("MongoDB")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DROPBOX_REFRESH_TOKEN = os.getenv("DROPBOX_REFRESH_TOKEN")
DROPBOX_APP_KEY = os.getenv("DROPBOX_APP_KEY")
DROPBOX_APP_SECRET = os.getenv("DROPBOX_APP_SECRET")

INDEX_FILE_DROPBOX = "/faiss_index.index"
TEXT_FILE_DROPBOX = "/text_store.json"
TOKEN_FILE = "dropbox_token.json" # Local file to store short-lived access tokens
MONGO_DB_NAME = "IntelLawDB_Dash"
FAISS_COLLECTION_NAME = "faiss_index_store"
TEXT_STORE_COLLECTION_NAME = "text_content_store"

# --- Global Backend State ---
gemini_model_genai = None
together_client = None
openai_client = None
dbx = None
mongo_client_instance = None
mongo_db_obj = None

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
faiss_index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
text_store = [] # List of dicts: {"text": str, "file_name": str, "file_hash": str}

AVAILABLE_MODELS_DICT = {
    "gemini-2.0-flash": {"price": "Custom", "type": "gemini", "name": "Gemini 2.0 Flash"},
    "openai-4o": {"price": "Custom", "type": "openai", "name": "OpenAI GPT-4o"},
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": {"price": "$0.88", "type": "together", "name": "Llama3.3 70B Turbo"},
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": {"price": "$3.50", "type": "together", "name": "Llama3.1 405B Turbo"},
    "microsoft/WizardLM-2-8x22B": {"price": "$1.20", "type": "together", "name": "WizardLM-2 8x22B"},
    "mistralai/Mixtral-8x22B-Instruct-v0.1": {"price": "$1.20", "type": "together", "name": "Mixtral 8x22B Instruct"},
    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO": {"price": "$0.60", "type": "together", "name": "Hermes-2 Mixtral DPO"},
}
AVAILABLE_MODELS_OPTIONS = [{'label': details['name'], 'value': model_id} for model_id, details in AVAILABLE_MODELS_DICT.items()]

# --- API Client Initializations ---
def initialize_api_clients_backend():
    global gemini_model_genai, together_client, openai_client
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
        gemini_model_genai = genai.GenerativeModel("gemini-2.0-flash") # Corrected model name if necessary
    else:
        gemini_model_genai = None
        print("Warning: GOOGLE_API_KEY not found. Gemini features will be disabled.")

    if TOGETHER_API_KEY:
        together_client = Together(api_key=TOGETHER_API_KEY)
    else:
        together_client = None
        print("Warning: TOGETHER_API_KEY not found. Together AI features will be disabled.")

    if OPENAI_API_KEY:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    else:
        openai_client = None
        print("Warning: OPENAI_API_KEY not found. OpenAI features will be disabled.")

# --- Dropbox Functions ---
def _load_access_token():
    if os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE, "r") as file: data = json.load(file); return data.get("access_token"), data.get("expires_at")
        except json.JSONDecodeError:
            print(f"Warning: Could not decode {TOKEN_FILE}. Will attempt to refresh token.")
            return None, None
    return None, None

def _save_access_token(access_token, expires_in):
    expires_at = int(time.time()) + expires_in - 300 # 5 min buffer
    with open(TOKEN_FILE, "w") as file: json.dump({"access_token": access_token, "expires_at": expires_at}, file)

def _get_dropbox_access_token():
    if not (DROPBOX_APP_KEY and DROPBOX_APP_SECRET and DROPBOX_REFRESH_TOKEN):
        raise Exception("Dropbox credentials (APP_KEY, APP_SECRET, REFRESH_TOKEN) not fully configured.")
    response = requests.post(
        "https://api.dropbox.com/oauth2/token",
        data={"grant_type": "refresh_token", "refresh_token": DROPBOX_REFRESH_TOKEN},
        auth=(DROPBOX_APP_KEY, DROPBOX_APP_SECRET)
    )
    if response.status_code == 200:
        data = response.json()
        _save_access_token(data["access_token"], data.get("expires_in", 14400)) # Default 4 hours
        return data["access_token"]
    else:
        raise Exception(f"Failed to refresh Dropbox token: {response.status_code} - {response.text}")

def _get_valid_access_token():
    access_token, expires_at = _load_access_token()
    if access_token and expires_at and int(time.time()) < expires_at:
        return access_token
    return _get_dropbox_access_token()

def initialize_dropbox_client_backend():
    global dbx
    if not (DROPBOX_REFRESH_TOKEN and DROPBOX_APP_KEY and DROPBOX_APP_SECRET):
        print("Warning: Dropbox credentials not fully set. Dropbox features disabled.")
        dbx = None
        return
    try:
        access_token = _get_valid_access_token()
        dbx = dropbox.Dropbox(access_token)
        dbx.users_get_current_account() # Test connection
        print("Dropbox client initialized successfully.")
    except Exception as e:
        print(f"Error connecting to Dropbox: {e}")
        dbx = None

def get_is_dropbox_configured_backend():
    return bool(DROPBOX_REFRESH_TOKEN and DROPBOX_APP_KEY and DROPBOX_APP_SECRET)

# --- MongoDB Functions ---
def initialize_mongodb_client_backend():
    global mongo_client_instance, mongo_db_obj
    if MONGO_URI and mongo_client_instance is None:
        try:
            mongo_client_instance = MongoClient(MONGO_URI, server_api=server_api.ServerApi('1'))
            mongo_client_instance.admin.command('ping') # Verify connection
            mongo_db_obj = mongo_client_instance[MONGO_DB_NAME]
            print("MongoDB client initialized successfully.")
        except Exception as e:
            print(f"MongoDB connection failed: {e}")
            mongo_client_instance = None
            mongo_db_obj = None
    elif not MONGO_URI:
        print("Warning: MONGO_URI not set. MongoDB features disabled.")
        mongo_client_instance = None
        mongo_db_obj = None

def get_mongo_uri_status_backend():
    return bool(MONGO_URI)

# --- Data Persistence Functions ---
def save_data_to_db(selected_db):
    global faiss_index, text_store, dbx, mongo_db_obj
    status = ""
    if selected_db == "Dropbox":
        if dbx is None: status = "Dropbox not initialized for saving."; print(status); return status
        try:
            temp_idx_file = "temp_faiss_to_dropbox.index"
            faiss.write_index(faiss_index, temp_idx_file)
            with open(temp_idx_file, "rb") as f:
                dbx.files_upload(f.read(), INDEX_FILE_DROPBOX, mode=dropbox.files.WriteMode.overwrite)
            os.remove(temp_idx_file)

            text_json = json.dumps(text_store, ensure_ascii=False, indent=4).encode('utf-8')
            dbx.files_upload(text_json, TEXT_FILE_DROPBOX, mode=dropbox.files.WriteMode.overwrite)
            status = f"Data saved to Dropbox. Index size: {faiss_index.ntotal}, Text items: {len(text_store)}"
            print(status)
        except Exception as e: status = f"Error saving to Dropbox: {e}"; print(status)
    elif selected_db == "MongoDB":
        if mongo_db_obj is None: status = "MongoDB not initialized for saving."; print(status); return status
        try:
            temp_idx_file = "temp_faiss_to_mongo.idx"
            faiss.write_index(faiss_index, temp_idx_file)
            with open(temp_idx_file, "rb") as f: index_bytes = f.read()
            os.remove(temp_idx_file)

            mongo_db_obj[FAISS_COLLECTION_NAME].update_one({"_id": "main_faiss_index"}, {"$set": {"index_data": index_bytes}}, upsert=True)
            mongo_db_obj[TEXT_STORE_COLLECTION_NAME].delete_many({}) # Clear before inserting
            if text_store: mongo_db_obj[TEXT_STORE_COLLECTION_NAME].insert_many(text_store)
            status = f"Data saved to MongoDB. Index size: {faiss_index.ntotal}, Text items: {len(text_store)}"
            print(status)
        except Exception as e: status = f"Error saving to MongoDB: {e}"; print(status)
    else:
        status = f"Unknown database type '{selected_db}' for saving."
        print(status)
    return status

def load_data_from_db(selected_db):
    global faiss_index, text_store, embedding_model, dbx, mongo_db_obj
    # Reset local stores before loading
    faiss_index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
    text_store = []
    alert_msg = ""

    if selected_db == "Dropbox":
        if dbx is None: initialize_dropbox_client_backend() # Attempt to re-initialize if not already
        if dbx is None: alert_msg = "Dropbox not initialized. Cannot load data."; print(alert_msg); return alert_msg
        try:
            _, res_index = dbx.files_download(path=INDEX_FILE_DROPBOX)
            temp_idx_file = "temp_faiss_from_dropbox.index"
            with open(temp_idx_file, "wb") as f: f.write(res_index.content)
            faiss_index = faiss.read_index(temp_idx_file)
            os.remove(temp_idx_file)

            _, res_text = dbx.files_download(path=TEXT_FILE_DROPBOX)
            text_store = json.loads(res_text.content.decode('utf-8'))
            alert_msg = f"Data loaded from Dropbox. {len(text_store)} text items, index has {faiss_index.ntotal} vectors."
        except dropbox.exceptions.ApiError as e:
            if isinstance(e.error, dropbox.files.DownloadError) and e.error.is_path() and e.error.get_path().is_not_found():
                alert_msg = "No existing data on Dropbox. Initialized empty store."
            else: alert_msg = f"Dropbox API error: {e}"
        except Exception as e: alert_msg = f"Error loading from Dropbox: {e}"

    elif selected_db == "MongoDB":
        if mongo_db_obj is None: initialize_mongodb_client_backend() # Attempt to re-initialize
        if mongo_db_obj is None: alert_msg = "MongoDB not initialized. Cannot load data."; print(alert_msg); return alert_msg
        try:
            index_doc = mongo_db_obj[FAISS_COLLECTION_NAME].find_one({"_id": "main_faiss_index"})
            if index_doc and "index_data" in index_doc:
                temp_idx_file = "temp_faiss_from_mongo.idx"
                with open(temp_idx_file, "wb") as f: f.write(index_doc["index_data"])
                faiss_index = faiss.read_index(temp_idx_file)
                os.remove(temp_idx_file)

            text_docs = mongo_db_obj[TEXT_STORE_COLLECTION_NAME].find({})
            text_store = [{k: v for k, v in doc.items() if k != '_id'} for doc in text_docs] # Exclude Mongo's _id
            alert_msg = f"Data loaded from MongoDB. {len(text_store)} text items, index has {faiss_index.ntotal} vectors."
        except Exception as e: alert_msg = f"Error loading from MongoDB: {e}"
    else:
        alert_msg = f"Unknown database type '{selected_db}' for loading."

    print(alert_msg) # Print status to console
    return alert_msg

# --- PDF Processing ---
def _chunk_text(text, chunk_size=400, min_chunk_length=20):
    paragraphs = re.split(r'\n{2,}', text)
    chunks = []
    for para in paragraphs:
        sentences = re.split(r'(?<=[.!?])\s+', para) # Split by sentence-ending punctuation
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 < chunk_size: # +1 for space
                current_chunk += sentence + " "
            else:
                cleaned_chunk = current_chunk.strip()
                if len(cleaned_chunk) >= min_chunk_length:
                    chunks.append(cleaned_chunk)
                current_chunk = sentence + " " # Start new chunk with current sentence
        # Add the last chunk if it's not empty
        cleaned_last_chunk = current_chunk.strip()
        if len(cleaned_last_chunk) >= min_chunk_length:
            chunks.append(cleaned_last_chunk)
    return chunks

def _extract_text_from_pdf_bytes(pdf_bytes):
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return ""

def process_uploaded_pdf_backend(pdf_bytes, file_name, selected_db):
    global text_store, faiss_index, embedding_model
    file_hash = hashlib.md5(pdf_bytes).hexdigest()

    if any(item['file_hash'] == file_hash for item in text_store):
        return f"File '{file_name}' (hash: {file_hash[:7]}) already exists.", False

    raw_text = _extract_text_from_pdf_bytes(pdf_bytes)
    if not raw_text.strip():
        return f"No text could be extracted from '{file_name}'.", False

    chunks = _chunk_text(raw_text)
    if not chunks:
        return f"Could not break '{file_name}' into processable chunks (text might be too short or unchunkable).", False

    try:
        embeddings = embedding_model.encode(chunks)
        embeddings_np = np.array(embeddings).astype("float32")
        if embeddings_np.shape[0] > 0:
            faiss_index.add(embeddings_np)
    except Exception as e:
        return f"Error generating or adding embeddings for '{file_name}': {e}", False

    for chunk_text_content in chunks:
        text_store.append({"text": chunk_text_content, "file_name": file_name, "file_hash": file_hash})

    save_data_to_db(selected_db) # Save after successful processing
    return f"Processed and added '{file_name}'. Chunks: {len(chunks)}, Index size now: {faiss_index.ntotal}", True

def delete_file_from_store_backend(file_hash_to_delete, selected_db):
    global text_store, faiss_index, embedding_model
    
    original_text_store_len = len(text_store)
    text_store_before_delete = [item for item in text_store if item["file_hash"] == file_hash_to_delete]
    text_store = [item for item in text_store if item["file_hash"] != file_hash_to_delete]

    if len(text_store) == original_text_store_len and not text_store_before_delete: # No items with this hash existed
        return f"File hash {file_hash_to_delete} not found for deletion.", False

    # Rebuild FAISS index
    faiss_index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
    if text_store:
        all_texts_for_reindex = [item["text"] for item in text_store]
        if all_texts_for_reindex:
            try:
                embeddings = embedding_model.encode(all_texts_for_reindex)
                embeddings_np = np.array(embeddings).astype("float32")
                if embeddings_np.shape[0] > 0:
                    faiss_index.add(embeddings_np)
            except Exception as e:
                # Attempt to restore text_store if re-indexing fails, though this is complex
                # For now, log error and proceed with inconsistent state (or halt)
                return f"Error re-indexing after deletion: {e}. Data might be inconsistent.", False

    save_data_to_db(selected_db)
    deleted_file_name = text_store_before_delete[0]['file_name'] if text_store_before_delete else "Unknown file"
    return f"File '{deleted_file_name}' (hash: {file_hash_to_delete[:7]}) and its data deleted. Index rebuilt.", True


def get_stored_files_list_backend():
    global text_store
    unique_files = {} # {hash: name}
    if text_store:
        for item in text_store:
            if item["file_hash"] not in unique_files:
                unique_files[item["file_hash"]] = item["file_name"]
    return [{"hash": f_hash, "name": f_name} for f_hash, f_name in unique_files.items()]

# --- AI Response Generation ---
def _generate_response_gemini(prompt, context, temp, top_p, system_prompt):
    if not gemini_model_genai: return "Gemini client not initialized (API key missing)."
    input_parts = [system_prompt + "\nContext:\n" + context, "Question:\n" + prompt]
    config = genai.GenerationConfig(max_output_tokens=2048, temperature=temp, top_p=top_p)
    try:
        response = gemini_model_genai.generate_content(input_parts, generation_config=config)
        return response.text
    except Exception as e: return f"Gemini Error: {e}"

def _generate_response_together_ai(prompt, context, model_id, temp, top_p, system_prompt):
    if not together_client: return "TogetherAI client not initialized (API key missing)."
    try:
        response = together_client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{prompt}"}
            ],
            temperature=temp,
            top_p=top_p
        )
        return response.choices[0].message.content.strip()
    except Exception as e: return f"TogetherAI Error ({model_id}): {e}"

def _generate_response_openai_api(prompt, context, temp, top_p, system_prompt):
    if not openai_client: return "OpenAI client not initialized (API key missing)."
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o", # Or make configurable if needed
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{prompt}"}
            ],
            temperature=temp,
            top_p=top_p
        )
        return response.choices[0].message.content
    except Exception as e: return f"OpenAI Error: {e}"

# --- RAG ---
def _retrieve_context_from_db(query, top_k=5):
    global text_store, faiss_index, embedding_model
    if faiss_index.ntotal == 0: return "No documents currently indexed. Cannot retrieve context."
    try:
        query_embedding = embedding_model.encode([query])
        distances, indices = faiss_index.search(query_embedding.astype(np.float32), top_k)
        
        valid_indices = [i for i in indices[0] if 0 <= i < len(text_store)]
        retrieved_texts = [text_store[idx]["text"] for idx in valid_indices]
        
        return "\n\n---\n\n".join(retrieved_texts) if retrieved_texts else "No relevant context found in the indexed documents for your query."
    except Exception as e:
        return f"Error during context retrieval: {e}"


def generate_chat_responses_backend(user_input, app_config, selected_db_for_saving_new_pdfs):
    global text_store, faiss_index # Accessed by retrieve_context_from_db
    
    context_text = _retrieve_context_from_db(user_input)
    
    model_responses_data = []
    temp_config = app_config['temperature']
    top_p_config = app_config['top_p']
    system_prompt_config = app_config['system_prompt']

    temp_values_to_run = [temp_config]
    if app_config['vary_temperature'] and temp_config > 0.01: # Avoid too many calls for 0 temp
        temp_values_to_run = sorted(list(set([
            round(max(0.01, temp_config * 0.5), 2), 
            temp_config, 
            round(min(1.0, temp_config * 1.5), 2) if temp_config * 1.5 <=1.0 else temp_config
        ])))

    top_p_values_to_run = [top_p_config]
    if app_config['vary_top_p'] and top_p_config > 0.01:
        top_p_values_to_run = sorted(list(set([
            round(max(0.01, top_p_config * 0.5), 2),
            top_p_config,
            round(min(1.0, top_p_config * 1.5),2) if top_p_config * 1.5 <= 1.0 else top_p_config
        ])))
        
    selected_ai_models = app_config.get('selected_models', [])
    if not selected_ai_models:
        return [{"model_name": "System", "temp": "-", "top_p": "-", "response": "No AI model selected in configuration."}]

    for model_id in selected_ai_models:
        model_detail = AVAILABLE_MODELS_DICT.get(model_id, {})
        model_type = model_detail.get("type")
        model_display_name = model_detail.get("name", model_id)

        for temp_val in temp_values_to_run:
            for top_p_val in top_p_values_to_run:
                response_content = f"Error: Could not generate response for {model_display_name}." # Default
                if model_type == "gemini":
                    response_content = _generate_response_gemini(user_input, context_text, temp_val, top_p_val, system_prompt_config)
                elif model_type == "together":
                    response_content = _generate_response_together_ai(user_input, context_text, model_id, temp_val, top_p_val, system_prompt_config)
                elif model_type == "openai":
                    response_content = _generate_response_openai_api(user_input, context_text, temp_val, top_p_val, system_prompt_config)
                else:
                    response_content = f"Model type '{model_type}' for '{model_id}' not recognized."
                
                model_responses_data.append({
                    "model_name": model_display_name, "temp": temp_val, "top_p": top_p_val,
                    "response": response_content, "model_info_str": f"{model_display_name} (T:{temp_val}, P:{top_p_val})"
                })
                if not app_config['vary_top_p']: break 
            if not app_config['vary_temperature']: break
    return model_responses_data


# --- Web Scraping ---
async def _fetch_page_async(url):
    # Use a timeout and handle potential errors during the request.
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; MyAppScraper/1.0; +http://mycoolapp.com/scraper)"})
            response.raise_for_status() # Raise an exception for bad status codes
            return response.text, str(response.url) # Return final URL after redirects
    except httpx.RequestError as e:
        print(f"HTTPX Request error fetching {url}: {e}")
        return None, str(e.request.url if e.request else url) # Return original or attempted URL on error
    except Exception as e:
        print(f"Generic error fetching {url}: {e}")
        return None, url # Fallback URL


async def _extract_pdf_links_from_url_async(url_to_scrape):
    try:
        html_content, base_url_after_redirects = await _fetch_page_async(url_to_scrape)
        if html_content is None:
            return [] # Fetch failed
        
        soup = BeautifulSoup(html_content, "html.parser")
        pdf_links = set()
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            # Check if href ends with .pdf (case-insensitive) or contains .pdf? (for query params)
            if href.lower().endswith(".pdf") or ".pdf?" in href.lower():
                # Construct absolute URL
                absolute_link = urljoin(base_url_after_redirects, href)
                pdf_links.add(absolute_link)
        return list(pdf_links)
    except Exception as e:
        print(f"Error scraping PDF links from {url_to_scrape}: {e}")
        return []

async def _process_scraped_pdf_links_async(urls_to_scan_for_pdfs):
    all_pdf_links = set()
    # Create tasks for all URLs to be scanned
    tasks = [_extract_pdf_links_from_url_async(u) for u in urls_to_scan_for_pdfs]
    # Gather results; this runs them concurrently
    results_list_of_lists = await asyncio.gather(*tasks, return_exceptions=True)
    
    for result in results_list_of_lists:
        if isinstance(result, Exception):
            print(f"An exception occurred during PDF link extraction: {result}")
        elif result: # result is a list of PDF links
            all_pdf_links.update(result)
    return list(all_pdf_links)


async def _download_and_process_scraped_pdf(session, pdf_link, selected_db_for_saving):
    try:
        # Sanitize filename from URL
        filename_from_url = os.path.basename(unquote(urlparse(pdf_link).path))
        if not filename_from_url.lower().endswith(".pdf"): # Basic check
            filename = f"{filename_from_url if filename_from_url else 'downloaded_pdf'}.pdf"
        else:
            filename = filename_from_url

        async with session.get(pdf_link, timeout=60) as response:
            if response.status == 200:
                pdf_bytes = await response.read()
                # process_uploaded_pdf_backend expects bytes, filename, and selected_db
                status_msg, success = process_uploaded_pdf_backend(pdf_bytes, filename, selected_db_for_saving)
                return status_msg, success, filename
            else:
                return f"Failed to download (status: {response.status})", False, filename
    except asyncio.TimeoutError:
        return f"Timeout downloading/processing", False, os.path.basename(pdf_link)
    except Exception as e:
        return f"Error processing: {e}", False, os.path.basename(pdf_link)

async def _batch_download_and_process_pdfs(pdf_links, selected_db_for_saving):
    results = [] # To store (status_msg, success, filename)
    # Using aiohttp.ClientSession for connection pooling
    async with aiohttp.ClientSession(headers={"User-Agent": "Mozilla/5.0"}) as session:
        tasks = [_download_and_process_scraped_pdf(session, link, selected_db_for_saving) for link in pdf_links]
        # Process tasks, potentially in batches if there are many, to avoid overwhelming resources
        # For simplicity here, asyncio.gather is used for all tasks.
        # Consider a semaphore for large numbers of PDFs: asyncio.Semaphore(10)
        for result in await asyncio.gather(*tasks, return_exceptions=True):
            if isinstance(result, Exception):
                # Handle exceptions from _download_and_process_scraped_pdf itself, if any (should be caught within)
                results.append((f"Unhandled exception during PDF processing task: {result}", False, "Unknown PDF"))
            else:
                results.append(result)
    return results

def _get_page_items_sync(url, base_url_val, listing_endpoint_val):
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        response.raise_for_status() # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
        
        soup = BeautifulSoup(response.text, "html.parser")
        items = set()
        
        # Refine link selection: look for links that are likely to be individual item pages
        # This logic is specific to the example imy.se structure and might need generalization
        for item_tag in soup.find_all("a", href=True):
            link = item_tag.get("href")
            # Ensure link is not None and is a string
            if link and isinstance(link, str):
                # Check if the link seems to be part of the listing and not pagination, RSS, etc.
                # This condition is crucial and highly site-dependent.
                # Example: link contains the listing endpoint, is not the base listing URL itself, and doesn't look like pagination.
                if f"/{listing_endpoint_val}/" in link and link != f"/{listing_endpoint_val}/" and not link.startswith("?page=") and not link.endswith("/rss"):
                    full_link = urljoin(base_url_val, link) # Use base_url_val for resolving relative links
                    if full_link != url: # Avoid adding the page URL itself if it matches this pattern
                         items.add(full_link)
        return list(items)
    except requests.exceptions.RequestException as e:
        print(f"Request error scraping page items from {url}: {e}")
        return []
    except Exception as e:
        print(f"Generic error scraping page items {url}: {e}")
        return []


def _get_all_page_urls_for_scraping(base_url_val, listing_endpoint_val, pagination_format_val, num_pages_val):
    all_page_urls_to_scrape_for_pdfs = set() # Stores URLs of actual content pages
    
    # Ensure num_pages_val is an int
    try:
        num_pages = int(num_pages_val)
        if num_pages < 1: num_pages = 1 # Scrape at least one page
    except ValueError:
        print(f"Warning: Invalid number of pages '{num_pages_val}', defaulting to 1.")
        num_pages = 1

    for page_num in range(1, num_pages + 1):
        # Construct the URL for the listing page (e.g., https://www.imy.se/tillsyner/?query=&page=1)
        # The pagination_format_val might be just "?page=" or a full path segment like "page/"
        if pagination_format_val.startswith("?"): # Query parameter based pagination
            current_listing_page_url = f"{base_url_val.strip('/')}/{listing_endpoint_val.strip('/')}/{pagination_format_val}{page_num}"
        else: # Path segment based pagination
            current_listing_page_url = f"{base_url_val.strip('/')}/{listing_endpoint_val.strip('/')}/{pagination_format_val.strip('/')}{page_num}"
        
        print(f"Scraping listing page: {current_listing_page_url}")
        # Get links to individual content pages from the current listing page
        content_page_links = _get_page_items_sync(current_listing_page_url, base_url_val, listing_endpoint_val)
        
        if not content_page_links:
            print(f"No new content page links found on {current_listing_page_url}, stopping pagination.")
            break # Stop if a page yields no new links (might be end of results)
        
        all_page_urls_to_scrape_for_pdfs.update(content_page_links)
        # Optional: Add a small delay if scraping multiple pages rapidly
        # time.sleep(0.5) 

    return list(all_page_urls_to_scrape_for_pdfs)


def run_web_scraping_and_processing_backend(base_url, endpoint, pagination, num_pages, selected_db):
    status_updates = []
    try:
        # This check is for asyncio on Windows.
        if sys.platform == "win32" and isinstance(asyncio.get_event_loop_policy(), asyncio.WindowsSelectorEventLoopPolicy):
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
        status_updates.append(f"Starting scraping for base URL: {base_url}...")
        
        # 1. Get URLs of pages that might contain PDF links (e.g., individual article/report pages)
        page_urls_to_scan_for_pdfs = _get_all_page_urls_for_scraping(base_url, endpoint, pagination, num_pages)
        status_updates.append(f"Found {len(page_urls_to_scan_for_pdfs)} site pages to scan for PDF links.")
        if not page_urls_to_scan_for_pdfs:
            status_updates.append("No site pages found to scan. Scraping aborted.")
            return status_updates

        # 2. From these pages, extract direct PDF links
        pdf_links_found = asyncio.run(_process_scraped_pdf_links_async(page_urls_to_scan_for_pdfs))
        status_updates.append(f"Found {len(pdf_links_found)} unique PDF links across scanned pages.")
        if not pdf_links_found:
            status_updates.append("No PDF links found on the scanned pages. Nothing to process.")
            return status_updates

        # 3. Download and process these PDFs
        status_updates.append(f"Starting download and processing for {len(pdf_links_found)} PDFs...")
        processing_results = asyncio.run(_batch_download_and_process_pdfs(pdf_links_found, selected_db))
        
        success_count = 0
        for msg, success, fname in processing_results:
            status_updates.append(f"{fname}: {'SUCCESS' if success else 'FAIL'} - {msg}")
            if success: success_count += 1
        
        status_updates.append(f"Scraping finished. Successfully processed {success_count} new PDFs out of {len(pdf_links_found)} found.")
        
    except Exception as e:
        status_updates.append(f"Scraping Pipeline Error: {e}")
        logging.error("Scraping pipeline error", exc_info=True)
    return status_updates

# --- Utility Functions for Frontend ---
def get_available_models_options_backend():
    return AVAILABLE_MODELS_OPTIONS

def get_initial_faiss_index_ntotal_backend():
    return faiss_index.ntotal

def get_initial_text_store_len_backend():
    return len(text_store)

def get_default_db_choice_backend():
    if get_is_dropbox_configured_backend():
        return "Dropbox"
    if get_mongo_uri_status_backend():
        return "MongoDB"
    return "Dropbox" # Fallback, will show warning if not configured

# --- Helper for URL parsing in scraping (if needed more broadly) ---
from urllib.parse import urlparse