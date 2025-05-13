# panel_app.py
import panel as pn
pn.extension(sizing_mode="stretch_width", notifications=True) # Enable notifications

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
from langdetect import detect
import json
from dotenv import load_dotenv
from io import BytesIO
from together import Together
import re
from pymongo import MongoClient, server_api
# import subprocess # Not used in this Panel version for Playwright
import logging
from openai import OpenAI
import sys
import httpx
from urllib.parse import urljoin
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
import dropbox
import hashlib

# --- Environment Variables & Initializations (Same as before) ---
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGO_URI = os.getenv("MongoDB")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DROPBOX_REFRESH_TOKEN = os.getenv("DROPBOX_REFRESH_TOKEN")
DROPBOX_APP_KEY = os.getenv("DROPBOX_APP_KEY")
DROPBOX_APP_SECRET = os.getenv("DROPBOX_APP_SECRET")

CONFIG_FILENAME = "config.json"
INDEX_FILE_DROPBOX = "/faiss_index.index"
TEXT_FILE_DROPBOX = "/text_store.json"
TOKEN_FILE = "dropbox_token.json"
MONGO_DB_NAME = "IntelLawDB_Panel" # Changed DB name
FAISS_COLLECTION_NAME = "faiss_index_store"
TEXT_STORE_COLLECTION_NAME = "text_content_store"

# Initialize API clients
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model_genai = genai.GenerativeModel("gemini-2.0-flash")
else: gemini_model_genai = None; print("Warning: GOOGLE_API_KEY not found.")
if TOGETHER_API_KEY: together_client = Together(api_key=TOGETHER_API_KEY)
else: together_client = None; print("Warning: TOGETHER_API_KEY not found.")
if OPENAI_API_KEY: openai_client = OpenAI(api_key=OPENAI_API_KEY)
else: openai_client = None; print("Warning: OPENAI_API_KEY not found.")

dbx = None
mongo_client_instance = None
mongo_db_obj = None

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
faiss_index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
# text_store will be a reactive list in Panel
text_store_rx = pn.rx.rx_list([])


AVAILABLE_MODELS_DICT = {
    "gemini-2.0-flash": {"price": "Custom", "type": "gemini", "name": "Gemini 2.0 Flash"},
    "openai-4o": {"price": "Custom", "type": "openai", "name": "OpenAI GPT-4o"},
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": {"price": "$0.88", "type": "together", "name": "Llama3.3 70B Turbo"},
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": {"price": "$3.50", "type": "together", "name": "Llama3.1 405B Turbo"},
    # ... (add all your models)
}
# For Panel MultiSelect, options can be a dictionary {name: id} or list of ids if names are ids
AVAILABLE_MODELS_PANEL_OPTIONS = {details['name']: model_id for model_id, details in AVAILABLE_MODELS_DICT.items()}


# --- Backend Functions (Dropbox, MongoDB, PDF, AI, RAG, Scraper - Same as before) ---
# ... (Copy ALL your backend Python functions here, ensure they use the global clients and faiss_index)
# For brevity, I'm omitting them, but they are ESSENTIAL.
# Ensure they modify the global faiss_index and text_store_rx.value where appropriate,
# or return values that Panel callbacks can use to update text_store_rx.

# --- Dropbox Functions ---
def load_access_token():
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "r") as file: data = json.load(file); return data.get("access_token"), data.get("expires_at")
    return None, None
def save_access_token(access_token, expires_in):
    expires_at = int(time.time()) + expires_in - 300
    with open(TOKEN_FILE, "w") as file: json.dump({"access_token": access_token, "expires_at": expires_at}, file)
def get_dropbox_access_token():
    if not (DROPBOX_APP_KEY and DROPBOX_APP_SECRET and DROPBOX_REFRESH_TOKEN):
        raise Exception("Dropbox credentials (APP_KEY, APP_SECRET, REFRESH_TOKEN) not fully configured.")
    response = requests.post("https://api.dropbox.com/oauth2/token", data={"grant_type": "refresh_token", "refresh_token": DROPBOX_REFRESH_TOKEN}, auth=(DROPBOX_APP_KEY, DROPBOX_APP_SECRET))
    if response.status_code == 200: data = response.json(); save_access_token(data["access_token"], data.get("expires_in", 14400)); return data["access_token"]
    else: raise Exception(f"Failed to refresh Dropbox token: {response.text}")
def get_valid_access_token():
    access_token, expires_at = load_access_token()
    if access_token and expires_at and int(time.time()) < expires_at: return access_token
    return get_dropbox_access_token()
def initialize_dropbox_client():
    global dbx
    if not (DROPBOX_REFRESH_TOKEN and DROPBOX_APP_KEY and DROPBOX_APP_SECRET):
        print("Warning: Dropbox credentials not fully set. Dropbox features disabled.")
        dbx = None; return
    try: access_token = get_valid_access_token(); dbx = dropbox.Dropbox(access_token)
    except Exception as e: print(f"Error connecting to Dropbox: {e}"); dbx = None

# --- MongoDB Functions ---
def initialize_mongodb_client():
    global mongo_client_instance, mongo_db_obj
    if MONGO_URI and mongo_client_instance is None:
        try:
            mongo_client_instance = MongoClient(MONGO_URI, server_api=server_api.ServerApi('1'))
            mongo_client_instance.admin.command('ping')
            mongo_db_obj = mongo_client_instance[MONGO_DB_NAME]
        except Exception as e:
            print(f"MongoDB connection failed: {e}")
            mongo_client_instance = None; mongo_db_obj = None
    elif not MONGO_URI:
        print("Warning: MONGO_URI not set. MongoDB features disabled.")
        mongo_client_instance = None; mongo_db_obj = None

# --- Data Persistence Functions ---
def save_data_to_selected_db(selected_db_val): # Renamed arg to avoid conflict
    global faiss_index, text_store_rx, dbx, mongo_db_obj # Use text_store_rx.value
    current_text_store = text_store_rx.value

    # Ensure clients are initialized
    if selected_db_val == "Dropbox" and dbx is None: initialize_dropbox_client()
    if selected_db_val == "MongoDB" and mongo_db_obj is None: initialize_mongodb_client()

    if selected_db_val == "Dropbox":
        if dbx is None: print("Dropbox not initialized for saving."); return
        try:
            temp_idx_file = "temp_faiss_to_dropbox.index"
            faiss.write_index(faiss_index, temp_idx_file)
            with open(temp_idx_file, "rb") as f: dbx.files_upload(f.read(), INDEX_FILE_DROPBOX, mode=dropbox.files.WriteMode.overwrite)
            os.remove(temp_idx_file)
            text_json_data = json.dumps(current_text_store, ensure_ascii=False, indent=4).encode('utf-8')
            dbx.files_upload(text_json_data, TEXT_FILE_DROPBOX, mode=dropbox.files.WriteMode.overwrite)
            print(f"Data saved to Dropbox. Index size: {faiss_index.ntotal}, Text items: {len(current_text_store)}")
        except Exception as e: print(f"Error saving to Dropbox: {e}")
    elif selected_db_val == "MongoDB":
        if mongo_db_obj is None: print("MongoDB not initialized for saving."); return
        try:
            temp_idx_file = "temp_faiss_to_mongo.idx"
            faiss.write_index(faiss_index, temp_idx_file)
            with open(temp_idx_file, "rb") as f: index_bytes = f.read()
            os.remove(temp_idx_file)
            mongo_db_obj[FAISS_COLLECTION_NAME].update_one({"_id": "main_faiss_index"}, {"$set": {"index_data": index_bytes}}, upsert=True)
            mongo_db_obj[TEXT_STORE_COLLECTION_NAME].delete_many({})
            if current_text_store: mongo_db_obj[TEXT_STORE_COLLECTION_NAME].insert_many(current_text_store)
            print(f"Data saved to MongoDB. Index size: {faiss_index.ntotal}, Text items: {len(current_text_store)}")
        except Exception as e: print(f"Error saving to MongoDB: {e}")

def load_data_from_selected_db(selected_db_val): # Renamed arg
    global faiss_index, text_store_rx, embedding_model, dbx, mongo_db_obj
    faiss_index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
    new_text_store = [] # Local variable to build the new store
    alert_msg = ""

    if selected_db_val == "Dropbox":
        if dbx is None: initialize_dropbox_client()
        if dbx is None: alert_msg = "Dropbox not initialized. Cannot load."; text_store_rx.value = []; return alert_msg
        try:
            _, res_index = dbx.files_download(path=INDEX_FILE_DROPBOX)
            temp_idx_file = "temp_faiss_from_dropbox.index"
            with open(temp_idx_file, "wb") as f: f.write(res_index.content)
            faiss_index = faiss.read_index(temp_idx_file)
            os.remove(temp_idx_file)
            _, res_text = dbx.files_download(path=TEXT_FILE_DROPBOX)
            new_text_store = json.loads(res_text.content.decode('utf-8'))
            alert_msg = f"Data loaded from Dropbox. {len(new_text_store)} text items, index has {faiss_index.ntotal} vectors."
        except dropbox.exceptions.ApiError as e:
            if isinstance(e.error, dropbox.files.DownloadError) and e.error.is_path() and e.error.get_path().is_not_found():
                alert_msg = "No existing data on Dropbox. Initialized empty store."
            else: alert_msg = f"Dropbox API error: {e}"
        except Exception as e: alert_msg = f"Error loading from Dropbox: {e}"
    elif selected_db_val == "MongoDB":
        if mongo_db_obj is None: initialize_mongodb_client()
        if mongo_db_obj is None: alert_msg = "MongoDB not initialized. Cannot load."; text_store_rx.value = []; return alert_msg
        try:
            index_doc = mongo_db_obj[FAISS_COLLECTION_NAME].find_one({"_id": "main_faiss_index"})
            if index_doc and "index_data" in index_doc:
                temp_idx_file = "temp_faiss_from_mongo.idx"
                with open(temp_idx_file, "wb") as f: f.write(index_doc["index_data"])
                faiss_index = faiss.read_index(temp_idx_file)
                os.remove(temp_idx_file)
            text_docs = mongo_db_obj[TEXT_STORE_COLLECTION_NAME].find({})
            new_text_store = [{k: v for k, v in doc.items() if k != '_id'} for doc in text_docs]
            alert_msg = f"Data loaded from MongoDB. {len(new_text_store)} text items, index has {faiss_index.ntotal} vectors."
        except Exception as e: alert_msg = f"Error loading from MongoDB: {e}"
    
    text_store_rx.value = new_text_store # Update the reactive list
    print(f"Load attempt for {selected_db_val}: {alert_msg}")
    return alert_msg

# --- PDF Processing ---
def chunk_text(text, chunk_size=400, min_chunk_length=20):
    # ... (same as before)
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
    # ... (same as before)
    reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes)); text = ""
    for page in reader.pages: page_text = page.extract_text(); text += page_text + "\n" if page_text else ""
    return text

def process_and_add_pdf(pdf_bytes_val, file_name_val, selected_db_val): # Renamed args
    global faiss_index, embedding_model, text_store_rx # Use reactive list
    current_text_store = text_store_rx.value.copy() # Work with a copy

    file_hash = hashlib.md5(pdf_bytes_val).hexdigest()
    if any(item['file_hash'] == file_hash for item in current_text_store):
        return f"File '{file_name_val}' (hash: {file_hash[:7]}) already exists.", False
    
    raw_text = extract_text_from_pdf_bytes(pdf_bytes_val)
    if not raw_text.strip(): return f"No text extracted from '{file_name_val}'.", False
        
    chunks = chunk_text(raw_text)
    if not chunks: return f"Could not chunk '{file_name_val}'.", False

    embeddings = embedding_model.encode(chunks)
    embeddings_np = np.array(embeddings).astype("float32")
    if embeddings_np.shape[0] > 0: faiss_index.add(embeddings_np) # faiss_index is global
    
    new_items_for_store = []
    for chunk_text_content in chunks:
        new_items_for_store.append({"text": chunk_text_content, "file_name": file_name_val, "file_hash": file_hash})
    
    current_text_store.extend(new_items_for_store)
    text_store_rx.value = current_text_store # Update reactive list

    save_data_to_selected_db(selected_db_val)
    return f"Processed '{file_name_val}'. Chunks: {len(chunks)}, Index: {faiss_index.ntotal}", True

# --- AI Response Generation (same as before, ensure they use global clients) ---
def generate_response_gemini(prompt, context, temp, top_p, system_prompt):
    if not gemini_model_genai: return "Gemini client not initialized (API key missing)."
    input_parts = [system_prompt + "\nContext: " + context, "Question: " + prompt]
    config = genai.GenerationConfig(max_output_tokens=2048, temperature=temp, top_p=top_p)
    try: response = gemini_model_genai.generate_content(input_parts, generation_config=config); return response.text
    except Exception as e: return f"Gemini Error: {e}"
def generate_response_together_ai(prompt, context, model_id, temp, top_p, system_prompt):
    if not together_client: return "TogetherAI client not initialized (API key missing)."
    try:
        response = together_client.chat.completions.create(
            model=model_id, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Context: {context}\nQuestion: {prompt}"}],
            temperature=temp, top_p=top_p
        )
        return response.choices[0].message.content.strip()
    except Exception as e: return f"TogetherAI Error ({model_id}): {e}"
def generate_response_openai_api(prompt, context, temp, top_p, system_prompt):
    if not openai_client: return "OpenAI client not initialized (API key missing)."
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Context: {context}\nQuestion: {prompt}"}],
            temperature=temp, top_p=top_p
        )
        return response.choices[0].message.content
    except Exception as e: return f"OpenAI Error: {e}"

# --- RAG ---
def retrieve_context_from_db(query, top_k=5):
    global faiss_index, embedding_model, text_store_rx # Use reactive list
    current_text_store = text_store_rx.value
    if faiss_index.ntotal == 0: return "No documents indexed."
    query_embedding = embedding_model.encode([query])
    distances, indices = faiss_index.search(query_embedding.astype(np.float32), top_k)
    valid_indices = [i for i in indices[0] if 0 <= i < len(current_text_store)]
    retrieved_texts = [current_text_store[idx]["text"] for idx in valid_indices]
    return "\n\n".join(retrieved_texts) if retrieved_texts else "No relevant context found."

# --- Web Scraping (same as before) ---
async def fetch_page_async(url): # ...
    async with httpx.AsyncClient() as client: response = await client.get(url, timeout=30.0); return response.text, str(response.url)
async def extract_pdf_links_from_url_async(url): # ...
    try: html, base_url = await fetch_page_async(url); soup = BeautifulSoup(html, "html.parser"); return [urljoin(base_url, a["href"]) for a in soup.find_all("a", href=True) if ".pdf" in a["href"].lower()]
    except Exception as e: print(f"Error scraping {url}: {e}"); return []
async def process_scraped_pdf_links_async(urls): # ...
    all_pdf_links = set()
    for url_group in await asyncio.gather(*[extract_pdf_links_from_url_async(u) for u in urls]): all_pdf_links.update(url_group)
    return list(all_pdf_links)
async def download_and_process_scraped_pdf(session, pdf_link, selected_db_val): # ...
    try:
        async with session.get(pdf_link, timeout=60) as response:
            if response.status == 200:
                pdf_bytes = await response.read()
                filename = os.path.basename(pdf_link)
                status_msg, success = process_and_add_pdf(pdf_bytes, filename, selected_db_val) # Uses global text_store_rx
                return status_msg, success, filename
            return f"Failed to download {pdf_link} (status: {response.status})", False, os.path.basename(pdf_link)
    except Exception as e: return f"Error processing {pdf_link}: {e}", False, os.path.basename(pdf_link)
async def batch_download_and_process_pdfs(pdf_links, selected_db_val): # ...
    results = []
    async with aiohttp.ClientSession() as session:
        tasks = [download_and_process_scraped_pdf(session, link, selected_db_val) for link in pdf_links]
        for result in await asyncio.gather(*tasks): results.append(result)
    return results
def get_page_items_sync(url, base_url_val, listing_endpoint_val): # ...
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        if response.status_code != 200: return []
        soup = BeautifulSoup(response.text, "html.parser"); items = set()
        for item in soup.find_all("a"):
            link = item.get("href")
            if link and f"/{listing_endpoint_val}/" in link and link != url and not link.endswith("/rss"):
                if not link.startswith("http"): link = urljoin(base_url_val, link)
                items.add(link)
        return list(items)
    except Exception as e: print(f"Error scraping page items {url}: {e}"); return []
def get_all_page_urls_for_scraping(base_url_val, listing_endpoint_val, pagination_format_val, num_pages_val): # ...
    all_page_urls_to_scrape = set()
    for page_num in range(1, int(num_pages_val) + 1):
        url = f"{base_url_val}/{listing_endpoint_val}/{pagination_format_val}{page_num}"
        page_items = get_page_items_sync(url, base_url_val, listing_endpoint_val)
        if not page_items: break
        all_page_urls_to_scrape.update(page_items)
    return list(all_page_urls_to_scrape)

# --- Initial Data Load (called once at the start of script execution) ---
initialize_dropbox_client()
initialize_mongodb_client()
initial_load_status_msg = load_data_from_selected_db("Dropbox") # Default to Dropbox
print(f"Panel App Initial Load: {initial_load_status_msg}")


# --- Panel Application State and UI Components ---

# Application Configuration (using pn.rx for reactivity)
app_config = {
    'selected_models': pn.rx.rx_list([list(AVAILABLE_MODELS_PANEL_OPTIONS.values())[0]] if AVAILABLE_MODELS_PANEL_OPTIONS else []),
    'vary_temperature': pn.rx.rx_bool(True),
    'temperature': pn.rx.rx_float(0.7),
    'vary_top_p': pn.rx.rx_bool(False),
    'top_p': pn.rx.rx_float(0.9),
    'system_prompt': pn.rx.rx_str("You are a helpful assistant...")
}

# Selected Database
selected_db_rx = pn.rx.rx_str("Dropbox")

# Chat History
chat_history_rx = pn.rx.rx_list([]) # List of dicts: {"sender": "User"|"AI", "message": "...", "model_info": "..."}


# --- UI Widget Definitions ---
# Config Tab
model_selector_pn = pn.widgets.MultiSelect(
    name="AI Models (Max 3)", options=AVAILABLE_MODELS_PANEL_OPTIONS,
    value=app_config['selected_models'].value # Initial value
)
vary_temp_cb_pn = pn.widgets.Checkbox(name="Vary Temperature", value=app_config['vary_temperature'].value)
temp_slider_pn = pn.widgets.FloatSlider(
    name="Temperature", start=0, end=1, step=0.05, value=app_config['temperature'].value
)
vary_top_p_cb_pn = pn.widgets.Checkbox(name="Vary Top-P", value=app_config['vary_top_p'].value)
top_p_slider_pn = pn.widgets.FloatSlider(
    name="Top-P", start=0, end=1, step=0.05, value=app_config['top_p'].value
)
system_prompt_pn = pn.widgets.TextAreaInput(
    name="System Prompt", value=app_config['system_prompt'].value, height=100,
    placeholder="Enter system prompt..."
)

# Link config widgets to app_config reactive variables
model_selector_pn.param.watch(lambda event: app_config['selected_models'].update(event.new[:3] if len(event.new)>3 else event.new), 'value')
vary_temp_cb_pn.param.watch(lambda event: app_config['vary_temperature'].update(event.new), 'value')
temp_slider_pn.param.watch(lambda event: app_config['temperature'].update(event.new), 'value')
vary_top_p_cb_pn.param.watch(lambda event: app_config['vary_top_p'].update(event.new), 'value')
top_p_slider_pn.param.watch(lambda event: app_config['top_p'].update(event.new), 'value')
system_prompt_pn.param.watch(lambda event: app_config['system_prompt'].update(event.new), 'value')


# Stored Files Tab
file_uploader_pn = pn.widgets.FileInput(multiple=True, accept='.pdf', name="Upload PDFs")
upload_status_pn = pn.pane.Markdown("") # For upload messages

# Web Scraper Tab
scrape_base_url_pn = pn.widgets.TextInput(name="Base URL", value="https://www.imy.se")
scrape_listing_pn = pn.widgets.TextInput(name="Listing Endpoint", value="tillsyner")
scrape_pagination_pn = pn.widgets.TextInput(name="Pagination Format", value="?query=&page=")
scrape_pages_pn = pn.widgets.IntInput(name="Num Pages", value=1, start=1, step=1)
scrape_button_pn = pn.widgets.Button(name="🚀 Start Scraping & Process PDFs", button_type="info")
scraper_output_pn = pn.pane.Alert("", alert_type="light", height=200, styles={'overflow-y': 'auto'})


# Chat Interface
chat_input_pn = pn.widgets.TextAreaInput(placeholder="Ask a question...", height=80, name="") # No name for cleaner look
send_chat_button_pn = pn.widgets.Button(name="Send", button_type="primary")


# --- Reactive Functions for UI Updates ---

@pn.rx.memo # Memoize to avoid re-rendering if text_store_rx hasn't changed
def display_stored_files():
    current_files = text_store_rx() # Access the reactive list's current value
    if not current_files:
        return pn.pane.Markdown("_No files in this database._")
    
    unique_files_dict = {}
    for item in current_files:
        if item["file_hash"] not in unique_files_dict:
            unique_files_dict[item["file_hash"]] = item["file_name"]
    
    if not unique_files_dict:
         return pn.pane.Markdown("_No unique files found._")

    file_rows = []
    for f_hash, f_name in unique_files_dict.items():
        # Panel's way of handling button clicks within a dynamic list is more involved
        # For simplicity, delete functionality is harder to implement reactively here
        # without more complex param/class structures or manual event handling.
        # We'll just list them.
        file_rows.append(pn.Row(
            pn.pane.Markdown(f"- {f_name} `(hash: {f_hash[:7]})`"),
            # delete_button = pn.widgets.Button(name="🗑️", width=40, button_type="danger")
            # delete_button.on_click(lambda event, h=f_hash: delete_file_handler(h)) # Requires delete_file_handler
        ))
    return pn.Column(*file_rows, sizing_mode="stretch_width")

# Handler for DB selection change
@pn.rx.effect
def db_selection_handler():
    db_val = selected_db_rx() # Get current value
    status_msg = load_data_from_selected_db(db_val) # This updates global text_store_rx
    pn.state.notifications.info(f"Switched to {db_val}. {status_msg}", duration=4000)
    # The display_stored_files pane will automatically update because text_store_rx changed.

# Handler for File Upload
@pn.rx.effect
def file_upload_handler():
    files_bytes_list = file_uploader_pn.value
    filenames_list = file_uploader_pn.filename
    db_val = selected_db_rx()

    if not files_bytes_list:
        upload_status_pn.object = "" # Clear status
        return

    all_status_msgs = []
    processed_new = False
    if not isinstance(filenames_list, list): # Handle single file upload
        filenames_list = [filenames_list]
        files_bytes_list = [files_bytes_list]

    for fbytes, fname in zip(files_bytes_list, filenames_list):
        status, success = process_and_add_pdf(fbytes, fname, db_val) # This updates text_store_rx
        all_status_msgs.append(status)
        if success: processed_new = True
    
    upload_status_pn.object = "\n\n".join(all_status_msgs)
    if processed_new:
        # display_stored_files will update automatically due to text_store_rx change
        pn.state.notifications.success("Files processed.", duration=3000)
    else:
        pn.state.notifications.warning("No new files processed or all were duplicates.", duration=3000)
    
    file_uploader_pn.value = None # Reset file input
    file_uploader_pn.filename = ""


# Handler for Web Scraping
async def run_web_scraper_async(event): # event for button click
    scraper_output_pn.object = "Starting scraping..."
    scraper_output_pn.alert_type = "info"
    
    base_url = scrape_base_url_pn.value
    endpoint = scrape_listing_pn.value
    pagination = scrape_pagination_pn.value
    num_pages = scrape_pages_pn.value
    selected_db_val = selected_db_rx()

    if not all([base_url, endpoint, pagination, num_pages]):
        scraper_output_pn.object = "All scraper fields are required."
        scraper_output_pn.alert_type = "warning"
        return

    try:
        status_updates_list = ["Starting scraping..."]
        page_urls = get_all_page_urls_for_scraping(base_url, endpoint, pagination, num_pages)
        status_updates_list.append(f"Found {len(page_urls)} site pages to scan.")
        scraper_output_pn.object = "\n".join(status_updates_list)
        if not page_urls: return

        if sys.platform == "win32" and isinstance(asyncio.get_event_loop_policy(), asyncio.WindowsSelectorEventLoopPolicy):
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
        pdf_links = await process_scraped_pdf_links_async(page_urls) # Use await
        status_updates_list.append(f"Found {len(pdf_links)} unique PDF links.")
        scraper_output_pn.object = "\n".join(status_updates_list)
        if not pdf_links: return

        status_updates_list.append("Starting PDF download and processing...")
        scraper_output_pn.object = "\n".join(status_updates_list)
        
        processing_results = await batch_download_and_process_pdfs(pdf_links, selected_db_val) # Use await
        
        success_count = 0
        for msg, success, fname in processing_results:
            status_updates_list.append(f"{fname}: {msg} ({'Success' if success else 'Failed'})")
            if success: success_count += 1
        
        final_status = f"Scraping finished. Processed {success_count} new PDFs out of {len(pdf_links)} found."
        status_updates_list.append(final_status)
        scraper_output_pn.object = "\n".join(status_updates_list)
        scraper_output_pn.alert_type = "success"
        pn.state.notifications.success(final_status, duration=5000)
        # The file list (display_stored_files) will update because text_store_rx was modified by process_and_add_pdf
    except Exception as e:
        error_msg = f"Scraping Error: {e}"
        scraper_output_pn.object = error_msg
        scraper_output_pn.alert_type = "danger"
        pn.state.notifications.error(error_msg, duration=5000)

scrape_button_pn.on_click(run_web_scraper_async) # Panel handles async on_click


# Handler for Chat
@pn.rx.memo
def display_chat_messages():
    messages_display = []
    for chat_item in chat_history_rx(): # Iterate over reactive list
        sender = chat_item['sender']
        message = chat_item['message']
        model_info = chat_item.get('model_info', '')

        if sender == "User":
            messages_display.append(
                pn.Row(pn.pane.Markdown(f"**You:** {message}", styles={'text-align':'right'}), styles={'justify-content': 'flex-end', 'margin-bottom':'5px'})
            )
        else: # AI or System
            ai_message_content = [pn.pane.Markdown(f"**AI {model_info}:**\n\n{message}", styles={'white-space':'pre-wrap'})]
            messages_display.append(
                pn.Card(*ai_message_content, styles={'margin-bottom':'5px', 'max-width':'80%'})
            )
    return pn.Column(*messages_display, sizing_mode="stretch_width", scroll=True, height=500, styles={'padding':'10px', 'border':'1px solid #ccc', 'border-radius':'5px'})

async def on_chat_submit_async(event): # For button click
    user_input = chat_input_pn.value
    if not user_input or not user_input.strip(): return

    current_history = chat_history_rx.value.copy()
    current_history.append({"sender": "User", "message": user_input})
    chat_history_rx.value = current_history # Update reactive list
    chat_input_pn.value = "" # Clear input

    context = retrieve_context_from_db(user_input)
    
    # Get current config values from reactive app_config dict
    cfg = {key: val() for key, val in app_config.items()} # Dereference reactive values

    temp_vals = [cfg['temperature']]
    if cfg['vary_temperature'] and cfg['temperature'] > 0.01:
        temp_vals = sorted(list(set([round(max(0.01, cfg['temperature']*0.5),2), cfg['temperature'], round(min(1.0,cfg['temperature']*1.5),2) if cfg['temperature']*1.5<=1 else cfg['temperature']])))
    
    top_p_vals = [cfg['top_p']]
    if cfg['vary_top_p'] and cfg['top_p'] > 0.01:
        top_p_vals = sorted(list(set([round(max(0.01, cfg['top_p']*0.5),2), cfg['top_p'], round(min(1.0,cfg['top_p']*1.5),2) if cfg['top_p']*1.5<=1 else cfg['top_p']])))

    updated_history_segment = [] # Collect AI responses for this turn

    if not cfg['selected_models']:
        updated_history_segment.append({"sender": "System", "message": "No AI model selected."})
    else:
        for model_id in cfg['selected_models']:
            detail = AVAILABLE_MODELS_DICT.get(model_id, {})
            m_type, m_name = detail.get("type"), detail.get("name", model_id)
            for t_val in temp_vals:
                for p_val in top_p_vals:
                    resp = f"Error generating for {m_name}"
                    if m_type == "gemini": resp = generate_response_gemini(user_input, context, t_val, p_val, cfg['system_prompt'])
                    elif m_type == "together": resp = generate_response_together_ai(user_input, context, model_id, t_val, p_val, cfg['system_prompt'])
                    elif m_type == "openai": resp = generate_response_openai_api(user_input, context, t_val, p_val, cfg['system_prompt'])
                    
                    m_info = f"({m_name} T:{t_val} P:{p_val})"
                    updated_history_segment.append({"sender": "AI", "message": resp, "model_info": m_info})
                    if not cfg['vary_top_p']: break
                if not cfg['vary_temperature']: break
    
    final_history = chat_history_rx.value.copy() # Get potentially updated history if other calls happened
    final_history.extend(updated_history_segment)
    chat_history_rx.value = final_history

send_chat_button_pn.on_click(on_chat_submit_async)
# For submitting with Enter key in TextAreaInput, Panel's built-in submit_on_enter doesn't directly trigger an async func
# A workaround might involve a regular on_click and checking event type, or a JSLink.
# For simplicity, this example relies on the button.


# --- Assemble Layout using Panel Template ---
db_radio_pn_styled = pn.widgets.RadioBoxGroup(
    name="Database Backend", options=["Dropbox", "MongoDB"], value=selected_db_rx.value,
    inline=True
)
# Link radio button to reactive string
db_radio_pn_styled.param.watch(lambda event: selected_db_rx.update(event.new), 'value')


sidebar_content = pn.Column(
    pn.pane.Markdown("## 🛠️ IntelLaw Controls", styles={'text-align':'center', 'color':'white'}),
    db_radio_pn_styled,
    pn.Tabs(
        ("🧠 Config", pn.Column(
            model_selector_pn, vary_temp_cb_pn, temp_slider_pn,
            vary_top_p_cb_pn, top_p_slider_pn, system_prompt_pn,
            sizing_mode="stretch_width"
        )),
        ("📁 Stored Files", pn.Column(
            file_uploader_pn, upload_status_pn, display_stored_files, # display_stored_files is reactive
            sizing_mode="stretch_width"
        )),
        ("🌐 Web Scraper", pn.Column(
            scrape_base_url_pn, scrape_listing_pn, scrape_pagination_pn, scrape_pages_pn,
            scrape_button_pn, scraper_output_pn,
            sizing_mode="stretch_width"
        )),
    ),
    sizing_mode="stretch_width"
)

main_area_content = pn.Column(
    pn.pane.Markdown("## 📄 IntelLaw - Chat with Documents", styles={'color':'#0D6EFD'}), # Primary color
    display_chat_messages, # Reactive pane
    pn.Row(chat_input_pn, send_chat_button_pn, sizing_mode="stretch_width"),
    # model_responses_container_pn, # If you want a separate area for detailed model responses
    sizing_mode="stretch_width"
)


template = pn.template.FastListTemplate(
    title="IntelLaw (Panel UI)",
    sidebar=[sidebar_content],
    main=[main_area_content],
    theme=pn.theme.MaterialDarkTheme,
    sidebar_width=380,
    accent_base_color="#0D6EFD", # Bootstrap primary
    header_background="#0D6EFD",
)

# Initial population of file list after template is defined
# (display_stored_files is reactive, so it should update when text_store_rx changes)
# No explicit call needed here as it's linked to text_store_rx


if __name__ == "__main__":
    # This makes the app runnable with `python panel_app.py`
    # and also servable with `panel serve panel_app.py --show`
    if "--show" in sys.argv or "--autoreload" in sys.argv or os.environ.get("PANEL_SHOW") == "true":
         # Heuristic for `python panel_app.py` or `panel serve ... --show`
        print("Starting Panel server with .show()...")
        template.show(title="IntelLaw Panel App")
    else:
        # Makes it servable for `panel serve panel_app.py`
        print("Making Panel app servable. Run with 'panel serve panel_app.py --show'")
        template.servable()