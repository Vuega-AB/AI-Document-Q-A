# gradio_app.py
import gradio as gr
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
# import subprocess # Not used in this Gradio version for Playwright
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
# --- Environment Variables & Initializations ---
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
MONGO_DB_NAME = "IntelLawDB_Gradio" # Changed DB name slightly for Gradio version
FAISS_COLLECTION_NAME = "faiss_index_store"
TEXT_STORE_COLLECTION_NAME = "text_content_store"

# Initialize API clients if keys are present
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model_genai = genai.GenerativeModel("gemini-2.0-flash")
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


dbx = None
mongo_client_instance = None
mongo_db_obj = None

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
faiss_index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
text_store = []

AVAILABLE_MODELS_DICT = {
    "gemini-2.0-flash": {"price": "Custom", "type": "gemini", "name": "Gemini 2.0 Flash"},
    "openai-4o": {"price": "Custom", "type": "openai", "name": "OpenAI GPT-4o"},
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": {"price": "$0.88", "type": "together", "name": "Llama3.3 70B Turbo"},
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": {"price": "$3.50", "type": "together", "name": "Llama3.1 405B Turbo"},
    "microsoft/WizardLM-2-8x22B": {"price": "$1.20", "type": "together", "name": "WizardLM-2 8x22B"},
    "mistralai/Mixtral-8x22B-Instruct-v0.1": {"price": "$1.20", "type": "together", "name": "Mixtral 8x22B Instruct"},
    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO": {"price": "$0.60", "type": "together", "name": "Hermes-2 Mixtral DPO"},
}
# For Gradio Dropdown, it's easier if choices are just list of strings (names)
# We'll map back to model_id using the name.
AVAILABLE_MODELS_NAMES = [details['name'] for details in AVAILABLE_MODELS_DICT.values()]
MODEL_NAME_TO_ID_MAP = {details['name']: model_id for model_id, details in AVAILABLE_MODELS_DICT.items()}


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
def save_data_to_selected_db(selected_db):
    global faiss_index, text_store, dbx, mongo_db_obj
    # Ensure clients are initialized if None (might happen if initial load failed but user selected DB later)
    if selected_db == "Dropbox" and dbx is None: initialize_dropbox_client()
    if selected_db == "MongoDB" and mongo_db_obj is None: initialize_mongodb_client()

    if selected_db == "Dropbox":
        if dbx is None: print("Dropbox not initialized for saving."); return
        try:
            temp_idx_file = "temp_faiss_to_dropbox.index"
            faiss.write_index(faiss_index, temp_idx_file)
            with open(temp_idx_file, "rb") as f: dbx.files_upload(f.read(), INDEX_FILE_DROPBOX, mode=dropbox.files.WriteMode.overwrite)
            os.remove(temp_idx_file)
            text_json = json.dumps(text_store, ensure_ascii=False, indent=4).encode('utf-8')
            dbx.files_upload(text_json, TEXT_FILE_DROPBOX, mode=dropbox.files.WriteMode.overwrite)
            print(f"Data saved to Dropbox. Index size: {faiss_index.ntotal}, Text items: {len(text_store)}")
        except Exception as e: print(f"Error saving to Dropbox: {e}")
    elif selected_db == "MongoDB":
        if mongo_db_obj is None: print("MongoDB not initialized for saving."); return
        try:
            temp_idx_file = "temp_faiss_to_mongo.idx"
            faiss.write_index(faiss_index, temp_idx_file)
            with open(temp_idx_file, "rb") as f: index_bytes = f.read()
            os.remove(temp_idx_file)
            mongo_db_obj[FAISS_COLLECTION_NAME].update_one({"_id": "main_faiss_index"}, {"$set": {"index_data": index_bytes}}, upsert=True)
            mongo_db_obj[TEXT_STORE_COLLECTION_NAME].delete_many({})
            if text_store: mongo_db_obj[TEXT_STORE_COLLECTION_NAME].insert_many(text_store)
            print(f"Data saved to MongoDB. Index size: {faiss_index.ntotal}, Text items: {len(text_store)}")
        except Exception as e: print(f"Error saving to MongoDB: {e}")

def load_data_from_selected_db(selected_db):
    global faiss_index, text_store, embedding_model, dbx, mongo_db_obj
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
        if mongo_db_obj is None: initialize_mongodb_client()
        if mongo_db_obj is None: alert_msg = "MongoDB not initialized. Cannot load."; return alert_msg
        try:
            index_doc = mongo_db_obj[FAISS_COLLECTION_NAME].find_one({"_id": "main_faiss_index"})
            if index_doc and "index_data" in index_doc:
                temp_idx_file = "temp_faiss_from_mongo.idx"
                with open(temp_idx_file, "wb") as f: f.write(index_doc["index_data"])
                faiss_index = faiss.read_index(temp_idx_file)
                os.remove(temp_idx_file)
            text_docs = mongo_db_obj[TEXT_STORE_COLLECTION_NAME].find({})
            text_store = [{k: v for k, v in doc.items() if k != '_id'} for doc in text_docs]
            alert_msg = f"Data loaded from MongoDB. {len(text_store)} text items, index has {faiss_index.ntotal} vectors."
        except Exception as e: alert_msg = f"Error loading from MongoDB: {e}"
    print(f"Load attempt for {selected_db}: {alert_msg}")
    return alert_msg

# --- PDF Processing ---
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
    for page in reader.pages: page_text = page.extract_text(); text += page_text + "\n" if page_text else ""
    return text
def process_and_add_pdf(pdf_bytes, file_name, selected_db):
    global text_store, faiss_index, embedding_model
    file_hash = hashlib.md5(pdf_bytes).hexdigest()
    if any(item['file_hash'] == file_hash for item in text_store):
        return f"File '{file_name}' (hash: {file_hash[:7]}) already exists.", False
    
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
    
    save_data_to_selected_db(selected_db)
    return f"Processed and added '{file_name}'. Chunks: {len(chunks)}, Index size: {faiss_index.ntotal}", True

# --- AI Response Generation ---
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
    global text_store, faiss_index, embedding_model
    if faiss_index.ntotal == 0: return "No documents indexed."
    query_embedding = embedding_model.encode([query])
    distances, indices = faiss_index.search(query_embedding.astype(np.float32), top_k)
    valid_indices = [i for i in indices[0] if 0 <= i < len(text_store)]
    retrieved_texts = [text_store[idx]["text"] for idx in valid_indices]
    return "\n\n".join(retrieved_texts) if retrieved_texts else "No relevant context found."

# --- Web Scraping ---
async def fetch_page_async(url):
    async with httpx.AsyncClient() as client: response = await client.get(url, timeout=30.0); return response.text, str(response.url)
async def extract_pdf_links_from_url_async(url):
    try: html, base_url = await fetch_page_async(url); soup = BeautifulSoup(html, "html.parser"); return [urljoin(base_url, a["href"]) for a in soup.find_all("a", href=True) if ".pdf" in a["href"].lower()]
    except Exception as e: print(f"Error scraping {url}: {e}"); return []
async def process_scraped_pdf_links_async(urls):
    all_pdf_links = set()
    for url_group in await asyncio.gather(*[extract_pdf_links_from_url_async(u) for u in urls]): all_pdf_links.update(url_group)
    return list(all_pdf_links)
async def download_and_process_scraped_pdf(session, pdf_link, selected_db):
    try:
        async with session.get(pdf_link, timeout=60) as response:
            if response.status == 200:
                pdf_bytes = await response.read()
                filename = os.path.basename(pdf_link)
                status_msg, success = process_and_add_pdf(pdf_bytes, filename, selected_db)
                return status_msg, success, filename
            return f"Failed to download {pdf_link} (status: {response.status})", False, os.path.basename(pdf_link)
    except Exception as e: return f"Error processing {pdf_link}: {e}", False, os.path.basename(pdf_link)
async def batch_download_and_process_pdfs(pdf_links, selected_db):
    results = []
    async with aiohttp.ClientSession() as session:
        tasks = [download_and_process_scraped_pdf(session, link, selected_db) for link in pdf_links]
        for result in await asyncio.gather(*tasks): results.append(result)
    return results
def get_page_items_sync(url, base_url_val, listing_endpoint_val):
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
def get_all_page_urls_for_scraping(base_url_val, listing_endpoint_val, pagination_format_val, num_pages_val):
    all_page_urls_to_scrape = set()
    for page_num in range(1, int(num_pages_val) + 1):
        url = f"{base_url_val}/{listing_endpoint_val}/{pagination_format_val}{page_num}"
        page_items = get_page_items_sync(url, base_url_val, listing_endpoint_val)
        if not page_items: break
        all_page_urls_to_scrape.update(page_items)
    return list(all_page_urls_to_scrape)


# --- Initial Data Load (call this once at the start of script execution) ---
initialize_dropbox_client()
initialize_mongodb_client()
initial_load_msg = load_data_from_selected_db("Dropbox") # Default to Dropbox
print(f"Gradio App Initial Load: {initial_load_msg}")


# --- Gradio UI Functions (Wrapper functions for your backend logic) ---
def get_current_file_list_md():
    files_md = "#### Stored Files\n---\n"
    if text_store:
        unique_files = {}
        for item in text_store:
            if item["file_hash"] not in unique_files: unique_files[item["file_hash"]] = item["file_name"]
        if unique_files:
            for f_name in unique_files.values():
                files_md += f"- {f_name}\n"
        else:
            files_md += "_No unique files found in current store._\n"
    else:
        files_md += "_No files in this database._\n"
    return files_md

def update_app_config_gr(models_names, vary_t, temp, vary_p, top_p, sys_prompt, current_app_config_state):
    # Map model names back to model IDs
    selected_model_ids = [MODEL_NAME_TO_ID_MAP[name] for name in models_names if name in MODEL_NAME_TO_ID_MAP]
    if len(selected_model_ids) > 3: selected_model_ids = selected_model_ids[:3]
    
    current_app_config_state.update({
        "selected_models": selected_model_ids,
        "vary_temperature": vary_t, "temperature": temp,
        "vary_top_p": vary_p, "top_p": top_p,
        "system_prompt": sys_prompt
    })
    return "Configuration updated.", current_app_config_state # Return updated state

def switch_db_gr(selected_db_val, current_selected_db_state):
    if selected_db_val == current_selected_db_state: # No change if same DB selected
        return current_selected_db_state, f"Already using {selected_db_val}. No change.", get_current_file_list_md()

    status_message = load_data_from_selected_db(selected_db_val) # This updates global text_store, faiss_index
    return selected_db_val, status_message, get_current_file_list_md()

def handle_pdf_upload_gr(files_obj_list, selected_db_from_state):
    if files_obj_list is None:
        return "No files uploaded.", get_current_file_list_md()

    alerts = []
    processed_any_new = False
    for file_obj in files_obj_list:
        file_path = file_obj.name # Gradio file object's name attribute is the temp file path
        file_display_name = os.path.basename(file_path)
        with open(file_path, 'rb') as f:
            pdf_bytes = f.read()
        
        status_msg, success = process_and_add_pdf(pdf_bytes, file_display_name, selected_db_from_state)
        alerts.append(status_msg)
        if success:
            processed_any_new = True
    
    status_summary = "\n".join(alerts)
    return status_summary, get_current_file_list_md()


def chat_interface_gr(user_input, chat_history_list, selected_db_state, app_config_state):
    # selected_db_state is not directly used here but good to have for context if needed
    if not user_input or not user_input.strip():
        return chat_history_list, ""

    chat_history_list.append((user_input, None))

    context_text = retrieve_context_from_db(user_input)
    
    model_responses_html = ""
    temp_config = app_config_state['temperature']
    top_p_config = app_config_state['top_p']
    system_prompt_config = app_config_state['system_prompt']
    
    temp_values_to_run = [temp_config]
    if app_config_state['vary_temperature'] and temp_config > 0.01:
        temp_values_to_run = sorted(list(set([
            round(max(0.01, temp_config * 0.5), 2), temp_config, 
            round(min(1.0, temp_config * 1.5), 2) if temp_config * 1.5 <=1.0 else temp_config])))

    top_p_values_to_run = [top_p_config]
    if app_config_state['vary_top_p'] and top_p_config > 0.01:
        top_p_values_to_run = sorted(list(set([
            round(max(0.01, top_p_config * 0.5), 2), top_p_config,
            round(min(1.0, top_p_config * 1.5),2) if top_p_config * 1.5 <= 1.0 else top_p_config])))
        
    selected_ai_model_ids = app_config_state.get('selected_models', [])
    if not selected_ai_model_ids:
        ai_response_text = "System: No AI model selected in configuration."
        chat_history_list.append((None, ai_response_text))
        model_responses_html = f"<p>{ai_response_text}</p>"
    else:
        combined_ai_responses_for_chat = []
        for model_id in selected_ai_model_ids:
            model_detail = AVAILABLE_MODELS_DICT.get(model_id, {})
            model_type = model_detail.get("type")
            model_display_name = model_detail.get("name", model_id)

            for temp_val in temp_values_to_run:
                for top_p_val in top_p_values_to_run:
                    response_content = f"Error for {model_display_name}."
                    if model_type == "gemini":
                        response_content = generate_response_gemini(user_input, context_text, temp_val, top_p_val, system_prompt_config)
                    elif model_type == "together":
                        response_content = generate_response_together_ai(user_input, context_text, model_id, temp_val, top_p_val, system_prompt_config)
                    elif model_type == "openai":
                        response_content = generate_response_openai_api(user_input, context_text, temp_val, top_p_val, system_prompt_config)
                    
                    model_info_str = f"{model_display_name} (T:{temp_val}, P:{top_p_val})"
                    combined_ai_responses_for_chat.append(f"--- {model_info_str} ---\n{response_content}")
                    model_responses_html += f"<div><h4>{model_info_str}</h4><pre style='white-space: pre-wrap; word-break: break-word;'>{response_content}</pre><hr/></div>"

                    if not app_config_state['vary_top_p']: break
                if not app_config_state['vary_temperature']: break
        
        if combined_ai_responses_for_chat:
            chat_history_list.append((None, "\n\n".join(combined_ai_responses_for_chat)))

    return chat_history_list, model_responses_html

def run_scraper_gr_wrapper(base_url, endpoint, pagination, num_pages, selected_db_val):
    # This is a synchronous wrapper for Gradio.
    if sys.platform == "win32" and isinstance(asyncio.get_event_loop_policy(), asyncio.WindowsSelectorEventLoopPolicy):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    status_updates = ["Starting scraping..."]
    page_urls_to_scan = get_all_page_urls_for_scraping(base_url, endpoint, pagination, num_pages)
    status_updates.append(f"Found {len(page_urls_to_scan)} site pages to scan for PDF links.")
    if not page_urls_to_scan: return "\n".join(status_updates), get_current_file_list_md()
    
    pdf_links_found = asyncio.run(process_scraped_pdf_links_async(page_urls_to_scan))
    status_updates.append(f"Found {len(pdf_links_found)} unique PDF links.")
    if not pdf_links_found: return "\n".join(status_updates), get_current_file_list_md()

    status_updates.append("Starting PDF download and processing...")
    processing_results = asyncio.run(batch_download_and_process_pdfs(pdf_links_found, selected_db_val))
    
    success_count = 0
    for msg, success, fname in processing_results:
        status_updates.append(f"{fname}: {msg} ({'Success' if success else 'Failed'})")
        if success: success_count += 1
    
    status_updates.append(f"Scraping finished. Processed {success_count} new PDFs out of {len(pdf_links_found)} found.")
    return "\n".join(status_updates), get_current_file_list_md()


# --- Gradio Blocks UI Definition ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.sky), title="IntelLaw Gradio") as demo:
    # --- State Variables for Gradio ---
    # These hold data that persists across interactions and can be passed to functions
    # Initial values are set from the global variables populated at script start
    selected_db_state_gr = gr.State(value="Dropbox") # Default selected DB
    app_config_state_gr = gr.State(value={
        "selected_models": [MODEL_NAME_TO_ID_MAP[AVAILABLE_MODELS_NAMES[0]]] if AVAILABLE_MODELS_NAMES else [],
        "vary_temperature": True, "temperature": 0.7,
        "vary_top_p": False, "top_p": 0.9,
        "system_prompt": "You are a helpful assistant. Answer questions strictly based on the provided context. If there is no context, say 'I don't have enough information to answer that.'"
    })

    gr.Markdown("# 📄 IntelLaw - Chat with Documents (Gradio UI)")

    with gr.Row():
        with gr.Column(scale=1, min_width=400):
            gr.Markdown("### 🛠️ Controls")
            with gr.Accordion("Database Backend", open=True):
                db_radio_gr = gr.Radio(
                    label="Choose Database",
                    choices=[("Dropbox", "Dropbox"), ("MongoDB", "MongoDB")],
                    value="Dropbox", # Initial value for the component
                    interactive=True
                )
                db_status_output_gr = gr.Textbox(label="DB Status", interactive=False, value=initial_load_msg)
            
            with gr.Tabs():
                with gr.TabItem("🧠 Config"):
                    model_selector_gr = gr.Dropdown(
                        label="AI Models (Max 3)",
                        choices=AVAILABLE_MODELS_NAMES,
                        value=[AVAILABLE_MODELS_NAMES[0]] if AVAILABLE_MODELS_NAMES else [],
                        multiselect=True, max_choices=3, allow_custom_value=False
                    )
                    vary_temp_cb_gr = gr.Checkbox(label="Vary Temperature", value=True)
                    temp_slider_gr = gr.Slider(label="Temperature", minimum=0, maximum=1, step=0.05, value=0.7)
                    vary_top_p_cb_gr = gr.Checkbox(label="Vary Top-P", value=False)
                    top_p_slider_gr = gr.Slider(label="Top-P", minimum=0, maximum=1, step=0.05, value=0.9)
                    system_prompt_ta_gr = gr.Textbox(label="System Prompt", value="You are a helpful assistant...", lines=4)
                    update_config_btn_gr = gr.Button("Update Config")
                    config_status_gr = gr.Textbox(label="Config Status", interactive=False)

                with gr.TabItem("📁 Stored Files"):
                    file_uploader_gr = gr.Files(label="Upload PDFs", file_count="multiple", type="filepath")
                    upload_status_gr = gr.Textbox(label="Upload Status", interactive=False, lines=3, max_lines=5)
                    stored_files_md_gr = gr.Markdown(value=get_current_file_list_md())

                with gr.TabItem("🌐 Web Scraper"):
                    scrape_base_url_gr = gr.Textbox(label="Base URL", value="https://www.imy.se")
                    scrape_listing_gr = gr.Textbox(label="Listing Endpoint", value="tillsyner")
                    scrape_pagination_gr = gr.Textbox(label="Pagination Format", value="?query=&page=")
                    scrape_pages_gr = gr.Number(label="Num Pages", value=1, precision=0)
                    scrape_button_gr = gr.Button("Start Scraping & Process")
                    scraper_output_gr = gr.Textbox(label="Scraper Output", lines=5, max_lines=10, interactive=False)

        with gr.Column(scale=3):
            gr.Markdown("### 💬 Chat Interface")
            chatbot_gr = gr.Chatbot(label="IntelLaw Chatbot", height=600, show_copy_button=True)
            chat_input_gr = gr.Textbox(label="Your Question:", placeholder="Ask anything about the documents...", show_label=False)
            send_button_gr = gr.Button("Send")
            
            gr.Markdown("### 🤖 Model Responses (Details)")
            model_responses_display_gr = gr.HTML()


    # --- Event Handlers for Gradio Components ---
    db_radio_gr.change(
        fn=switch_db_gr,
        inputs=[db_radio_gr, selected_db_state_gr], # Pass current state to compare
        outputs=[selected_db_state_gr, db_status_output_gr, stored_files_md_gr]
    )

    update_config_btn_gr.click(
        fn=update_app_config_gr,
        inputs=[model_selector_gr, vary_temp_cb_gr, temp_slider_gr, vary_top_p_cb_gr, top_p_slider_gr, system_prompt_ta_gr, app_config_state_gr],
        outputs=[config_status_gr, app_config_state_gr] # Update state
    )

    file_uploader_gr.upload(
        fn=handle_pdf_upload_gr,
        inputs=[file_uploader_gr, selected_db_state_gr],
        outputs=[upload_status_gr, stored_files_md_gr]
    )

    # Chat interaction
    chat_inputs = [chat_input_gr, chatbot_gr, selected_db_state_gr, app_config_state_gr]
    chat_outputs = [chatbot_gr, model_responses_display_gr]

    send_button_gr.click(fn=chat_interface_gr, inputs=chat_inputs, outputs=chat_outputs)\
                  .then(lambda: gr.update(value=""), outputs=chat_input_gr) # Clear input

    chat_input_gr.submit(fn=chat_interface_gr, inputs=chat_inputs, outputs=chat_outputs)\
                   .then(lambda: gr.update(value=""), outputs=chat_input_gr) # Clear input
    
    scrape_button_gr.click(
        fn=run_scraper_gr_wrapper,
        inputs=[scrape_base_url_gr, scrape_listing_gr, scrape_pagination_gr, scrape_pages_gr, selected_db_state_gr],
        outputs=[scraper_output_gr, stored_files_md_gr] # Also update file list after scraping
    )

if __name__ == "__main__":
    demo.launch()