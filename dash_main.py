# dash_main.py (or app.py)
import dash
from dash import dcc, html, Input, Output, State, no_update, callback_context, ALL, MATCH
import dash_bootstrap_components as dbc
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
from langdetect import detect
import json
from dotenv import load_dotenv
from io import BytesIO
from together import Together
import re
from pymongo import MongoClient, server_api
import subprocess # Keep for potential other uses, but Playwright specific call removed
import logging
from openai import OpenAI
import sys
import httpx
from urllib.parse import urljoin
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
import dropbox
import hashlib

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
MONGO_DB_NAME = "IntelLawDB_Dash"
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
    print(alert_msg) # Print status to console
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


# --- Dash App ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME], suppress_callback_exceptions=True)
server = app.server

# --- Initial Data Load ---
initialize_dropbox_client()
initialize_mongodb_client()
initial_load_status = load_data_from_selected_db("Dropbox")


# --- Layout Definition ---
sidebar_tabs = dbc.Tabs(
    id="sidebar-tabs", active_tab="tab-config", children=[
        dbc.Tab(label="Config", tab_id="tab-config", children=[
            dbc.Label("Selected AI Models (Max 3)", html_for="model-selector"),
            dcc.Dropdown(id='model-selector', options=AVAILABLE_MODELS_OPTIONS, multi=True, value=[AVAILABLE_MODELS_OPTIONS[0]['value']] if AVAILABLE_MODELS_OPTIONS else []),
            dbc.Checkbox(id='vary-temp-checkbox', label="Vary Temperature", value=True, className="mt-2"),
            dbc.Label("Temperature", html_for="temperature-slider", className="mt-2"),
            dcc.Slider(id='temperature-slider', min=0, max=1, step=0.05, value=0.7, marks={i/10: str(i/10) for i in range(0,11,2)}),
            dbc.Checkbox(id='vary-top-p-checkbox', label="Vary Top-P", value=False, className="mt-2"),
            dbc.Label("Top-P", html_for="top-p-slider", className="mt-2"),
            dcc.Slider(id='top-p-slider', min=0, max=1, step=0.05, value=0.9, marks={i/10: str(i/10) for i in range(0,11,2)}),
            dbc.Label("System Prompt", html_for="system-prompt-area", className="mt-2"),
            dbc.Textarea(id='system-prompt-area', value="You are a helpful assistant. Answer questions strictly based on the provided context. If there is no context, say 'I don't have enough information to answer that.'", rows=4),
        ]),
        dbc.Tab(label="Web Scraper", tab_id="tab-scraper", children=[
            dbc.Input(id='scrape-base-url', placeholder="Base URL (e.g., https://www.imy.se)", value="https://www.imy.se"),
            dbc.Input(id='scrape-listing-endpoint', placeholder="Listing Endpoint (e.g., tillsyner)", value="tillsyner", className="mt-2"),
            dbc.Input(id='scrape-pagination-format', placeholder="Pagination (e.g., ?query=&page=)", value="?query=&page=", className="mt-2"),
            dbc.Input(id='scrape-num-pages', placeholder="Num Pages (e.g., 3)", type="number", value=1, min=1, step=1, className="mt-2"),
            dbc.Button("Start Scraping & Process PDFs", id='start-scraping-button', color="info", className="w-100 mt-3"),
            html.Div(id='scraper-status-output', className="mt-2", style={'maxHeight': '200px', 'overflowY': 'auto'})
        ]),
        dbc.Tab(label="Stored Files", tab_id="tab-files", children=[
            dcc.Upload(id='upload-pdf-sidebar', children=html.Div(['Drag/Drop or ', html.A('Select PDFs')]),
                       style={'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'padding': '10px 0'}, multiple=True, className="mb-2"),
            html.Div(id='sidebar-upload-status-display', className="mb-2"),
            dbc.ListGroup(id='stored-files-display-list', style={'maxHeight': '300px', 'overflowY': 'auto'}),
        ]),
    ]
)

sidebar_layout = dbc.Card([
    dbc.CardHeader(html.H4("IntelLaw Controls", className="text-white")),
    dbc.CardBody([
        dbc.Label("Database Backend"),
        dbc.RadioItems(id='db-selection-radioitems', options=[
            {'label': 'Dropbox', 'value': 'Dropbox'},
            {'label': 'MongoDB', 'value': 'MongoDB', 'disabled': not MONGO_URI}
        ], value='Dropbox', inline=True, className="mb-3"),
        sidebar_tabs,
    ])
], color="dark", inverse=True, className="vh-100 overflow-auto", style={'position': 'fixed', 'top': 0, 'left': 0, 'bottom': 0, 'width': '24rem', 'padding': '1rem'})

main_content_layout = html.Div([
    dbc.Container([
        html.H1("📄 IntelLaw - Chat with Documents", className="my-4"),
        dbc.Alert(id='main-notifications-alert', is_open=False, duration=5000, dismissable=True),
        html.Div(id='chat-history-container', style={'height': '60vh', 'overflowY': 'auto', 'border': '1px solid #ccc', 'padding': '10px', 'marginBottom': '10px'}),
        dbc.InputGroup([
            dbc.Textarea(id='chat-user-input', placeholder="Ask a question...", rows=2),
            dbc.Button("Send", id='send-chat-msg-button', color="primary")
        ]),
        html.Div(id='model-response-output-tabs', className="mt-3")
    ], fluid=True)
], style={'marginLeft': '25rem', 'padding': '2rem'})


app.layout = dbc.Container([
    dcc.Store(id='selected-db-store', data='Dropbox'),
    dcc.Store(id='chat-history-store', data=[]),
    dcc.Store(id='app-configuration-store', data={
        "selected_models": [AVAILABLE_MODELS_OPTIONS[0]['value']] if AVAILABLE_MODELS_OPTIONS else [],
        "vary_temperature": True, "temperature": 0.7,
        "vary_top_p": False, "top_p": 0.9,
        "system_prompt": "You are a helpful assistant. Answer questions strictly based on the provided context. If there is no context, say 'I don't have enough information to answer that.'"
    }),
    dcc.Location(id='url', refresh=False),
    sidebar_layout,
    main_content_layout
], fluid=True, className="dbc")


# --- Callbacks ---
@app.callback(
    Output('app-configuration-store', 'data'),
    Input('model-selector', 'value'),
    Input('vary-temp-checkbox', 'value'), Input('temperature-slider', 'value'),
    Input('vary-top-p-checkbox', 'value'), Input('top-p-slider', 'value'),
    Input('system-prompt-area', 'value'),
    State('app-configuration-store', 'data')
)
def update_app_config(models, vary_t, temp, vary_p, top_p, sys_prompt, current_config):
    if models is not None and len(models) > 3: models = models[:3]
    current_config.update({
        "selected_models": models if models is not None else [],
        "vary_temperature": vary_t, "temperature": temp,
        "vary_top_p": vary_p, "top_p": top_p,
        "system_prompt": sys_prompt
    })
    return current_config

@app.callback(
    Output('selected-db-store', 'data'),
    Output('main-notifications-alert', 'children'),
    Output('main-notifications-alert', 'is_open'),
    Output('main-notifications-alert', 'color'),
    Output('stored-files-display-list', 'children', allow_duplicate=True),
    Input('db-selection-radioitems', 'value'),
    prevent_initial_call=True
)
def switch_database_and_reload(selected_db_val):
    global text_store, faiss_index
    status_message = load_data_from_selected_db(selected_db_val)
    
    files_list_items = []
    if text_store:
        unique_files = {}
        for item in text_store:
            if item["file_hash"] not in unique_files: unique_files[item["file_hash"]] = item["file_name"]
        for f_hash, f_name in unique_files.items():
            files_list_items.append(dbc.ListGroupItem([
                f_name,
                dbc.Button(html.I(className="fas fa-trash-alt"), id={'type': 'delete-file-btn', 'index': f_hash}, color="danger", size="sm", className="float-end")
            ]))
    else:
        files_list_items.append(dbc.ListGroupItem("No files in this database."))

    return selected_db_val, status_message, True, "info", files_list_items

@app.callback(
    Output('sidebar-upload-status-display', 'children'),
    Output('stored-files-display-list', 'children', allow_duplicate=True),
    Input('upload-pdf-sidebar', 'contents'),
    State('upload-pdf-sidebar', 'filename'),
    State('selected-db-store', 'data'),
    prevent_initial_call=True
)
def handle_sidebar_pdf_upload_files(list_of_contents, list_of_names, selected_db):
    if list_of_contents is None: return no_update, no_update
    
    alerts = []
    processed_any_new = False
    for content, name in zip(list_of_contents, list_of_names):
        content_type, content_string = content.split(',')
        decoded_content = base64.b64decode(content_string)
        status_msg, success = process_and_add_pdf(decoded_content, name, selected_db)
        alerts.append(dbc.Alert(status_msg, color="success" if success else "warning", dismissable=True, duration=6000))
        if success: processed_any_new = True
    
    if processed_any_new:
        files_list_items = []
        if text_store:
            unique_files = {}
            for item in text_store:
                if item["file_hash"] not in unique_files: unique_files[item["file_hash"]] = item["file_name"]
            for f_hash, f_name in unique_files.items():
                files_list_items.append(dbc.ListGroupItem([
                    f_name,
                    dbc.Button(html.I(className="fas fa-trash-alt"), id={'type': 'delete-file-btn', 'index': f_hash}, color="danger", size="sm", className="float-end")
                ]))
        else:
            files_list_items.append(dbc.ListGroupItem("No files in this database."))
        return alerts, files_list_items
    return alerts, no_update

@app.callback(
    Output('stored-files-display-list', 'children', allow_duplicate=True),
    Output('main-notifications-alert', 'children', allow_duplicate=True),
    Output('main-notifications-alert', 'is_open', allow_duplicate=True),
    Output('main-notifications-alert', 'color', allow_duplicate=True),
    Input({'type': 'delete-file-btn', 'index': ALL}, 'n_clicks'),
    State('selected-db-store', 'data'),
    prevent_initial_call=True
)
def delete_stored_file(n_clicks_list, selected_db_state): # Added selected_db_state
    global text_store, faiss_index, embedding_model
    ctx = callback_context
    if not ctx.triggered_id or not any(n_clicks_list): return no_update, no_update, False, no_update

    button_id = ctx.triggered_id
    file_hash_to_delete = button_id['index']
    
    original_text_store_len = len(text_store)
    text_store = [item for item in text_store if item["file_hash"] != file_hash_to_delete]

    if len(text_store) == original_text_store_len:
        return no_update, f"File hash {file_hash_to_delete} not found for deletion.", True, "warning"

    faiss_index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
    if text_store:
        all_texts_for_reindex = [item["text"] for item in text_store]
        if all_texts_for_reindex:
            embeddings = embedding_model.encode(all_texts_for_reindex)
            embeddings_np = np.array(embeddings).astype("float32")
            if embeddings_np.shape[0] > 0: faiss_index.add(embeddings_np)
    
    save_data_to_selected_db(selected_db_state) # Use the state for selected_db
    
    files_list_items = []
    if text_store:
        unique_files = {}
        for item in text_store:
            if item["file_hash"] not in unique_files: unique_files[item["file_hash"]] = item["file_name"]
        for f_hash, f_name in unique_files.items():
            files_list_items.append(dbc.ListGroupItem([
                f_name,
                dbc.Button(html.I(className="fas fa-trash-alt"), id={'type': 'delete-file-btn', 'index': f_hash}, color="danger", size="sm", className="float-end")
            ]))
    else:
        files_list_items.append(dbc.ListGroupItem("No files in this database."))
        
    return files_list_items, f"File with hash {file_hash_to_delete[:7]} and its data deleted.", True, "success"


@app.callback(
    Output('stored-files-display-list', 'children', allow_duplicate=True),
    Input('url', 'pathname'),
    prevent_initial_call='initial_duplicate' # CORRECTED LINE
)
def populate_initial_file_list(_):
    files_list_items = []
    if text_store:
        unique_files = {}
        for item in text_store:
            if item["file_hash"] not in unique_files: unique_files[item["file_hash"]] = item["file_name"]
        for f_hash, f_name in unique_files.items():
            files_list_items.append(dbc.ListGroupItem([
                f_name,
                dbc.Button(html.I(className="fas fa-trash-alt"), id={'type': 'delete-file-btn', 'index': f_hash}, color="danger", size="sm", className="float-end")
            ]))
    else:
        files_list_items.append(dbc.ListGroupItem("No files found or database not loaded."))
    return files_list_items

@app.callback(
    Output('scraper-status-output', 'children'),
    Input('start-scraping-button', 'n_clicks'),
    State('scrape-base-url', 'value'), State('scrape-listing-endpoint', 'value'),
    State('scrape-pagination-format', 'value'), State('scrape-num-pages', 'value'),
    State('selected-db-store', 'data'),
    prevent_initial_call=True
)
def handle_web_scraping(n_clicks, base_url, endpoint, pagination, num_pages, selected_db):
    if n_clicks is None: return no_update
    if not all([base_url, endpoint, pagination, num_pages]):
        return dbc.Alert("All scraper fields are required.", color="warning")
    
    try:
        status_updates = [html.P("Starting scraping...")]
        
        page_urls_to_scan = get_all_page_urls_for_scraping(base_url, endpoint, pagination, num_pages)
        status_updates.append(html.P(f"Found {len(page_urls_to_scan)} site pages to scan for PDF links."))
        if not page_urls_to_scan: return status_updates
        
        if sys.platform == "win32" and isinstance(asyncio.get_event_loop_policy(), asyncio.WindowsSelectorEventLoopPolicy):
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
        pdf_links_found = asyncio.run(process_scraped_pdf_links_async(page_urls_to_scan))
        status_updates.append(html.P(f"Found {len(pdf_links_found)} unique PDF links."))
        if not pdf_links_found: return status_updates

        status_updates.append(html.P("Starting PDF download and processing..."))
        
        processing_results = asyncio.run(batch_download_and_process_pdfs(pdf_links_found, selected_db))
        
        success_count = 0
        for msg, success, fname in processing_results:
            color = "success" if success else "danger"
            status_updates.append(html.Div(f"{fname}: {msg}", style={'color': color, 'fontSize': 'small'}))
            if success: success_count += 1
        
        status_updates.append(html.P(f"Scraping finished. Processed {success_count} new PDFs out of {len(pdf_links_found)} found."))
        # Here, you might want to trigger a refresh of the stored files list if the tab is active
        # This would require another callback or a more complex state update.
        return status_updates

    except Exception as e:
        return dbc.Alert(f"Scraping Error: {e}", color="danger")


@app.callback(
    Output('chat-history-container', 'children'),
    Output('chat-user-input', 'value'),
    Output('chat-history-store', 'data', allow_duplicate=True),
    Output('model-response-output-tabs', 'children'),
    Input('send-chat-msg-button', 'n_clicks'),
    State('chat-user-input', 'value'),
    State('chat-history-store', 'data'),
    State('app-configuration-store', 'data'),
    prevent_initial_call=True
)
def handle_chat_interaction(n_clicks, user_input_val, current_chat_history, app_config):
    if n_clicks is None or not user_input_val or not user_input_val.strip():
        return no_update, no_update, no_update, no_update

    current_chat_history.append({"sender": "User", "message": user_input_val})
    
    context_text = retrieve_context_from_db(user_input_val)
    
    model_responses_data = [] # To store data for tabbed display
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
        current_chat_history.append({"sender": "System", "message": "No AI model selected in configuration."})
    else:
        for model_id in selected_ai_models:
            model_detail = AVAILABLE_MODELS_DICT.get(model_id, {})
            model_type = model_detail.get("type")
            model_display_name = model_detail.get("name", model_id)

            for temp_val in temp_values_to_run:
                for top_p_val in top_p_values_to_run:
                    response_content = f"Error: Could not generate response for {model_display_name}." # Default error
                    if model_type == "gemini":
                        response_content = generate_response_gemini(user_input_val, context_text, temp_val, top_p_val, system_prompt_config)
                    elif model_type == "together":
                        response_content = generate_response_together_ai(user_input_val, context_text, model_id, temp_val, top_p_val, system_prompt_config)
                    elif model_type == "openai":
                        response_content = generate_response_openai_api(user_input_val, context_text, temp_val, top_p_val, system_prompt_config)
                    else:
                        response_content = f"Model type '{model_type}' for '{model_id}' not recognized."
                    
                    model_info_str = f"{model_display_name} (T:{temp_val}, P:{top_p_val})"
                    current_chat_history.append({
                        "sender": "AI", "message": response_content, "model_info": model_info_str
                    })
                    model_responses_data.append({ # For tabbed display
                        "model_name": model_display_name, "temp": temp_val, "top_p": top_p_val,
                        "response": response_content
                    })
                    if not app_config['vary_top_p']: break 
                if not app_config['vary_temperature']: break


    chat_display_elements = []
    for item in current_chat_history:
        color = "primary" if item['sender'] == "User" else "light" # AI is light for better readability
        align = "ms-auto" if item['sender'] == "User" else "me-auto"
        card_body_content = [html.Pre(item['message'], style={'whiteSpace': 'pre-wrap', 'wordBreak': 'break-word'})] # Use Pre for formatting
        
        card_elements = [dbc.CardBody(card_body_content)]
        if item['sender'] == "AI" and item.get('model_info'):
            card_elements.insert(0, dbc.CardHeader(html.Small(item['model_info'], className="text-muted"), style={'padding': '0.25rem 0.5rem', 'fontSize': '0.75em', 'backgroundColor': '#f8f9fa'}))
        
        chat_display_elements.append(
            dbc.Card(card_elements, color=color, inverse=(item['sender'] == "User"), className=f"mb-2 {align}", style={'maxWidth': '80%'})
        )

    model_response_tabs_content = []
    if model_responses_data:
        tabs_children = []
        for i, resp_data in enumerate(model_responses_data):
            tabs_children.append(dbc.Tab(
                dbc.Card(dbc.CardBody(html.Pre(resp_data['response'], style={'whiteSpace': 'pre-wrap', 'wordBreak': 'break-word'}))),
                label=f"{resp_data['model_name']} (T:{resp_data['temp']}, P:{resp_data['top_p']})",
                tab_id=f"resp-tab-{i}"
            ))
        model_response_tabs_content = dbc.Tabs(tabs_children, active_tab=f"resp-tab-0" if tabs_children else None, className="mt-3")

    return chat_display_elements, "", current_chat_history, model_response_tabs_content


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')