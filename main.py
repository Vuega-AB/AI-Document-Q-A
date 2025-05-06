import streamlit as st
# Add custom CSS to hide the GitHub icon
st.markdown(
    """
    <style>
    [data-testid="stToolbar"]{
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
import os
import requests
from bs4 import BeautifulSoup
import aiohttp
import asyncio
import PyPDF2
import io
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
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import subprocess
import logging
from openai import OpenAI
import sys
import httpx
from urllib.parse import urljoin
import google.generativeai as genai
from google.api_core import exceptions
import dropbox
import hashlib


# ================== Environment Variables ==================
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGO_URI = os.getenv("MongoDB")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DROPBOX_REFRESH_TOKEN = os.getenv("DROPBOX_REFRESH_TOKEN")
DROPBOX_APP_KEY = os.getenv("DROPBOX_APP_KEY")
DROPBOX_APP_SECRET = os.getenv("DROPBOX_APP_SECRET")

# Initialize models and configurations
CONFIG_FILENAME = "config.json"
INDEX_FILE_DROPBOX = "/faiss_index.index"
TEXT_FILE_DROPBOX = "/text_store.json"
TOKEN_FILE = "dropbox_token.json"

# MongoDB Configuration
MONGO_DB_NAME = "IntelLawDB"
FAISS_COLLECTION_NAME = "faiss_index_store"
TEXT_STORE_COLLECTION_NAME = "text_content_store"

# =================== Connections ============================
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")
together_client = Together(api_key=TOGETHER_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize DB client variables
dbx = None
mongo_client_instance = None
mongo_db_obj = None


AVAILABLE_MODELS_DICT = {
    "gemini-2.0-flash": {"price": "Custom", "type": "gemini"},
    "openai-4o": {"price": "Custom", "type": "openai"},
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": {"price": "$0.88", "type": "together"},
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": {"price": "$3.50", "type": "together"},
    "microsoft/WizardLM-2-8x22B": {"price": "$1.20", "type": "together"},
    "mistralai/Mixtral-8x22B-Instruct-v0.1": {"price": "$1.20", "type": "together"},
    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO": {"price": "$0.60", "type": "together"},
}
AVAILABLE_MODELS = list(AVAILABLE_MODELS_DICT.keys())

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "config" not in st.session_state:
    st.session_state.config = {
        "temperature": 0.7,
        "top_p": 0.9,
        "system_prompt": "You are a helpful assistant. Answer questions strictly based on the provided context. If there is no context, say 'I don't have enough information to answer that.'",
        "selected_models": AVAILABLE_MODELS[:1],
        "vary_temperature": True,
        "vary_top_p": False
    }
if "selected_db" not in st.session_state:
    st.session_state.selected_db = "Dropbox"

# -----------------------------------------------------------------------------
# Dropbox Functions
# -----------------------------------------------------------------------------
def load_access_token():
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "r") as file:
            data = json.load(file)
            return data.get("access_token"), data.get("expires_at")
    return None, None

def save_access_token(access_token, expires_in):
    expires_at = int(time.time()) + expires_in - 300
    with open(TOKEN_FILE, "w") as file:
        json.dump({"access_token": access_token, "expires_at": expires_at}, file)

def get_dropbox_access_token():
    response = requests.post(
        "https://api.dropbox.com/oauth2/token",
        data={"grant_type": "refresh_token", "refresh_token": DROPBOX_REFRESH_TOKEN},
        auth=(DROPBOX_APP_KEY, DROPBOX_APP_SECRET),
    )
    if response.status_code == 200:
        data = response.json()
        access_token = data["access_token"]
        expires_in = data.get("expires_in", 14400)
        save_access_token(access_token, expires_in)
        return access_token
    else:
        st.error(f"Failed to refresh Dropbox token: {response.text}")
        raise Exception(f"Failed to refresh token: {response.text}")

def get_valid_access_token():
     access_token, expires_at = load_access_token()
     if access_token and expires_at and int(time.time()) < expires_at:
         return access_token
     return get_dropbox_access_token()

def initialize_dropbox():
    global dbx
    if not (DROPBOX_REFRESH_TOKEN and DROPBOX_APP_KEY and DROPBOX_APP_SECRET):
        return None
    try:
        access_token = get_valid_access_token()
        dbx_client = dropbox.Dropbox(access_token)
        return dbx_client
    except Exception as e:
        st.sidebar.error(f"Error connecting to Dropbox: {e}")
        return None

def save_data_to_dropbox(current_faiss_index, current_text_store):
    global dbx
    if dbx is None: # MODIFIED
        st.error("Dropbox not initialized. Cannot save.")
        return
    try:
        temp_index_filename = "temp_faiss_for_dropbox.index"
        faiss.write_index(current_faiss_index, temp_index_filename)
        with open(temp_index_filename, "rb") as f:
            dbx.files_upload(f.read(), INDEX_FILE_DROPBOX, mode=dropbox.files.WriteMode.overwrite)
        os.remove(temp_index_filename)

        text_json = json.dumps(current_text_store, ensure_ascii=False, indent=4).encode('utf-8')
        dbx.files_upload(text_json, TEXT_FILE_DROPBOX, mode=dropbox.files.WriteMode.overwrite)
    except Exception as e:
        st.error(f"Error saving data to Dropbox: {e}")

def load_data_from_dropbox(embedding_model_instance):
    global dbx
    index = faiss.IndexFlatL2(embedding_model_instance.get_sentence_embedding_dimension())
    text_store = []
    if dbx is None: # MODIFIED
        st.warning("Dropbox not initialized. Cannot load from Dropbox.")
        return embedding_model_instance, index, text_store
    try:
        _, res_index = dbx.files_download(path=INDEX_FILE_DROPBOX)
        temp_index_filename = "temp_faiss_from_dropbox.index"
        with open(temp_index_filename, "wb") as f_index:
            f_index.write(res_index.content)
        index = faiss.read_index(temp_index_filename)
        os.remove(temp_index_filename)

        _, res_text = dbx.files_download(path=TEXT_FILE_DROPBOX)
        text_store = json.loads(res_text.content.decode('utf-8'))
    except dropbox.exceptions.ApiError as e:
        if isinstance(e.error, dropbox.files.DownloadError) and e.error.is_path() and e.error.get_path().is_not_found():
            st.info("No existing data found on Dropbox. Initializing new store.")
        else:
            st.error(f"API error loading data from Dropbox: {e}")
    except Exception as e:
        st.error(f"Error loading data from Dropbox: {e}")
    return embedding_model_instance, index, text_store

# -----------------------------------------------------------------------------
# MongoDB Functions
# -----------------------------------------------------------------------------
def init_mongo_client_global():
    global mongo_client_instance, mongo_db_obj
    if MONGO_URI and mongo_client_instance is None: # MODIFIED
        try:
            mongo_client_instance = MongoClient(MONGO_URI, server_api=ServerApi('1'))
            mongo_client_instance.admin.command('ping')
            mongo_db_obj = mongo_client_instance[MONGO_DB_NAME]
        except Exception as e:
            st.sidebar.error(f"MongoDB connection failed: {e}")
            mongo_client_instance = None
            mongo_db_obj = None
    elif not MONGO_URI and 'mongo_warned' not in st.session_state :
        st.session_state.mongo_warned = True


def save_data_to_mongodb(current_faiss_index, current_text_store):
    global mongo_db_obj
    if mongo_db_obj is None: # MODIFIED
        st.error("MongoDB not connected. Cannot save.")
        return
    try:
        temp_index_filename = "temp_faiss_for_mongo.idx"
        faiss.write_index(current_faiss_index, temp_index_filename)
        with open(temp_index_filename, "rb") as f:
            index_bytes = f.read()
        os.remove(temp_index_filename)
        mongo_db_obj[FAISS_COLLECTION_NAME].update_one(
            {"_id": "main_faiss_index"},
            {"$set": {"index_data": index_bytes, "timestamp": time.time()}},
            upsert=True
        )

        mongo_db_obj[TEXT_STORE_COLLECTION_NAME].delete_many({})
        if current_text_store:
            mongo_db_obj[TEXT_STORE_COLLECTION_NAME].insert_many(current_text_store)
    except Exception as e:
        st.error(f"Error saving data to MongoDB: {e}")

def load_data_from_mongodb(embedding_model_instance):
    global mongo_db_obj
    index = faiss.IndexFlatL2(embedding_model_instance.get_sentence_embedding_dimension())
    text_store = []
    if mongo_db_obj is None: # MODIFIED - This was the line causing the error
        st.warning("MongoDB not connected. Cannot load from MongoDB.")
        return embedding_model_instance, index, text_store
    try:
        index_doc = mongo_db_obj[FAISS_COLLECTION_NAME].find_one({"_id": "main_faiss_index"})
        if index_doc and "index_data" in index_doc:
            index_bytes = index_doc["index_data"]
            temp_index_filename = "temp_faiss_from_mongo.idx"
            with open(temp_index_filename, "wb") as f:
                f.write(index_bytes)
            index = faiss.read_index(temp_index_filename)
            os.remove(temp_index_filename)
        else:
            st.info("No FAISS index found in MongoDB. Initializing new one.")

        text_docs_cursor = mongo_db_obj[TEXT_STORE_COLLECTION_NAME].find({})
        text_store = [{k: v for k, v in doc.items() if k != '_id'} for doc in text_docs_cursor]
    except Exception as e:
        st.error(f"Error loading data from MongoDB: {e}")
    return embedding_model_instance, index, text_store

# -----------------------------------------------------------------------------
# Generic Data Functions & Initialization
# -----------------------------------------------------------------------------
def save_data():
    global faiss_index, text_store
    if st.session_state.selected_db == "Dropbox":
        save_data_to_dropbox(faiss_index, text_store)
    elif st.session_state.selected_db == "MongoDB":
        save_data_to_mongodb(faiss_index, text_store)
    else:
        st.error(f"Unknown database selection: {st.session_state.selected_db}")

def initialize_and_load_data():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    idx = faiss.IndexFlatL2(model.get_sentence_embedding_dimension())
    txt_store = []

    if st.session_state.selected_db == "Dropbox":
        global dbx
        if dbx is None:
             dbx = initialize_dropbox()
        return load_data_from_dropbox(model)
    elif st.session_state.selected_db == "MongoDB":
        global mongo_db_obj
        if mongo_db_obj is None:
            init_mongo_client_global()
        return load_data_from_mongodb(model)
    else:
        st.error(f"Unknown database selection for loading: {st.session_state.selected_db}")
        return model, idx, txt_store

# Initialize clients at the start
dbx = initialize_dropbox()
init_mongo_client_global()

embedding_model, faiss_index, text_store = initialize_and_load_data()


# Function to save config as a downloadable JSON file
def save_config(config):
    json_bytes = json.dumps(config, indent=4).encode('utf-8')
    return BytesIO(json_bytes)

def load_config(uploaded_file):
    try:
        config_data = json.load(uploaded_file)
        st.session_state.config.update(config_data)
        st.sidebar.success("Configuration loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Failed to load configuration: {e}")

# -----------------------------------------------------------------------------
# PDF Processing Functions
# -----------------------------------------------------------------------------
def chunk_text(text, chunk_size=400, min_chunk_length=20):
    paragraphs = re.split(r'\n{2,}', text)
    chunks = []
    for para in paragraphs:
        sentences = re.split(r'(?<=[.!?])\s+', para)
        temp_chunk = ""
        for sentence in sentences:
            if len(temp_chunk) + len(sentence) < chunk_size:
                temp_chunk += sentence + " "
            else:
                cleaned_chunk = temp_chunk.strip()
                if len(cleaned_chunk) >= min_chunk_length:
                    chunks.append(cleaned_chunk)
                temp_chunk = sentence + " "
        cleaned_chunk = temp_chunk.strip()
        if len(cleaned_chunk) >= min_chunk_length:
            chunks.append(cleaned_chunk)
    return chunks

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def update_vector_db(texts, file_name, file_hash):
    global text_store, faiss_index, embedding_model
    embeddings = embedding_model.encode(texts)
    embeddings_np = np.array(embeddings).astype("float32")
    if embeddings_np.shape[0] > 0:
        faiss_index.add(embeddings_np)
    for text in texts:
        text_store.append({
            "text": text,
            "file_name": file_name,
            "file_hash": file_hash
        })
    save_data()

def process_pdf(file, file_name, file_hash):
    text = extract_text_from_pdf(file)
    chunks = chunk_text(text)
    if chunks:
        update_vector_db(chunks, file_name, file_hash)
    return chunks

# -----------------------------------------------------------------------------
# AI Generation Functions
# -----------------------------------------------------------------------------
def generate_response_gemini(prompt, context, temp, top_p):
    system_prompt = st.session_state.config["system_prompt"]
    input_parts = [system_prompt + "\n" + "Context: " + context, "Question: " + prompt]
    generation_config = genai.GenerationConfig(
        max_output_tokens=2048,
        temperature=temp,
        top_p=top_p,
        top_k=32
    )
    retries = 3
    for attempt in range(retries):
        try:
            response = gemini_model.generate_content(input_parts, generation_config=generation_config)
            return response.text
        except exceptions.ResourceExhausted:
            st.warning(f"Gemini API quota exceeded. Retrying... ({attempt+1}/{retries})")
            time.sleep(5)
        except Exception as e:
            return f"Error generating Gemini response: {str(e)}"
    st.error("Gemini API quota exceeded after retries. Please try again later.")
    return "Error generating Gemini response."

def generate_response_together(prompt, context, model, temp, top_p):
    system_prompt = st.session_state.config["system_prompt"]
    try:
        response = together_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": f"{system_prompt}"},
                {"role": "user", "content": f"Context: {context}. Question: {prompt}"}
            ],
            temperature=temp,
            top_p=top_p
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating Together AI response: {str(e)}"

def generate_response_openai(prompt, context, temp, top_p):
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": st.session_state.config["system_prompt"]},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {prompt}"}
            ],
            temperature=temp,
            top_p=top_p
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating OpenAI response: {str(e)}"

# -----------------------------------------------------------------------------
# Retrieval Function (RAG)
# -----------------------------------------------------------------------------
def retrieve_context(query, top_k=20):
    global text_store, faiss_index, embedding_model
    if faiss_index.ntotal == 0:
        return "No documents have been indexed yet. Please upload or scrape PDFs."
    try:
        query_embedding = embedding_model.encode([query])
        distances, indices = faiss_index.search(query_embedding.astype(np.float32), top_k)
        valid_indices = [i for i in indices[0] if 0 <= i < len(text_store)]
        retrieved_texts = [text_store[idx]["text"] for idx in valid_indices]
        return "\n\n".join(retrieved_texts) if retrieved_texts else "No relevant context found in the documents."
    except Exception as e:
        st.error(f"Error during context retrieval: {e}")
        return "Error retrieving context."

# ========= PDFs Link Extraction via URL =========
async def fetch_page(url):
    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=None)
        return response.text, str(response.url)

async def extract_info(url):
    html, base_url = await fetch_page(url)
    soup = BeautifulSoup(html, "html.parser")
    pdf_links = [urljoin(base_url, a["href"]) for a in soup.find_all("a", href=True) if ".pdf" in a["href"].lower()]
    return pdf_links

async def process_scraped_links(urls):
    results = await asyncio.gather(*[extract_info(url) for url in urls])
    return results

def get_page_items(url, base_url, listing_endpoint):
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code != 200:
            return []
        soup = BeautifulSoup(response.text, "html.parser")
        items = set()
        for item in soup.find_all("a"):
            link = item.get("href")
            if link and f"/{listing_endpoint}/" in link and link != url and not link.endswith("/rss"):
                if not link.startswith("http"):
                    link = urljoin(base_url, link)
                items.add(link)
        return list(items)
    except Exception as e:
        logging.error(f"Error scraping {url}: {e}")
        return []

def get_all_items(base_url, listing_endpoint, pagination_format, num_pages):
    all_items = set()
    for page in range(1, num_pages + 1):
        url = f"{base_url}/{listing_endpoint}/{pagination_format}{page}"
        page_items = get_page_items(url, base_url, listing_endpoint)
        if not page_items:
            break
        all_items.update(page_items)
    return list(all_items)

# -----------------------------------------------------------------------------
# File Deletion Functions
# -----------------------------------------------------------------------------
def delete_pdf(file_hash):
    global text_store, faiss_index, embedding_model
    try:
        new_text_store = [item for item in text_store if item["file_hash"] != file_hash]
        if len(new_text_store) == len(text_store):
            st.sidebar.warning(f"No file found with hash {file_hash} to delete.")
            return
        text_store = new_text_store
        if text_store:
            texts_for_reindex = [item["text"] for item in text_store]
            embeddings = embedding_model.encode(texts_for_reindex)
            embeddings_np = np.array(embeddings).astype("float32")
            new_faiss_index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
            if embeddings_np.shape[0] > 0:
                new_faiss_index.add(embeddings_np)
            faiss_index = new_faiss_index
        else:
            faiss_index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
        save_data()
        st.sidebar.success(f"File (hash: {file_hash}) and its data deleted successfully!")
        st.rerun()
    except Exception as e:
        st.error(f"Error deleting PDF data: {e}")

# -----------------------------------------------------------------------------
# Store Scraped PDFs
# -----------------------------------------------------------------------------
async def store_scraped_pdfs_in_db(pdf_links):
    global text_store
    async with aiohttp.ClientSession() as session:
        processed_count = 0
        skipped_count = 0
        existing_file_hashes = {item["file_hash"] for item in text_store}
        for pdf_link in pdf_links:
            try:
                async with session.get(pdf_link) as response:
                    if response.status == 200:
                        pdf_bytes = await response.read()
                        file_hash = hashlib.md5(pdf_bytes).hexdigest()
                        if file_hash not in existing_file_hashes:
                            pdf_file_like = BytesIO(pdf_bytes)
                            filename = os.path.basename(pdf_link)
                            process_pdf(pdf_file_like, filename, file_hash)
                            existing_file_hashes.add(file_hash)
                            processed_count += 1
                            st.success(f"Processed and stored PDF: {filename}")
                        else:
                            skipped_count +=1
                            st.info(f"Skipped (already exists): {os.path.basename(pdf_link)}")
                    else:
                        st.error(f"Failed to download PDF: {pdf_link} (Status: {response.status})")
            except Exception as e:
                st.error(f"Error processing {pdf_link}: {e}")
    st.success(f"Finished processing PDF links. Processed: {processed_count}, Skipped: {skipped_count}.")


# =================== Streamlit UI ============================
is_dark_mode = st.get_option("theme.base") == "dark"
background_color = "#1E1E1E" if is_dark_mode else "#f9f9f9"
border_color = "#BB86FC" if is_dark_mode else "#fc0303"
text_color_chat = "#E0E0E0" if is_dark_mode else "#000000"
user_background = "#333" if is_dark_mode else "#e3f2fd"
user_text_color = "#FFF" if is_dark_mode else "#000"

st.title("📄 IntelLaw")

with st.sidebar:
    st.header("Database Selection")
    db_options = ["Dropbox", "MongoDB"]
    current_db_index = db_options.index(st.session_state.selected_db) if st.session_state.selected_db in db_options else 0

    if not MONGO_URI:
        st.session_state.selected_db = "Dropbox" # Force Dropbox if no URI
        st.radio(
            "Select Database Backend",
            db_options,
            index=db_options.index("Dropbox"), # Default to Dropbox
            disabled=True,
            help="MongoDB URI not configured in .env file. Only Dropbox is available."
        )
        if 'mongo_uri_missing_warned' not in st.session_state:
            st.warning("MongoDB URI not set. Using Dropbox. Configure MONGO_URI in .env to enable MongoDB.")
            st.session_state.mongo_uri_missing_warned = True
    else:
        new_selected_db = st.radio(
            "Select Database Backend",
            db_options,
            index=current_db_index
        )
        if new_selected_db != st.session_state.selected_db:
            st.session_state.selected_db = new_selected_db
            embedding_model, faiss_index, text_store = initialize_and_load_data()
            st.rerun()

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Configuration", "Web Scraper", "Stored Files"])

    with tab1:
        st.header("Configuration")
        st.session_state.config["selected_models"] = st.multiselect(
            "Select AI Models (Up to 3)",
            AVAILABLE_MODELS,
            default=st.session_state.config["selected_models"],
        )
        with st.expander("Model Pricing"):
            for model, details in AVAILABLE_MODELS_DICT.items():
                st.write(f"**{model.split('/')[-1]}**: {details['price']}")

        st.session_state.config["vary_temperature"] = st.checkbox("Vary Temperature", value=st.session_state.config.get("vary_temperature", False))
        st.session_state.config["vary_top_p"] = st.checkbox("Vary Top-P", value=st.session_state.config.get("vary_top_p", False))
        st.session_state.config["temperature"] = st.slider("Temperature", 0.0, 1.0, value=st.session_state.config.get("temperature", 0.5), step=0.05)
        st.session_state.config["top_p"] = st.slider("Top-P", 0.0, 1.0, value=st.session_state.config.get("top_p", 0.5), step = 0.05)
        st.session_state.config["system_prompt"] = st.text_area("System Prompt", value=st.session_state.config.get("system_prompt", ""))

        if "config_uploader_key" not in st.session_state:
            st.session_state.config_uploader_key = 0
        config_file = st.file_uploader("Upload Configuration", type=['json'], key=f"config_uploader_{st.session_state.config_uploader_key}")
        if config_file:
            load_config(config_file)
            st.session_state.config_uploader_key += 1
            st.rerun()
        st.download_button("Download Config", data=save_config(st.session_state.config), file_name="config.json", mime="application/json")

    with tab2:
        st.header("Web Scraper")
        base_url = st.text_input("Enter Base URL", "https://www.imy.se")
        listing_endpoint = st.text_input("Enter Listing Endpoint", "tillsyner")
        pagination_format = st.text_input("Enter Pagination Format", "?query=&page=")
        num_pages = st.number_input("Enter Number of Pages", 1, 20, 3)

        if st.button("Start Scraping and Store PDFs"):
            if sys.platform == "win32" and isinstance(asyncio.get_event_loop_policy(), asyncio.WindowsSelectorEventLoopPolicy):
                 asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

            all_page_urls = []
            with st.spinner("Gathering page links..."):
                all_page_urls = get_all_items(base_url, listing_endpoint, pagination_format, num_pages)

            if all_page_urls:
                st.success(f"Found {len(all_page_urls)} potential pages with PDF links.")
                extracted_pdf_links_nested = []
                with st.spinner("Extracting PDF links from pages..."):
                    extracted_pdf_links_nested = asyncio.run(process_scraped_links(all_page_urls))
                unique_pdf_links = set()
                for link_list in extracted_pdf_links_nested:
                    for link in link_list:
                        unique_pdf_links.add(link)
                final_pdf_links = list(unique_pdf_links)
                if final_pdf_links:
                    st.write(f"**Found {len(final_pdf_links)} unique PDF links to process:**")
                    with st.spinner("Downloading and storing PDFs in selected database..."):
                        asyncio.run(store_scraped_pdfs_in_db(final_pdf_links))
                else:
                    st.warning("No PDF links found on the scraped pages.")
            else:
                st.warning("No pages found from initial scraping.")

    with tab3:
        st.subheader(f"📂 Stored Files in {st.session_state.selected_db}")
        if text_store:
            unique_files_display = {}
            for item in text_store:
                if item["file_hash"] not in unique_files_display:
                    unique_files_display[item["file_hash"]] = item["file_name"]
            if unique_files_display:
                for file_hash, file_name_to_display in unique_files_display.items():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(file_name_to_display)
                    with col2:
                        if st.button("🗑️", key=f"delete_{file_hash}_{st.session_state.selected_db}"):
                            delete_pdf(file_hash)
            else:
                st.write(f"No distinct files found in the current text store for {st.session_state.selected_db}.")
        else:
            st.write(f"No documents uploaded or processed yet for {st.session_state.selected_db}.")

st.header("📤 Upload PDFs")
if "file_uploader_key" not in st.session_state:
    st.session_state.file_uploader_key = 0
uploaded_files = st.file_uploader(
    "Upload PDFs to selected database",
    type="pdf",
    accept_multiple_files=True,
    key=f"file_uploader_{st.session_state.file_uploader_key}_{st.session_state.selected_db}"
)
if uploaded_files:
    files_processed_this_run = False
    existing_file_hashes = {item["file_hash"] for item in text_store}
    for file in uploaded_files:
        file_name = file.name
        file_bytes = file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()
        if file_hash not in existing_file_hashes:
            with io.BytesIO(file_bytes) as pdf_file_like:
                process_pdf(pdf_file_like, file_name, file_hash)
            st.success(f"Processed and stored '{file_name}' in {st.session_state.selected_db}.")
            existing_file_hashes.add(file_hash)
            files_processed_this_run = True
        else:
            st.info(f"File '{file_name}' has already been processed and exists in the current view.")
    if files_processed_this_run:
        st.session_state.file_uploader_key += 1
        st.rerun()

st.header("💬 Chat with Documents")
if prompt := st.chat_input("Ask a question about the documents..."):
    try:
        lang = detect(prompt)
    except Exception:
        lang = "en"
    with st.spinner("Retrieving relevant context..."):
        context = retrieve_context(prompt)
    st.markdown(f"""
        <div style="
            border: 2px solid {border_color}; padding: 10px; border-radius: 10px;
            background-color: {user_background}; color: {user_text_color}; margin-bottom: 10px;">
            <strong>User:</strong> {prompt}
        </div>
    """, unsafe_allow_html=True)
    if not st.session_state.config["selected_models"]:
        st.warning("Please select at least one AI model from the Configuration sidebar.")
    else:
        model_tabs = st.tabs([model.split("/")[-1] for model in st.session_state.config["selected_models"]])
        temp_values = [0.1, st.session_state.config["temperature"] / 2, st.session_state.config["temperature"]] if st.session_state.config["vary_temperature"] else [st.session_state.config["temperature"]]
        top_p_values = [0.1, st.session_state.config["top_p"] / 2, st.session_state.config["top_p"]] if st.session_state.config["vary_top_p"] else [st.session_state.config["top_p"]]
        temp_values = [max(0.01, round(t,2)) for t in temp_values]
        top_p_values = [max(0.01, round(p,2)) for p in top_p_values]
        for tab, model_name_full in zip(model_tabs, st.session_state.config["selected_models"]):
            with tab:
                model_type = AVAILABLE_MODELS_DICT[model_name_full]["type"]
                for temp in temp_values:
                    for top_p in top_p_values:
                        response_text = ""
                        spinner_message = f"Generating response from {model_name_full.split('/')[-1]} (Temp={temp}, Top-P={top_p})..."
                        with st.spinner(spinner_message):
                            if model_type == "together":
                                response_text = generate_response_together(prompt, context, model_name_full, temp, top_p)
                            elif model_type == "gemini":
                                response_text = generate_response_gemini(prompt, context, temp, top_p)
                            elif model_type == "openai":
                                response_text = generate_response_openai(prompt, context, temp, top_p)
                        st.markdown(f"""
                            <div style="
                                border: 1px solid {border_color}; padding: 15px; border-radius: 10px;
                                background-color: {background_color}; color: {text_color_chat}; margin-top: 10px;">
                                <strong style="color:#4CAF50;">Model:</strong> {model_name_full.split('/')[-1]}<br>
                                <strong style="color:#FF9800;">Temp:</strong> {temp:.2f}, <strong style="color:#2196F3;">Top-P:</strong> {top_p:.2f}<br>
                                <hr style="border-color: {border_color};">
                                <strong>Response:</strong> {response_text}
                            </div>
                        """, unsafe_allow_html=True)
                        st.download_button(
                            label="Download Response",
                            data=response_text,
                            file_name=f"response_{model_name_full.split('/')[-1]}_T{temp:.2f}_P{top_p:.2f}.txt",
                            mime="text/plain",
                            key=f"download_{model_name_full}_{temp}_{top_p}"
                        )