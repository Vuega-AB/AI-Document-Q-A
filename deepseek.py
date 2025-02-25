import streamlit as st
import os
import requests
import io
import json
import re
import time
import asyncio
import aiohttp
import numpy as np
import faiss
from dotenv import load_dotenv
from io import BytesIO
from langdetect import detect
from sentence_transformers import SentenceTransformer
import PyPDF2
import dropbox
import openai
# from together import Together
from google.api_core import exceptions
import asyncio
import nest_asyncio
nest_asyncio.apply()
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
import dropbox
import hashlib
import openai
import time

# -----------------------------------------------------------------------------
# Environment and Client Initialization
# -----------------------------------------------------------------------------
load_dotenv()
# TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
DROPBOX_REFRESH_TOKEN = os.getenv("DROPBOX_REFRESH_TOKEN")
DROPBOX_APP_KEY = os.getenv("DROPBOX_APP_KEY")
DROPBOX_APP_SECRET = os.getenv("DROPBOX_APP_SECRET")

# Initialize Together.AI client
# together_client = Together(api_key=TOGETHER_API_KEY)

# Initialize models and configurations
INDEX_FILE = "faiss_index.index"
CONFIG_FILENAME = "config.json"
INDEX_FILE_DROPBOX = "/faiss_index.index"  # Path on Dropbox
TEXT_FILE_DROPBOX = "/text_store.json"  # Path on Dropbox
TOKEN_FILE = "dropbox_token.json"


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "config" not in st.session_state:
    st.session_state.config = {
        "temperature": 0.7,
        "top_p": 0.9,
        "system_prompt": "You are a helpful assistant. Answer questions strictly based on the provided context. If there is no context, say 'I don't have enough information to answer that.'",
        "stored_pdfs": [],
        "text_chunks": []
    }


def load_access_token():
    """Load access token from file if available and not expired."""
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "r") as file:
            data = json.load(file)
            return data.get("access_token"), data.get("expires_at")
    return None, None

def save_access_token(access_token, expires_in):
    """Save access token to file with expiry timestamp."""
    expires_at = int(time.time()) + expires_in - 60  # Buffer time to refresh before expiry
    with open(TOKEN_FILE, "w") as file:
        json.dump({"access_token": access_token, "expires_at": expires_at}, file)

def get_dropbox_access_token():
    """Fetch a new Dropbox access token using the refresh token."""
    response = requests.post(
        "https://api.dropbox.com/oauth2/token",
        data={
            "grant_type": "refresh_token",
            "refresh_token": DROPBOX_REFRESH_TOKEN,
        },
        auth=(DROPBOX_APP_KEY, DROPBOX_APP_SECRET),
    )
    
    if response.status_code == 200:
        data = response.json()
        access_token = data["access_token"]
        expires_in = data.get("expires_in", 14400)  # Default to 4 hours if not provided
        print(data)
        save_access_token(access_token, expires_in)
        return access_token
    else:
        raise Exception(f"Failed to refresh token: {response.text}")

def get_valid_access_token():
    """Retrieve a valid access token, refreshing if necessary."""
    import time
    access_token, expires_at = load_access_token()
    if access_token and expires_at and int(time.time()) < expires_at:
        return access_token
    return get_dropbox_access_token()

def initialize_dropbox():
    """Initialize Dropbox client with a valid access token."""
    try:
        access_token = get_valid_access_token()
        dbx = dropbox.Dropbox(access_token)
        return dbx
    except Exception as e:
        st.error(f"Error connecting to Dropbox: {e}")
        return None

dbx = initialize_dropbox()

def initialize_and_load_data():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.IndexFlatL2(384)  # Initialize a new index (in memory)
    text_store = []
    try:
        with open(INDEX_FILE, "wb") as f_index:
            metadata_index, res_index = dbx.files_download(path=INDEX_FILE_DROPBOX)
            f_index.write(res_index.content)
        index = faiss.read_index(INDEX_FILE)

        # Load text store from Dropbox directly into memory
        metadata_text, res_text = dbx.files_download(path=TEXT_FILE_DROPBOX)
        text_store = json.loads(res_text.content.decode('utf-8'))

        # st.info("Data loaded from Dropbox.")
    except Exception as e:
        print(e)
        

    return model, index, text_store

embedding_model, faiss_index, text_store = initialize_and_load_data()

def save_data_to_dropbox():
    global text_store
    try:
        with open(INDEX_FILE, "rb") as f:
            dbx.files_upload(f.read(), INDEX_FILE_DROPBOX, mode=dropbox.files.WriteMode.overwrite)

        # Save text store directly to Dropbox
        text_json = json.dumps(text_store, ensure_ascii=False, indent=4).encode('utf-8')
        dbx.files_upload(text_json, TEXT_FILE_DROPBOX, mode=dropbox.files.WriteMode.overwrite)

        # st.success("Data saved to Dropbox.")
    except Exception as e:
        st.error(f"Error saving data to Dropbox: {e}")


# -----------------------------------------------------------------------------
# Model Definitions and Session State
# -----------------------------------------------------------------------------
# Define models with their pricing and type (together vs gemini)
# AVAILABLE_MODELS_DICT = {
#     "meta-llama/Llama-3.3-70B-Instruct-Turbo": {"price": "$0.88", "type": "together"},
#     "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": {"price": "$3.50", "type": "together"},
#     "databricks/dbrx-instruct": {"price": "$1.20", "type": "together"},
#     "microsoft/WizardLM-2-8x22B": {"price": "$1.20", "type": "together"},
#     "mistralai/Mixtral-8x22B-Instruct-v0.1": {"price": "$1.20", "type": "together"},
#     "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO": {"price": "$0.60", "type": "together"},
#     "gemini-2.0-flash": {"price": "Custom", "type": "gemini"}
# }
# AVAILABLE_MODELS = list(AVAILABLE_MODELS_DICT.keys())

# if "messages" not in st.session_state:
#     st.session_state.messages = []
# if "config" not in st.session_state:
#     st.session_state.config = {
#         "temperature": 0.7,
#         "top_p": 0.9,
#         "system_prompt": "You are a helpful assistant. Answer questions based on the provided context.",
#         "selected_models": AVAILABLE_MODELS[:3],
#         "vary_temperature": True,
#         "vary_top_p": True
#     }

# -----------------------------------------------------------------------------
# Configuration Save & Load
# -----------------------------------------------------------------------------
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
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def chunk_text(text, chunk_size=400, min_chunk_length=20):
    paragraphs = re.split(r'\n{2,}', text)  # Split by paragraphs
    chunks = []

    for para in paragraphs:
        sentences = re.split(r'(?<=[.!?])\s+', para)  # Split paragraph into sentences
        temp_chunk = ""

        for sentence in sentences:
            if len(temp_chunk) + len(sentence) < chunk_size:
                temp_chunk += sentence + " "
            else:
                cleaned_chunk = temp_chunk.strip()
                if len(cleaned_chunk) >= min_chunk_length:  # Remove very short chunks
                    chunks.append(cleaned_chunk)
                temp_chunk = sentence + " "  # Start a new chunk
        
        cleaned_chunk = temp_chunk.strip()
        if len(cleaned_chunk) >= min_chunk_length:  # Append remaining text if valid
            chunks.append(cleaned_chunk)

    return chunks

def process_pdf(file, file_name, file_hash):
    text = extract_text_from_pdf(file)
    chunks = chunk_text(text)
    update_vector_db(chunks, file_name, file_hash)
    save_data_to_dropbox()
    return chunks

def update_vector_db(texts, file_name, file_hash):
    global text_store, faiss_index
    embeddings = embedding_model.encode(texts)
    embeddings = np.array(embeddings).astype("float32")

    start_idx = len(text_store)
    faiss_index.add(embeddings)

    for text in texts:
        text_store.append({
            "text": text,
            "file_name": file_name,
            "file_hash": file_hash
        })

    faiss.write_index(faiss_index, INDEX_FILE)

# -----------------------------------------------------------------------------
# Retrieval Function (RAG)
# -----------------------------------------------------------------------------
def retrieve_context(query, top_k=20):
    global text_store, faiss_index
    query_embedding = embedding_model.encode([query])
    distances, indices = faiss_index.search(query_embedding, top_k)
    valid_indices = [i for i in indices[0] if i != -1 and i < len(text_store)]
    retrieved_texts = [text_store[idx]["text"] for idx in valid_indices]
    return "\n\n".join(retrieved_texts) if retrieved_texts else "No relevant context found."

# -----------------------------------------------------------------------------
# AI Generation Functions
# -----------------------------------------------------------------------------
def generate_response(prompt, context):
    try:        
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "developer", "content": st.session_state.config["system_prompt"]},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {prompt}"}
            ],
            temperature=st.session_state.config["temperature"],
            top_p=st.session_state.config["top_p"]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

# -----------------------------------------------------------------------------
# URL PDF Extraction (Asynchronous)
# -----------------------------------------------------------------------------
async def fetch_and_process_pdf_links(url: str):
    browser_config = BrowserConfig()
    run_config = CrawlerRunConfig(remove_overlay_elements=True)
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=url, config=run_config)
        internal_links = result.links.get("internal", [])
        pdf_links = []
        for link in internal_links:
            href = ""
            if isinstance(link, dict):
                href = link.get("href", "")
            elif isinstance(link, str):
                href = link
            if ".pdf" in href.lower():
                pdf_links.append(href)
        if not pdf_links:
            st.info("No PDF links found on the provided URL.")
            return
        async with aiohttp.ClientSession() as session:
            unique_file_hashes = set(item["file_hash"] for item in text_store)
            for pdf_link in pdf_links:
                try:
                    async with session.get(pdf_link) as response:
                        if response.status == 200:
                            pdf_bytes = await response.read()
                            
                            file_hash = hashlib.md5(pdf_bytes).hexdigest()
                            if file_hash not in unique_file_hashes:
                                pdf_file = BytesIO(pdf_bytes)
                                filename = os.path.basename(pdf_link)
                                process_pdf(pdf_file, filename)
                                st.success(f"Processed PDF: {filename}")
                                unique_file_hashes.add(file_hash)
                        else:
                            st.error(f"Failed to download PDF: {pdf_link}")
                except Exception as e:
                    st.error(f"Error processing {pdf_link}: {e}")
        st.success("Finished processing all PDF links.")

def process_pdf_links_from_url_sync(url: str):
    loop = asyncio.ProactorEventLoop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(fetch_and_process_pdf_links(url))

# -----------------------------------------------------------------------------
# File Deletion Functions
# -----------------------------------------------------------------------------
def delete_pdf(file_hash):
    global text_store, faiss_index

    try:
        indices_to_remove = [i for i, item in enumerate(text_store) if item["file_hash"] == file_hash]

        # Remove items from text_store
        for index in sorted(indices_to_remove, reverse=True):
            del text_store[index]

        texts = [item["text"] for item in text_store]
        if texts:
            embeddings = embedding_model.encode(texts)
            embeddings = np.array(embeddings).astype("float32")
            faiss_index = faiss.IndexFlatL2(384)
            faiss_index.add(embeddings)
        else:
            faiss_index = faiss.IndexFlatL2(384)

        faiss.write_index(faiss_index, INDEX_FILE)

        save_data_to_dropbox()
        st.sidebar.success(f"PDF file deleted successfully!")
        st.rerun()  # Force a rerun to update the UI immediately

    except Exception as e:
        st.error(f"Error deleting PDF: {e}")

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
st.title("📄 AI Document Q&A with Gemini & Together.AI")

# Sidebar: Configuration
with st.sidebar:
    st.header("Configuration")
    # st.session_state.config["selected_models"] = st.multiselect(
    #     "Select AI Models (Up to 3)",
    #     AVAILABLE_MODELS,
    #     default=AVAILABLE_MODELS[:3]
    # )
    # with st.expander("Model Pricing"):
    #     for model, details in AVAILABLE_MODELS_DICT.items():
    #         st.write(f"**{model.split('/')[-1]}**: {details['price']}")
    st.session_state.config["temperature"] = st.slider("Temperature", 0.0, 1.0, st.session_state.config["temperature"], 0.05)
    st.session_state.config["top_p"] = st.slider("Top-P", 0.0, 1.0, st.session_state.config["top_p"], 0.05)
    st.session_state.config["system_prompt"] = st.text_area("System Prompt", value=st.session_state.config["system_prompt"])
    # st.session_state.config["vary_temperature"] = st.checkbox("Vary Temperature", value=st.session_state.config.get("vary_temperature", True))
    # st.session_state.config["vary_top_p"] = st.checkbox("Vary Top-P", value=st.session_state.config.get("vary_top_p", True))
    config_file = st.file_uploader("Upload Configuration", type=['json'])
    if config_file:
        load_config(config_file)
    if st.button("Update and Download Configuration"):
        config_bytes = save_config(st.session_state.config)
        st.download_button("Download Config", data=config_bytes, file_name="config.json", mime="application/json")

    st.header("Uploaded Documents")
    if text_store:
        unique_file_hashes = set(item["file_hash"] for item in text_store)

        for file_hash in unique_file_hashes:
            file_name_to_display = "Unknown"
            for item in text_store:
                if item["file_hash"] == file_hash:
                    file_name_to_display = item["file_name"]
                    break

            col1, col2 = st.columns([3, 1])

            with col1:
                st.write(file_name_to_display)

            with col2:
                if st.button("🗑️", key=f"delete_{file_hash}"):
                    delete_pdf(file_hash)
    else:
        st.write("No documents uploaded yet.")

# File Uploader for PDFs
st.header("📤 Upload PDFs")
# Initialize the file uploader with a unique key
if "file_uploader_key" not in st.session_state:
    st.session_state.file_uploader_key = 0

uploaded_files = st.file_uploader(
    "Upload PDFs", 
    type="pdf", 
    accept_multiple_files=True, 
    key=f"file_uploader_{st.session_state.file_uploader_key}"
)
if uploaded_files:
    for file in uploaded_files:
        file_name = file.name
        unique_file_hashes = set(item["file_hash"] for item in text_store)
        file_hash = hashlib.md5(file.getvalue()).hexdigest()
        if file_hash not in unique_file_hashes:
            with io.BytesIO(file.getvalue()) as pdf_file:
                process_pdf(pdf_file, file_name, file_hash)
            # st.success(f"Processed '{file_name}'")
        else:
            st.info(f"File '{file_name}' has already been processed.")
    # Reset the file uploader by incrementing the key
    st.session_state.file_uploader_key += 1
    st.rerun()  # Force a rerun to update the UI immediately

# PDF Link Extraction via URL
st.header("🔗 Add PDFs via URL")
pdf_url = st.text_input("Enter a URL to crawl for PDFs:")
if st.button("Extract PDFs from URL"):
    if pdf_url:
        process_pdf_links_from_url_sync(pdf_url)
        st.rerun()
    else:
        st.warning("Please enter a valid URL.")

# Chat Interface
st.header("💬 Chat with Documents")
for message in st.session_state.get("messages", []):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question"):
    try:
        lang = detect(prompt)
    except Exception:
        lang = "en"
    retrieved_context = retrieve_context(prompt)
    
    # Define temperature and top-p values based on configuration flags
    # if st.session_state.config.get("vary_temperature", True):
    #     temp_values = [st.session_state.config["temperature"] / 3, st.session_state.config["temperature"]]
    # else:
    #     temp_values = [st.session_state.config["temperature"]]
    
    # if st.session_state.config.get("vary_top_p", True):
    #     top_p_values = [st.session_state.config["top_p"] / 3, st.session_state.config["top_p"]]
    # else:
    #     top_p_values = [st.session_state.config["top_p"]]
    
    # Create a tab for each selected model
    # tabs = st.tabs([model.split("/")[-1] for model in st.session_state.config["selected_models"]])
    # responses = {}
    
    # for tab, model in zip(tabs, st.session_state.config["selected_models"]):
    #     with tab:
    #         model_type = AVAILABLE_MODELS_DICT[model]["type"]
    #         model_responses = []
    #         for temp in temp_values:
    #             for top_p in top_p_values:
    #                 with st.spinner(f"Generating response from {model} (Temp={temp}, Top-P={top_p})..."):
    #                     if model_type == "together":
    #                         resp = generate_response_together(prompt, context, model, temp, top_p)
    #                     elif model_type == "gemini":
    #                         resp = generate_response_gemini(prompt, context, temp, top_p)
    #                     model_responses.append(f"Temp={temp}, Top-P={top_p}: {resp}")
    #         response_text = "\n\n".join(model_responses)
    #         st.markdown(f"""
    #             <div style="
    #                 border: 2px solid #fc0303;
    #                 padding: 15px;
    #                 border-radius: 10px;
    #                 background-color: #f9f9f9;
    #                 margin-top: 10px;">
    #                 <strong style="color:#4CAF50;">Model:</strong> {model}<br>
    #                 <strong style="color:#FF9800;">Responses:</strong><br>{response_text}
    #             </div>
    #         """, unsafe_allow_html=True)
    #         responses[model] = model_responses

    with st.spinner("Generating response..."):
        response = generate_response(prompt, retrieved_context)
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        st.markdown(response)
