import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
import asyncio
import nest_asyncio
nest_asyncio.apply()
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from crawl4ai import AsyncWebCrawler
import aiohttp
import asyncio
import PyPDF2
import io
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langdetect import detect
import json
from dotenv import load_dotenv
from io import BytesIO  
import re
import subprocess
import logging
from openai import OpenAI
import sys
import asyncio
import httpx
import aiofiles
from urllib.parse import urljoin
import dropbox
import hashlib
import openai
import time
# ================== Environment Variables ==================
load_dotenv()
DROPBOX_REFRESH_TOKEN = os.getenv("DROPBOX_REFRESH_TOKEN")
DROPBOX_APP_KEY = os.getenv("DROPBOX_APP_KEY")
DROPBOX_APP_SECRET = os.getenv("DROPBOX_APP_SECRET")
# Initialize models and configurations
INDEX_FILE = "faiss_index.index"
CONFIG_FILENAME = "config.json"
INDEX_FILE_DROPBOX = "/faiss_index.index"  # Path on Dropbox
TEXT_FILE_DROPBOX = "/text_store.json"  # Path on Dropbox
TOKEN_FILE = "dropbox_token.json"

# =================== Connections ============================
# mongo_client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
# db = mongo_client["userembeddings"]
# collection = db["embeddings"]
# client = Together(api_key=API_KEY)

# # Available Together.AI models
# MODEL_PRICING = {
#     "meta-llama/Llama-3.3-70B-Instruct-Turbo": "$0.88",
#     "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": "$3.50",
#     "databricks/dbrx-instruct": "$1.20",
#     "microsoft/WizardLM-2-8x22B": "$1.20",
#     "mistralai/Mixtral-8x22B-Instruct-v0.1": "$1.20",
#     "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO": "$0.60",
    
# }

# AVAILABLE_MODELS = list(MODEL_PRICING.keys())

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "config" not in st.session_state:
    st.session_state.config = {
        "temperature": 0.7,
        "top_p": 0.9,
        "system_prompt": "You are a helpful assistant. Answer questions based on the provided context.",
        "text_chunks": [],
        # "selected_models": AVAILABLE_MODELS[:3],
        # "vary_temperature": True,
        # "vary_top_p": False
    }

# Function to save config as a downloadable JSON file
def save_config(config):
    json_bytes = json.dumps(config, indent=4).encode('utf-8')
    return BytesIO(json_bytes)

# Function to load config from an uploaded JSON file
def load_config(uploaded_file):
    try:
        config_data = json.load(uploaded_file)
        st.session_state.config.update(config_data)
        st.sidebar.success("Configuration loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Failed to load configuration: {e}")
# ================== dropbox ==================

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

# ================== FAISS and Sentence Transformer Initialization ==================
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

# # Load stored text chunks
# text_chunks_file = "text_chunks.json"
# if os.path.exists(text_chunks_file):
#     with open(text_chunks_file, "r") as f:
#         st.session_state.config["text_chunks"] = json.load(f)

# ================== Helper Functions ==================

# PDF Processing Functions
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def process_pdf(file, file_name, file_hash):
    text = extract_text_from_pdf(file)
    chunks = chunk_text(text)
    update_vector_db(chunks, file_name, file_hash)
    save_data_to_dropbox()
    return chunks

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

# ================== Database Connection ==================
# RAG Pipeline
def retrieve_context(query, top_k=20):
    global text_store, faiss_index
    query_embedding = embedding_model.encode([query])
    distances, indices = faiss_index.search(query_embedding, top_k)
    valid_indices = [i for i in indices[0] if i != -1 and i < len(text_store)]
    retrieved_texts = [text_store[idx]["text"] for idx in valid_indices]
    return "\n\n".join(retrieved_texts) if retrieved_texts else "No relevant context found."


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
# ================== delete  ==================

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

# ========= PDFs Link Extraction via URL =========
try:
    import playwright
    subprocess.run(["playwright", "install"], check=True)
except Exception as e:
    print(f"Error installing Playwright: {e}")

async def fetch_and_process_pdf_links(url):
    """Scrapes a webpage, extracts text, finds PDFs, and summarizes content."""
    # try:
    browser_config = BrowserConfig(
        browser_type="chromium",  # Use Chromium for compatibility
        headless=True,  # Run in headless mode for Streamlit
        use_managed_browser=False,  # Disable managed mode to prevent conflicts
        debugging_port=None,  # No debugging port needed
        proxy=None,  # Disable proxy unless explicitly required
        text_mode=True,  # Optimize for text scraping (faster)
        light_mode=True,  # Further performance optimizations
        verbose=True,  # Enable logging for debugging
        ignore_https_errors=True,  # Avoid SSL certificate issues
        java_script_enabled=True  # Enable JS for dynamic content
    )

    run_config = CrawlerRunConfig(remove_overlay_elements=True)

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=url, config=run_config)

        # Extract Text from Page
        soup = BeautifulSoup(result.html, "html.parser")
        paragraphs = "\n".join([p.get_text() for p in soup.find_all("p")])
        summarized_text = summarize_text(paragraphs) if paragraphs else "No text available to summarize."

        # Extract PDFs
        internal_links = result.links.get("internal", [])
        pdf_links = [link['href'] for link in internal_links if '.pdf' in link['href'].lower()]
        
        print("Number of found PDFs: ", len(pdf_links))

        # Download PDFs
        extracted_texts = []
        if pdf_links:
            async with aiohttp.ClientSession() as session:
                for i, link in enumerate(pdf_links):
                    pdf_path = f"document_{i}.pdf"
                    saved_path = await download_pdf(link, session, pdf_path)
                    if saved_path:
                        extracted_texts.append(extract_text_from_pdf(saved_path))

        return summarized_text, pdf_links, extracted_texts

async def download_pdf(url, session, save_path):
    """Downloads a PDF file asynchronously."""
    try:
        async with session.get(url) as response:
            if response.status == 200:
                with open(save_path, 'wb') as f:
                    f.write(await response.read())
                return save_path
    except Exception as e:
        logging.error(f"Error downloading {url}: {e}")
    return None

def process_pdf_links_from_url_sync(url: str):
    # Replace asyncio.get_event_loop() with ProactorEventLoop as requested
    loop = asyncio.ProactorEventLoop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(fetch_and_process_pdf_links(url))

def get_page_items(url, base_url, listing_endpoint):
    """Extracts all item links from a page."""
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code != 200:
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        items = set()

        for item in soup.find_all("a"):
            link = item.get("href")
            title = item.text.strip()
            if link and f"/{listing_endpoint}/" in link and link != url and not link.endswith("/rss"):
                if not link.startswith("http"):
                    link = base_url + link
                items.add((title, link))

        return list(items)
    except Exception as e:
        logging.error(f"Error scraping {url}: {e}")
        return []


def get_all_items(base_url, listing_endpoint, pagination_format, num_pages):
    """Scrapes multiple pages to collect all links."""
    all_items = set()

    for page in range(1, num_pages + 1):
        url = f"{base_url}/{listing_endpoint}/{pagination_format}{page}"
        items = get_page_items(url, base_url, listing_endpoint)
        if not items:
            break
        
        all_items.update(items)
        st.write(f"Scraped page {page}")

    return list(all_items)


def summarize_text(text):
    """Summarizes extracted text using OpenAI."""
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Summarize the following text into a concise paragraph."},
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error in summarization: {e}")
        return "Summarization failed."

async def run_scraper(items):
    for _, link in items:
        print(link) 
    tasks = [fetch_and_process_pdf_links(link) for _, link in items]
    results = await asyncio.gather(*tasks)
    return results

# =================== Try another way ============================


async def fetch_page(url):
    """Fetch page content asynchronously."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=None)
        return response.text, str(response.url)  # Return HTML + Base URL


async def download_with_retries(url, retries=3, delay=5):
    for attempt in range(retries):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=60.0)
                response.raise_for_status()  # Ensure the request was successful
                return response.content
        except (httpx.ReadTimeout, httpx.RequestError) as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(delay)  # Wait before retrying
    raise Exception(f"Failed to download {url} after {retries} attempts")

async def extract_info(url):
    """Extract text and PDF links from a webpage."""
    html, base_url = await fetch_page(url)
    soup = BeautifulSoup(html, "html.parser")

    # Extract paragraphs
    paragraphs = "\n".join(p.get_text() for p in soup.find_all("p"))
    print(f"Text from {url}:\n{paragraphs[:300]}...")

    # Extract PDF links and convert relative to absolute URLs
    pdf_links = [urljoin(base_url, a["href"]) for a in soup.find_all("a", href=True) if ".pdf" in a["href"].lower()]
    print(f"Found {len(pdf_links)} PDF links on {url}")

    # Download and extract PDF content
    pdf_texts = []
    for link in pdf_links:
        pdf_texts.append(await download_and_extract_pdf(link))

    return paragraphs, pdf_links, pdf_texts

async def download_and_extract_pdf(url):
    """Download a PDF and extract text."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        filename = url.split("/")[-1]

        # Save PDF file
        async with aiofiles.open(filename, "wb") as f:
            await f.write(response.content)
        print(f"Downloaded: {filename}")

        # Extract text from PDF
        return extract_text_from_pdf(filename)

async def main(urls):
    """Scrape multiple pages concurrently."""
    results = await asyncio.gather(*[extract_info(url) for url in urls])
    return results


# =================== Streamlit UI ============================
st.title("📄 AI Document Q&A and Web Scraper")

# Sidebar with Tabs
with st.sidebar:
    tab1, tab2, tab3 = st.tabs(["Configuration", "Web Scraper", "Uploaded Documents"])

    with tab1:
        st.header("Configuration")
        st.session_state.config = {}
        # st.session_state.config["selected_models"] = st.multiselect(
        #     "Select AI Models (Up to 3)", 
        #     AVAILABLE_MODELS,
        #     default=AVAILABLE_MODELS[:3],
        # )
    
        # with st.expander("Model Pricing"):
        #     for model, price in MODEL_PRICING.items():
        #         st.write(f"**{model.split('/')[-1]}**: {price}")

        # # Grok-3 Integration
        # use_grok = st.checkbox("Use Grok-3 Model", value=True)
        # if use_grok:
        #     st.session_state.config["selected_models"].append("grok-3")

                
        # st.session_state.config["vary_temperature"] = st.checkbox("Vary Temperature", value=True)
        # st.session_state.config["vary_top_p"] = st.checkbox("Vary Top-P", value=False)
        st.session_state.config["temperature"] = st.slider("Temperature", 0.0, 1.0, 0.5, 0.05)
        st.session_state.config["top_p"] = st.slider("Top-P", 0.0, 1.0, 0.9, 0.05)
        st.session_state.config["system_prompt"] = st.text_area("System Prompt", value="You are an AI assistant.")

        config_file = st.file_uploader("Upload Configuration", type=['json'])
        if config_file:
            load_config(config_file)

        if st.button("Update and Download Configuration"):
            config_bytes = save_config(st.session_state.config)
            st.download_button("Download Config", data=config_bytes, file_name="config.json", mime="application/json")

    with tab2:
        st.header("Web Scraper")

        base_url = st.text_input("Enter Base URL", "https://www.imy.se")
        listing_endpoint = st.text_input("Enter Listing Endpoint", "tillsyner")
        pagination_format = st.text_input("Enter Pagination Format", "?query=&page=")
        num_pages = st.number_input("Enter Number of Pages", 1, 20, 3)

        if st.button("Start Scraping"):

            if sys.platform == "win32":
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

            with st.spinner("Scraping in progress..."):
                items = get_all_items(base_url, listing_endpoint, pagination_format, num_pages)

            if items:
                st.success(f"Found {len(items)} items!")

                # Debugging
                for title, link in items:
                    print(link)

                scrape_results = []
                urls = [link for _, link in items]
                with st.spinner("Scraping in progress..."):
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)

                    scrape_results = loop.run_until_complete(main(urls))
                    print(scrape_results)

                # Save summaries and extracted texts to files
                for i, (summary, pdf_links, extracted_texts) in enumerate(scrape_results):
                    summary_file = f"summaries/summary_{i+1}.txt"
                    text_file = f"texts/text_{i+1}.txt"

                    with open(summary_file, "w", encoding="utf-8") as sf:
                        sf.write(summary)

                    with open(text_file, "w", encoding="utf-8") as tf:
                        tf.write("\n".join(extracted_texts))

                    # st.download_button("Download Summary", data=open(summary_file, "rb"), file_name=summary_file)
                    # st.download_button("Download Extracted Text", data=open(text_file, "rb"), file_name=text_file)

                    if pdf_links:
                        st.write("**Extracted PDFs:**")
                        for pdf in pdf_links:
                            st.markdown(f"[Download PDF]({pdf})")
            else:
                st.warning("No items found.")

        with tab3:
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

# TODO: Database Connection
# File Uploader for PDFs
st.header("Upload PDFs")
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

# Chat UI with Multiple Models
st.header("Chat with Documents")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question (English/Swedish)"):
    try:
        lang = detect(prompt)
    except:
        lang = "en"
        
    context = retrieve_context(prompt)
    
    print(context)
    with st.spinner("Generating response..."):
        response = generate_response(prompt, context)
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        st.markdown(response)