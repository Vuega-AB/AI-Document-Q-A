import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
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
from together import Together
import re
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import subprocess
import logging
from openai import OpenAI
import sys
import asyncio
import httpx
import aiofiles
from urllib.parse import urljoin

# ================== Environment Variables ==================
load_dotenv()
API_KEY = os.getenv("TOGETHER_API_KEY")
API_KEY = os.getenv("GOOGLE_API_KEY")
MONGO_URI = os.getenv("MongoDB")
OpenAI_API_KEY = os.getenv("OPENAI_API_KEY")


# =================== Connections ============================
mongo_client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
db = mongo_client["userembeddings"]
collection = db["embeddings"]
client = Together(api_key=API_KEY)

# Available Together.AI models
MODEL_PRICING = {
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": "$0.88",
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": "$3.50",
    "databricks/dbrx-instruct": "$1.20",
    "microsoft/WizardLM-2-8x22B": "$1.20",
    "mistralai/Mixtral-8x22B-Instruct-v0.1": "$1.20",
    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO": "$0.60"
}

AVAILABLE_MODELS = list(MODEL_PRICING.keys())

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "config" not in st.session_state:
    st.session_state.config = {
        "temperature": 0.7,
        "top_p": 0.9,
        "system_prompt": "You are a helpful assistant. Answer questions based on the provided context.",
        "stored_pdfs": [],
        "text_chunks": [],
        "selected_models": AVAILABLE_MODELS[:3],
        "vary_temperature": True,
        "vary_top_p": False
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

# ================== FAISS and Sentence Transformer Initialization ==================
def initialize_vector_db():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index_file = "faiss_index.index"

    if os.path.exists(index_file):
        try:
            index = faiss.read_index(index_file)
            print("FAISS index loaded successfully!")
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            index = faiss.IndexFlatL2(384)
    else:
        index = faiss.IndexFlatL2(384)

    return model, index

# Initialize FAISS and embedding model at startup
embedding_model, faiss_index = initialize_vector_db()

# Load stored text chunks
text_chunks_file = "text_chunks.json"
if os.path.exists(text_chunks_file):
    with open(text_chunks_file, "r") as f:
        st.session_state.config["text_chunks"] = json.load(f)

# ================== Helper Functions ==================

# PDF Processing Functions
def extract_text(file):
    reader = PyPDF2.PdfReader(file)
    text = "".join([page.extract_text() + "\n" for page in reader.pages if page.extract_text()])
    return text

def process_pdf(file, filename):
    text = extract_text(file)
    # Chunk the text into pieces
    chunks = chunk_text(text)
    st.session_state.config["text_chunks"].extend(chunks)
    update_vector_db(chunks, filename)

# Together.AI Integration
def generate_response(prompt, context, model, temp, top_p):
    system_prompt = st.session_state.config["system_prompt"]
    try:
        response = client.chat.completions.create(
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
        return f"Error generating response: {str(e)}"

# ================== Database Connection ==================
# RAG Pipeline
def retrieve_context(query, top_k=15):
    query_embedding = embedding_model.encode([query]).tolist()[0]
    stored_docs = list(collection.find({}, {"_id": 0, "embedding": 1, "text": 1}))

    if not stored_docs:
        return []

    embeddings = np.array([doc["embedding"] for doc in stored_docs], dtype="float32")
    texts = [doc["text"] for doc in stored_docs]

    if faiss_index.ntotal == 0:
        faiss_index.add(np.array(embeddings, dtype="float32"))

    top_k = min(top_k, len(texts))  # Avoid requesting more results than available
    distances, indices = faiss_index.search(np.array([query_embedding], dtype="float32"), top_k)

    # Remove duplicates from results
    seen = set()
    unique_texts = []
    for i in indices[0]:
        if i < len(texts) and texts[i] not in seen:
            seen.add(texts[i])
            unique_texts.append(texts[i])

    return unique_texts


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

def delete_file(filename):
    """Delete a specific file and its embeddings from MongoDB."""
    collection.delete_many({"filename": filename})
    st.session_state.file_uploader_key += 1  # Reset file uploader
    st.rerun()

def delete_all_files():
    """Delete all files from MongoDB and FAISS."""
    collection.drop()
    st.session_state.stored_pdfs = []
    st.session_state.file_uploader_key += 1  # Reset file uploader
    st.rerun()

# **Store FAISS Embeddings in MongoDB**
def update_vector_db(texts, filename):
    embeddings = embedding_model.encode(texts).tolist()
    documents = [{"filename": filename, "text": text, "embedding": emb} for text, emb in zip(texts, embeddings)]
    collection.insert_many(documents)
    faiss_index.add(np.array(embeddings, dtype="float32"))
    # Optionally, save the faiss index to disk:
    # faiss.write_index(faiss_index, "faiss.index")

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
                        extracted_texts.append(extract_text(saved_path))

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
        client = OpenAI(api_key=OpenAI_API_KEY)
        response = client.chat.completions.create(
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
        return extract_text(filename)

async def main(urls):
    """Scrape multiple pages concurrently."""
    results = await asyncio.gather(*[extract_info(url) for url in urls])
    return results
# =================== Streamlit UI ============================
st.title("📄 AI Document Q&A and Web Scraper")

# Sidebar with Tabs
with st.sidebar:
    tab1, tab2 = st.tabs(["Configuration", "Web Scraper"])

    with tab1:
        st.header("Configuration")
        st.session_state.config = {}
        st.session_state.config["selected_models"] = st.multiselect(
            "Select AI Models (Up to 3)", 
            AVAILABLE_MODELS,
            default=AVAILABLE_MODELS[:3],
        )

        with st.expander("Model Pricing"):
            for model, price in MODEL_PRICING.items():
                st.write(f"**{model.split('/')[-1]}**: {price}")

        st.session_state.config["vary_temperature"] = st.checkbox("Vary Temperature", value=True)
        st.session_state.config["vary_top_p"] = st.checkbox("Vary Top-P", value=False)
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
                    scrape_results = asyncio.run(main(urls))
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

# TODO: Database Connection
# File Uploader for PDFs
st.header("Upload PDFs")
pdf_files = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)
if pdf_files:
    for pdf_file in pdf_files:
        text_chunks = process_pdf(pdf_file)
        st.sidebar.success(f"Processed {pdf_file.name}, extracted {len(text_chunks)} text chunks.")

# Chat UI with Multiple Models
st.header("Chat with Documents")
if prompt := st.chat_input("Ask a question"):
    context_indices = retrieve_context(prompt)
    context = " ".join([st.session_state.config["text_chunks"][i] for i in context_indices]) if context_indices else "No relevant context found."

    tabs = st.tabs([model.split("/")[-1] for model in st.session_state.config["selected_models"]])

    temp_values = [0, st.session_state.config["temperature"] / 3, st.session_state.config["temperature"]]
    top_p_values = [0, st.session_state.config["top_p"] / 3, st.session_state.config["top_p"]]

    for tab, model in zip(tabs, st.session_state.config["selected_models"]):
        with tab:
            st.markdown(f"""
                <div style="
                    border: 2px solid #2196F3;
                    padding: 10px;
                    border-radius: 10px;
                    background-color: #e3f2fd;
                    margin-bottom: 10px;">
                    <strong>User:</strong> {prompt}
                </div>
            """, unsafe_allow_html=True)

            for temp in temp_values if st.session_state.config["vary_temperature"] else [st.session_state.config["temperature"]]:
                for top_p in top_p_values if st.session_state.config["vary_top_p"] else [st.session_state.config["top_p"]]:
                    with st.spinner(f"Generating response from {model} (Temp={temp}, Top-P={top_p})..."):
                        response = generate_response(prompt, context, model, temp, top_p)

                    # Enhanced UI with clear separation
                    st.markdown(f"""
                        <div style="
                            border: 2px solid #fc0303; 
                            padding: 15px; 
                            border-radius: 10px; 
                            background-color: #f9f9f9;
                            margin-top: 10px;">
                            <strong style="color:#4CAF50;">Model:</strong> {model}<br>
                            <strong style="color:#FF9800;">Temperature:</strong> {temp}<br>
                            <strong style="color:#2196F3;">Top-P:</strong> {top_p}<br>
                            <hr>
                            <strong>Response:</strong> {response}
                        </div>
                    """, unsafe_allow_html=True)

                    # Download Button
                    st.download_button(label="Download Response", data=response, file_name=f"response_{model}_temp{temp}_topP{top_p}.txt", mime="text/plain")

