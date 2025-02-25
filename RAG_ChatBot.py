import streamlit as st
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
import asyncio
import httpx
import aiofiles
from urllib.parse import urljoin
import pdfplumber
import google.generativeai as genai
from google.api_core import exceptions
import openai

# ================== Environment Variables ==================
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGO_URI = os.getenv("MongoDB")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY")


# =================== Connections ============================
# Configure Gemini (Google Generative AI)
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# MongoDB Connection
mongo_client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
db = mongo_client["userembeddings"]
collection = db["embeddings"]

client = Together(api_key=TOGETHER_API_KEY)

try:
    import playwright
    subprocess.run(["playwright", "install"], check=True)
except Exception as e:
    print(f"Error installing Playwright: {e}")


# Initialize FAISS and Embedding Model
def initialize_vector_db():
    model_local = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.IndexFlatL2(384)
    return model_local, index

embedding_model, faiss_index = initialize_vector_db()

# Available Together.AI models
AVAILABLE_MODELS_DICT = {
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": {"price": "$0.88", "type": "together"},
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": {"price": "$3.50", "type": "together"},
    "databricks/dbrx-instruct": {"price": "$1.20", "type": "together"},
    "microsoft/WizardLM-2-8x22B": {"price": "$1.20", "type": "together"},
    "mistralai/Mixtral-8x22B-Instruct-v0.1": {"price": "$1.20", "type": "together"},
    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO": {"price": "$0.60", "type": "together"},
    "gemini-2.0-flash": {"price": "Custom", "type": "gemini"},
    "openai-4o": {"price": "Custom", "type": "openai"}
}

AVAILABLE_MODELS = list(AVAILABLE_MODELS_DICT.keys())

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "config" not in st.session_state:
    st.session_state.config = {
        "temperature": 0.7,
        "top_p": 0.9,
        "system_prompt": "You are a helpful assistant. Answer questions strictly based on the provided context. If there is no context, say 'I don't have enough information to answer that.",
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


def extract_text(file):
    reader = PyPDF2.PdfReader(file)
    text = "".join([page.extract_text() + "\n" for page in reader.pages if page.extract_text()])
    return text
# ================== Generate Response ==================

def update_vector_db(texts, filename="uploaded"):
    if not texts:
        return
    embeddings = embedding_model.encode(texts).tolist()
    documents = [{"filename": filename, "text": text, "embedding": emb} for text, emb in zip(texts, embeddings)]
    collection.insert_many(documents)
    faiss_index.add(np.array(embeddings, dtype="float32"))

def process_pdf(file, filename="uploaded"):
    text = extract_text(file)
    chunks = chunk_text(text)
    update_vector_db(chunks, filename)
    return chunks

# -----------------------------------------------------------------------------
# AI Generation Functions
# -----------------------------------------------------------------------------
def generate_response_gemini(prompt, context, temp, top_p):
    system_prompt = st.session_state.config["system_prompt"]
    input_parts = [system_prompt + "\n" + context, prompt]
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
            st.warning(f"API quota exceeded. Retrying... ({attempt+1}/{retries})")
            time.sleep(5)
        except Exception as e:
            return f"Error generating response: {str(e)}"
    st.error("API quota exceeded. Please try again later.")
    return "Error generating response."

#openAi
def generate_response_openAi(prompt, context, temp, top_p):
    try:
        # max_context_tokens = 6000
        # truncated_context = context[:max_context_tokens]
        
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "developer", "content": st.session_state.config["system_prompt"]},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {prompt}"}
            ],
            temperature=temp,
            top_p=top_p
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Together.AI Integration
def generate_response(prompt, context, model, temp, top_p):
    system_prompt = st.session_state.config["system_prompt"]
    if model == "grok-3":
        if not GROK_API_KEY:
            raise ValueError("Missing xAI API Key. Set the GROK_API_KEY environment variable.")

        # Define the API endpoint and headers
        url = "https://api.x.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROK_API_KEY}",
            "Content-Type": "application/json"
        }

        # Construct the payload based on the xAI Grok API
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context: {context}. Question: {prompt}"}
            ],
            "temperature": temp,
            "top_p": top_p,
            "stream": False
        }
        try:
            # Send the POST request to the xAI API
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)
            return response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        except requests.exceptions.RequestException as e:
            print(f"Error communicating with xAI API: {e}")
            return ""
    else:
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


# -----------------------------------------------------------------------------
# Retrieval Function (RAG)
# -----------------------------------------------------------------------------
def retrieve_context(query, top_k=15):
    query_embedding = embedding_model.encode([query]).tolist()[0]
    stored_docs = list(collection.find({}, {"_id": 0, "embedding": 1, "text": 1}))
    if not stored_docs:
        return []
    embeddings = np.array([doc["embedding"] for doc in stored_docs], dtype="float32")
    texts = [doc["text"] for doc in stored_docs]
    if faiss_index.ntotal == 0:
        faiss_index.add(np.array(embeddings, dtype="float32"))
    top_k = min(top_k, len(texts))
    distances, indices = faiss_index.search(np.array([query_embedding], dtype="float32"), top_k)
    seen = set()
    unique_texts = []
    for i in indices[0]:
        if i < len(texts) and texts[i] not in seen:
            seen.add(texts[i])
            unique_texts.append(texts[i])
    return unique_texts


# ========= PDFs Link Extraction via URL =========
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
            # title = item.text.strip()
            if link and f"/{listing_endpoint}/" in link and link != url and not link.endswith("/rss"):
                if not link.startswith("http"):
                    link = base_url + link
                items.add(link)
            

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


# def summarize_text(text):
#     """Summarizes extracted text using OpenAI."""
#     try:
#         client = OpenAI(api_key=OPENAI_API_KEY)
#         response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": "Summarize the following text into a concise paragraph."},
#                 {"role": "user", "content": text}
#             ]
#         )
#         return response.choices[0].message.content
#     except Exception as e:
#         logging.error(f"Error in summarization: {e}")
#         return "Summarization failed."

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

    # # Extract paragraphs
    # paragraphs = "\n".join(p.get_text() for p in soup.find_all("p"))
    # print(f"Text from {url}:\n{paragraphs[:300]}...")

    # Extract PDF links and convert relative to absolute URLs
    pdf_links = [urljoin(base_url, a["href"]) for a in soup.find_all("a", href=True) if ".pdf" in a["href"].lower()]

    # # Download and extract PDF content
    # pdf_texts = []
    # for link in pdf_links:
    #     pdf_texts.append(await download_and_extract_pdf(link))

    return pdf_links

# async def download_and_extract_pdf(url):
#     """Download a PDF and extract text."""
#     async with httpx.AsyncClient() as client:
#         response = await client.get(url)
#         filename = url.split("/")[-1]

#         # Save PDF file
#         async with aiofiles.open(filename, "wb") as f:
#             await f.write(response.content)
#         print(f"Downloaded: {filename}")

#         # Extract text from PDF
#         return extract_text(filename)

async def main(urls):
    """Scrape multiple pages concurrently."""
    results = await asyncio.gather(*[extract_info(url) for url in urls])
    return results

# -----------------------------------------------------------------------------
# File Deletion Functions
# -----------------------------------------------------------------------------
def delete_file(filename):
    collection.delete_many({"filename": filename})
    st.rerun()

def delete_all_files():
    collection.drop()
    st.rerun()

# -----------------------------------------------------------------------------
# File Deletion Functions
# -----------------------------------------------------------------------------
async def store_in_DB(pdf_links):
    async with aiohttp.ClientSession() as session:
        for pdf_link in pdf_links:
            try:
                async with session.get(pdf_link) as response:
                    if response.status == 200:
                        pdf_bytes = await response.read()
                        pdf_file = BytesIO(pdf_bytes)
                        filename = os.path.basename(pdf_link)
                        process_pdf(pdf_file, filename)
                        st.success(f"Processed PDF: {filename}")
                    else:
                        st.error(f"Failed to download PDF: {pdf_link}")
            except Exception as e:
                st.error(f"Error processing {pdf_link}: {e}")
    st.success("Finished processing all PDF links.")

# =================== Streamlit UI ============================
st.title("📄 AI Document Q&A and Web Scraper")

# Sidebar with Tabs
with st.sidebar:
    tab1, tab2, tab3 = st.tabs(["Configuration", "Web Scraper", "Database"])

    with tab1:
        st.header("Configuration")
        st.session_state.config = {}
        st.session_state.config["selected_models"] = st.multiselect(
            "Select AI Models (Up to 3)", 
            AVAILABLE_MODELS,
            default=AVAILABLE_MODELS[:3],
        )
    
        with st.expander("Model Pricing"):
            for model, details in AVAILABLE_MODELS_DICT.items():
                st.write(f"**{model.split('/')[-1]}**: {details['price']}")

        # Grok-3 Integration
        use_grok = st.checkbox("Use Grok-3 Model", value=True)
        if use_grok:
            st.session_state.config["selected_models"].append("grok-3")
                
        st.session_state.config["vary_temperature"] = st.checkbox("Vary Temperature", value=True)
        st.session_state.config["vary_top_p"] = st.checkbox("Vary Top-P", value=False)
        st.session_state.config["temperature"] = st.slider("Temperature", 0.0, 1.0, 0.5, 0.05)
        st.session_state.config["top_p"] = st.slider("Top-P", 0.0, 1.0, 0.9, 0.05)
        st.session_state.config["system_prompt"] = st.text_area("System Prompt", value="You are a helpful assistant. Answer questions strictly based on the provided context. If there is no context, say 'I don't have enough information to answer that.")

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
        num_pages = st.number_input("Enter Number of Pages", 1, 10, 3)

        if st.button("Start Scraping"):

            if sys.platform == "win32":
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

            with st.spinner("Scraping in progress..."):
                urls = get_all_items(base_url, listing_endpoint, pagination_format, num_pages)

            if urls:
                st.success(f"Found {len(urls)} items!")

                scrape_results = []
                with st.spinner("Scraping in progress..."):
                    scrape_results = asyncio.run(main(urls))

                pdf_links = set()  # Use a set to store unique links
                for i, extracted_data in enumerate(scrape_results):
                    pdf_links.update(extracted_data)

                # Convert set back to list
                pdf_links = list(pdf_links)

                if pdf_links:
                    st.write("**Extracted PDFs:**")
                    for pdf in pdf_links:
                        st.markdown(pdf)

            else:
                st.warning("No items found.")

            if pdf_links:
                asyncio.run(store_in_DB(pdf_links))

    with tab3:
        # Display Stored Files in MongoDB
        st.subheader("📂 Stored Files in Database")
        stored_files = list(collection.distinct("filename"))
        if stored_files:
            for filename in stored_files:
                col1, col2 = st.columns([0.8, 0.2])
                col1.write(f"📄 {filename}")
                if col2.button("🗑️ Delete", key=filename):
                    delete_file(filename)
            if st.button("🗑️ Delete All Files"):
                delete_all_files()
        else:
            st.info("No files stored in the database.")

# File Uploader for PDFs
st.header("📤 Upload PDFs")
pdf_files = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)
if pdf_files:
    for pdf_file in pdf_files:
        chunks = process_pdf(pdf_file, pdf_file.name)
        st.sidebar.success(f"Processed {pdf_file.name}, extracted {len(chunks)} text chunks.")

# Chat UI with Multiple Models
st.header("💬 Chat with Documents")
if prompt := st.chat_input("Ask a question"):
    try:
        lang = detect(prompt)
    except Exception:
        lang = "en"
    retrieved_context = retrieve_context(prompt)
    context = " ".join(retrieved_context) if retrieved_context else "No relevant context found."
    
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

            model_type = AVAILABLE_MODELS_DICT[model]["type"]
            for temp in temp_values if st.session_state.config["vary_temperature"] else [st.session_state.config["temperature"]]:
                for top_p in top_p_values if st.session_state.config["vary_top_p"] else [st.session_state.config["top_p"]]:
                    with st.spinner(f"Generating response from {model} (Temp={temp}, Top-P={top_p})..."):
                        response = generate_response(prompt, context, model, temp, top_p)
                        if model_type == "together":
                            response = generate_response(prompt, context, model, temp, top_p)
                        elif model_type == "gemini":
                            response = generate_response_gemini(prompt, context, temp, top_p)
                        elif model_type == "openai":
                            response = generate_response_openAi(prompt, context, temp, top_p)
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

