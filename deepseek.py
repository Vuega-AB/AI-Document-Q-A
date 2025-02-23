import streamlit as st
import os
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
import pdfplumber
from together import Together
import google.generativeai as genai
from google.api_core import exceptions
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig

# -----------------------------------------------------------------------------
# Environment and Client Initialization
# -----------------------------------------------------------------------------
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGO_URI = os.getenv("MONGO_URI") or os.getenv("MongoDB")

# Initialize Together.AI client
together_client = Together(api_key=TOGETHER_API_KEY)

# Configure Gemini (Google Generative AI)
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# MongoDB Connection
mongo_client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
db = mongo_client["userembeddings"]
collection = db["embeddings"]

# Initialize FAISS and Embedding Model
def initialize_vector_db():
    model_local = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.IndexFlatL2(384)
    return model_local, index

embedding_model, faiss_index = initialize_vector_db()

# -----------------------------------------------------------------------------
# Model Definitions and Session State
# -----------------------------------------------------------------------------
# Define models with their pricing and type (together vs gemini)
AVAILABLE_MODELS_DICT = {
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": {"price": "$0.88", "type": "together"},
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": {"price": "$3.50", "type": "together"},
    "databricks/dbrx-instruct": {"price": "$1.20", "type": "together"},
    "microsoft/WizardLM-2-8x22B": {"price": "$1.20", "type": "together"},
    "mistralai/Mixtral-8x22B-Instruct-v0.1": {"price": "$1.20", "type": "together"},
    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO": {"price": "$0.60", "type": "together"},
    "gemini-2.0-flash": {"price": "Custom", "type": "gemini"}
}
AVAILABLE_MODELS = list(AVAILABLE_MODELS_DICT.keys())

if "messages" not in st.session_state:
    st.session_state.messages = []
if "config" not in st.session_state:
    st.session_state.config = {
        "temperature": 0.7,
        "top_p": 0.9,
        "system_prompt": "You are a helpful assistant. Answer questions based on the provided context.",
        "selected_models": AVAILABLE_MODELS[:3],
        "vary_temperature": True,
        "vary_top_p": True
    }

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

def extract_text(uploaded_file):
    text = ""
    try:
        file_bytes = io.BytesIO(uploaded_file.getvalue())
        with pdfplumber.open(file_bytes) as pdf:
            text = ''.join([page.extract_text() or " " for page in pdf.pages])
    except Exception as e:
        st.error(f"An error occurred with pdfplumber: {e}")
    if not text:
        try:
            file_bytes.seek(0)
            reader = PyPDF2.PdfReader(file_bytes)
            text = ''.join([page.extract_text() or " " for page in reader.pages])
        except Exception as e:
            st.error(f"An error occurred with PyPDF2: {e}")
    return text

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

# -----------------------------------------------------------------------------
# AI Generation Functions
# -----------------------------------------------------------------------------
def generate_response_together(prompt, context, model, temp, top_p):
    system_prompt = st.session_state.config["system_prompt"]
    try:
        response = together_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context: {context}\nQuestion: {prompt}"}
            ],
            temperature=temp,
            top_p=top_p
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"

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

def process_pdf_links_from_url_sync(url: str):
    loop = asyncio.ProactorEventLoop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(fetch_and_process_pdf_links(url))

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
# Streamlit UI
# -----------------------------------------------------------------------------
st.title("📄 AI Document Q&A with Gemini & Together.AI")

# Sidebar: Configuration
with st.sidebar:
    st.header("Configuration")
    st.session_state.config["selected_models"] = st.multiselect(
        "Select AI Models (Up to 3)",
        AVAILABLE_MODELS,
        default=AVAILABLE_MODELS[:3]
    )
    with st.expander("Model Pricing"):
        for model, details in AVAILABLE_MODELS_DICT.items():
            st.write(f"**{model.split('/')[-1]}**: {details['price']}")
    st.session_state.config["temperature"] = st.slider("Temperature", 0.0, 1.0, st.session_state.config["temperature"], 0.05)
    st.session_state.config["top_p"] = st.slider("Top-P", 0.0, 1.0, st.session_state.config["top_p"], 0.05)
    st.session_state.config["system_prompt"] = st.text_area("System Prompt", value=st.session_state.config["system_prompt"])
    st.session_state.config["vary_temperature"] = st.checkbox("Vary Temperature", value=st.session_state.config.get("vary_temperature", True))
    st.session_state.config["vary_top_p"] = st.checkbox("Vary Top-P", value=st.session_state.config.get("vary_top_p", True))
    config_file = st.file_uploader("Upload Configuration", type=['json'])
    if config_file:
        load_config(config_file)
    if st.button("Update and Download Configuration"):
        config_bytes = save_config(st.session_state.config)
        st.download_button("Download Config", data=config_bytes, file_name="config.json", mime="application/json")

# File Uploader for PDFs
st.header("📤 Upload PDFs")
pdf_files = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)
if pdf_files:
    for pdf_file in pdf_files:
        chunks = process_pdf(pdf_file, pdf_file.name)
        st.sidebar.success(f"Processed {pdf_file.name}, extracted {len(chunks)} text chunks.")

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
# for message in st.session_state.get("messages", []):
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

if prompt := st.chat_input("Ask a question"):
    try:
        lang = detect(prompt)
    except Exception:
        lang = "en"
    retrieved_context = retrieve_context(prompt)
    context = " ".join(retrieved_context) if retrieved_context else "No relevant context found."
    
    # Define temperature and top-p values based on configuration flags
    if st.session_state.config.get("vary_temperature", True):
        temp_values = [st.session_state.config["temperature"] / 3, st.session_state.config["temperature"]]
    else:
        temp_values = [st.session_state.config["temperature"]]
    
    if st.session_state.config.get("vary_top_p", True):
        top_p_values = [st.session_state.config["top_p"] / 3, st.session_state.config["top_p"]]
    else:
        top_p_values = [st.session_state.config["top_p"]]
    
    # Create a tab for each selected model
    tabs = st.tabs([model.split("/")[-1] for model in st.session_state.config["selected_models"]])
    responses = {}
    
    for tab, model in zip(tabs, st.session_state.config["selected_models"]):
        with tab:
            model_type = AVAILABLE_MODELS_DICT[model]["type"]
            model_responses = []
            for temp in temp_values:
                for top_p in top_p_values:
                    with st.spinner(f"Generating response from {model} (Temp={temp}, Top-P={top_p})..."):
                        if model_type == "together":
                            resp = generate_response_together(prompt, context, model, temp, top_p)
                        elif model_type == "gemini":
                            resp = generate_response_gemini(prompt, context, temp, top_p)
                        model_responses.append(f"Temp={temp}, Top-P={top_p}: {resp}")
            response_text = "\n\n".join(model_responses)
            st.markdown(f"""
                <div style="
                    border: 2px solid #fc0303;
                    padding: 15px;
                    border-radius: 10px;
                    background-color: #f9f9f9;
                    margin-top: 10px;">
                    <strong style="color:#4CAF50;">Model:</strong> {model}<br>
                    <strong style="color:#FF9800;">Responses:</strong><br>{response_text}
                </div>
            """, unsafe_allow_html=True)
            responses[model] = model_responses

    # st.session_state.messages.append({"role": "user", "content": prompt})
    # st.session_state.messages.append({"role": "assistant", "content": json.dumps(responses, indent=2)})
    # with st.chat_message("user"):
    #     st.markdown(prompt)
    # with st.chat_message("assistant"):
    #     st.markdown(json.dumps(responses, indent=2))
