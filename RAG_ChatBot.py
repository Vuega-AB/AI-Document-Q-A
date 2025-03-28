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
import sys
import asyncio
import httpx
from urllib.parse import urljoin
import google.generativeai as genai
from google.api_core import exceptions
import openai
import hashlib
from hashlib import md5
import random
import smtplib
from email.message import EmailMessage
from email_validator import validate_email, EmailNotValidError
# ================== Environment Variables ==================
load_dotenv()
# TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# MONGO_URI = os.getenv("MongoDB")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TOGETHER_API_KEY = st.secrets["TOGETHER_API_KEY"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
MONGO_URI = st.secrets["MongoDB"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# =================== Connections ============================
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

mongo_client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
db = mongo_client["userembeddings"]
collection = db["embeddings"]

client = Together(api_key=TOGETHER_API_KEY)
# =================== Authentication UI =========================

# Configure email sender credentials
EMAIL_SENDER = "nancyhisham2003@gmail.com"
EMAIL_PASSWORD = "sike xztt teak orkr"

# Function to send OTP
def send_otp(email, otp):
    msg = EmailMessage()
    msg.set_content(f"Your OTP for login is: {otp}")
    msg["Subject"] = "Your Login OTP"
    msg["From"] = EMAIL_SENDER
    msg["To"] = email

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Error sending email: {e}")
        return False

# Initialize session state variables if they don't exist
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None

# Only show login fields if user is not logged in
if not st.session_state.logged_in:
    # Streamlit UI
    st.title("Secure User Authentication")
    user_email = st.text_input("Enter your email:")
    
    if st.button("Send OTP"):
        if user_email:
            otp = random.randint(100000, 999999)  # Generate a 6-digit OTP
            if send_otp(user_email, otp):
                st.session_state["otp"] = otp
                st.session_state["email"] = user_email
                st.success("OTP sent! Check your email.")
    
    if "otp" in st.session_state:
        otp_input = st.text_input("Enter OTP:", type="password")

        if st.button("Login"):
            if otp_input and int(otp_input) == st.session_state["otp"]:
                # Extract username from email
                st.session_state.username = st.session_state["email"].split("@")[0]
                
                # Generate user-specific collection
                user_collection_name = f"user_{st.session_state['email'].replace('@', '_').replace('.', '_')}"
                user_collection = db[user_collection_name]

                # Store user data if new
                if not user_collection.find_one({"email": st.session_state["email"]}):
                    user_collection.insert_one({"email": st.session_state["email"], "data": []})

                st.success(f"Welcome, {st.session_state.username}!")
                st.session_state.logged_in = True  # Set login status
                st.rerun()


# =================== Main App =========================

def initialize_vector_db():
    model_local = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
    index = faiss.IndexFlatL2(768)
    return model_local, index

embedding_model, faiss_index = initialize_vector_db()

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

if "messages" not in st.session_state:
    st.session_state.messages = []
if "config" not in st.session_state:
    st.session_state.config = {
        "temperature": 0.7,
        "top_p": 0.9,
        "system_prompt": "You are a helpful assistant. Answer questions strictly based on the provided context. If there is no context, say 'I don't have enough information to answer that.",
        "selected_models": AVAILABLE_MODELS[:1],
        "vary_temperature": 1,
        "vary_top_p": 0
    }

def save_config(config):
    """Save configuration as a JSON file."""
    json_bytes = json.dumps(config, indent=4).encode('utf-8')
    return BytesIO(json_bytes)

def load_config(uploaded_file):
    """Load configuration from a JSON file uploaded by the user."""
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

def update_vector_db(username, texts, filehash, filename="uploaded"):
    if not texts:
        return
    embeddings = embedding_model.encode(texts).tolist()
    # embedding_array = np.array(embeddings, dtype="float32")
    # print(f"Embedding shape: {embedding_array.shape}")
    documents = [{"username": username, "filename": filename, "text": text, "filehash": filehash, "embedding": emb} for text, emb in zip(texts, embeddings)] 
    try:
        collection.insert_many(documents, ordered=False)
    except Exception as e:
        print(f"Error inserting documents: {e}")

    faiss_index.add(np.array(embeddings, dtype="float32"))

def process_pdf(username, file, filehash=None, filename="uploaded"):
    text = extract_text(file)
    chunks = chunk_text(text)
    update_vector_db(username, chunks, filehash, filename)
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

def generate_response_openAi(prompt, context, temp, top_p):
    try:
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


# -----------------------------------------------------------------------------
# Retrieval Function (RAG)
# -----------------------------------------------------------------------------
def retrieve_context(username, query, top_k=15):
    if not username:
        return []
    
    query_embedding = embedding_model.encode([query]).tolist()[0]
    stored_docs = list(collection.find({"username": username}, {"_id": 0, "embedding": 1, "text": 1}))
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

# =================== Try another way ============================
async def fetch_page(url):
    """Fetch page content asynchronously."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=None)
        return response.text, str(response.url)


async def download_with_retries(url, retries=3, delay=5):
    for attempt in range(retries):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=60.0)
                response.raise_for_status()
                return response.content
        except (httpx.ReadTimeout, httpx.RequestError) as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(delay)
    raise Exception(f"Failed to download {url} after {retries} attempts")

async def extract_info(url):
    """Extract text and PDF links from a webpage."""
    html, base_url = await fetch_page(url)
    soup = BeautifulSoup(html, "html.parser")

    pdf_links = [urljoin(base_url, a["href"]) for a in soup.find_all("a", href=True) if ".pdf" in a["href"].lower()]

    return pdf_links

async def main(urls):
    """Scrape multiple pages concurrently."""
    results = await asyncio.gather(*[extract_info(url) for url in urls])
    return results

# -----------------------------------------------------------------------------
# File Deletion Functions
# -----------------------------------------------------------------------------
def delete_file(username, filename):
    collection.delete_many({"username": username, "filename": filename})
    st.rerun()

def delete_all_files(username):
    collection.delete_many({"username": username})
    st.rerun()

# -----------------------------------------------------------------------------
# File Deletion Functions
# -----------------------------------------------------------------------------
async def store_in_DB(username, pdf_links):
    async with aiohttp.ClientSession() as session:
        unique_file_hashes = set(item["filehash"] for item in collection.find({"username": username}, {"filehash": 1}))
        for pdf_link in pdf_links:
            try:
                async with session.get(pdf_link) as response:
                    if response.status == 200:
                        with st.spinner(f"Processing PDF: {pdf_link}"):
                            pdf_bytes = await response.read()
                            filehash = hashlib.md5(pdf_bytes).hexdigest()
                            if filehash not in unique_file_hashes:
                                pdf_file = BytesIO(pdf_bytes)
                                filename = os.path.basename(pdf_link)
                                process_pdf(pdf_file, filehash, filename )
                                st.success(f"Processed PDF: {filename}")
                                unique_file_hashes.add(filehash)
                    else:
                        st.error(f"Failed to download PDF: {pdf_link}")
            except Exception as e:
                st.error(f"Error processing {pdf_link}: {e}")
    st.success("Finished processing all PDF links.")

# =================== Streamlit UI ============================
is_dark_mode = st.get_option("theme.base") == "dark"

background_color = "#1E1E1E" if is_dark_mode else "#f9f9f9"
border_color = "#BB86FC" if is_dark_mode else "#fc0303"
text_color = "#E0E0E0" if is_dark_mode else "#000000"
user_background = "#333" if is_dark_mode else "#e3f2fd"
user_text_color = "#FFF" if is_dark_mode else "#000"

if st.session_state.username:  # Only show if user has logged in
    st.sidebar.write(f"👋 Welcome, {st.session_state.username}")

username = st.session_state.get("username")
if not username:
    st.stop()

if st.sidebar.button("Logout"):
    st.session_state.clear()  # Clears all stored session data
    st.session_state.authenticated = False
    st.session_state.pop("username", None)
    st.rerun()

st.title("📄 IntelLaw")

with st.sidebar:
    tab1, tab2, tab3 = st.tabs(["Configuration", "Web Scraper", "Database"])

    with tab1:
        st.header("Configuration")        
        selected_models = st.multiselect(
            "Select AI Models (Up to 3)", 
            AVAILABLE_MODELS,
            default=st.session_state.config["selected_models"],
        )
    
        with st.expander("Model Pricing"):
            for model, details in AVAILABLE_MODELS_DICT.items():
                st.write(f"**{model.split('/')[-1]}**: {details['price']}")

        st.session_state.config["vary_temperature"] = st.checkbox(
            "Vary Temperature", value=st.session_state.config.get("vary_temperature", True)
        )
        st.session_state.config["vary_top_p"] = st.checkbox(
            "Vary Top-P", value=st.session_state.config.get("vary_top_p", False)
        )
        st.session_state.config["temperature"] = st.slider(
            "Temperature", 0.0, 1.0, st.session_state.config.get("temperature", 0.7), 0.05
        )
        st.session_state.config["top_p"] = st.slider(
            "Top-P", 0.0, 1.0, st.session_state.config.get("top_p", 0.9), 0.05
        )
        st.session_state.config["system_prompt"] = st.text_area(
            "System Prompt", value=st.session_state.config.get(
                "system_prompt", "You are a helpful assistant. Answer questions based on the provided context."
            )
        )

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

                pdf_links = set()
                for i, extracted_data in enumerate(scrape_results):
                    pdf_links.update(extracted_data)

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
        st.subheader("📂 Stored Files in Database")
        st.header("📤 Upload PDFs")
        if "file_uploader_key" not in st.session_state:
            st.session_state.file_uploader_key = 0

        pdf_files = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True, key=f"file_uploader_{st.session_state.file_uploader_key}")
        if pdf_files:
            unique_file_hashes = set(item["filehash"] for item in collection.find({"username": username}, {"filehash": 1}))
            for pdf_file in pdf_files:
                file_hash = hashlib.md5(pdf_file.getvalue()).hexdigest()
                if file_hash in unique_file_hashes:
                    st.warning(f"⚠️ {pdf_file.name} already exists. Skipping...")
                    continue

                chunks = process_pdf(username, pdf_file, file_hash, pdf_file.name)
                unique_file_hashes.add(file_hash)
                st.success(f"Processed {pdf_file.name}, extracted {len(chunks)} text chunks.")
            
            st.session_state.file_uploader_key += 1
            st.rerun()
            
        stored_files = list(collection.distinct("filename", {"username": username}))
        if stored_files:
            for filename in stored_files:
                col1, col2 = st.columns([0.8, 0.2])
                col1.write(f"📄 {filename}")
                if col2.button("🗑️ Delete", key=filename):
                    delete_file(username, filename)
            if st.button("🗑️ Delete All Files"):
                delete_all_files(username)
        else:
            st.info("No files stored in the database.")

st.header("💬 Chat with Documents")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 🔹 **Step 1: Show Chat History Without Triggering Response Generation**
tabs = st.tabs([model.split("/")[-1] for model in selected_models])

for tab, model in zip(tabs, selected_models):
    with tab:
        model_type = AVAILABLE_MODELS_DICT[model]["type"]

        # Display stored chat history for this model
        for message in st.session_state.chat_history:
            # Show user messages for all models, but model responses only in their respective tab
            if message["model_name"] == model:
                role = "User" if message["role"] == "user" else "Model"
                message_color = user_background if message["role"] == "user" else background_color
                text_color = user_text_color if message["role"] == "user" else text_color

                if role == "Model":
                    st.markdown(f"""
                        <div style="
                            border: 2px solid {border_color}; 
                            padding: 15px; 
                            border-radius: 10px; 
                            background-color: {background_color};
                            color: {text_color};
                            margin-top: 10px;">
                            <strong style="color:#4CAF50;">Model:</strong> {model}<br>
                            <strong style="color:#FF9800;">Temperature:</strong> {message["temp"]}<br>
                            <strong style="color:#2196F3;">Top-P:</strong> {message["top_p"]}<br>
                            <hr>
                            <strong>Response:</strong> {message["text"]}
                        </div>
                    """, unsafe_allow_html=True)
                elif role == "User":
                    st.markdown(f"""
                        <div style="
                            border: 2px solid {border_color};
                            padding: 15px;
                            border-radius: 10px;
                            background-color: {message_color};
                            color: {text_color};
                            margin-bottom: 10px;">
                            <strong>{role}:</strong> {message["text"]}
                        </div>
                    """, unsafe_allow_html=True)


# 🔹 **Step 2: Generate Response Only When New Input is Entered**
if prompt := st.chat_input("Ask a question"):
    try:
        lang = detect(prompt)
    except Exception:
        lang = "en"
    
    retrieved_context = []
    with st.spinner("Processing your Query..."):
        retrieved_context = retrieve_context(username, prompt)
    context = " ".join(retrieved_context) if retrieved_context else "No relevant context found."

    # Store user input only once (not per model)
    

    # Iterate through selected models and generate responses
    temp_values = [0, st.session_state.config["temperature"] / 2, st.session_state.config["temperature"]]
    top_p_values = [0, st.session_state.config["top_p"] / 2, st.session_state.config["top_p"]]

    for model in selected_models:
        model_type = AVAILABLE_MODELS_DICT[model]["type"]
        st.session_state.chat_history.append({"role": "user", "text": prompt, "model_name": model})
        for temp in temp_values if st.session_state.config["vary_temperature"] else [st.session_state.config["temperature"]]:
            for top_p in top_p_values if st.session_state.config["vary_top_p"] else [st.session_state.config["top_p"]]:
                with st.spinner(f"Generating response from {model} (Temp={temp}, Top-P={top_p})..."):
                    if model_type == "together":
                        response = generate_response(prompt, context, model, temp, top_p)
                    elif model_type == "gemini":
                        response = generate_response_gemini(prompt, context, temp, top_p)
                    elif model_type == "openai":
                        response = generate_response_openAi(prompt, context, temp, top_p)

                # Store model response in chat history with its model name
                st.session_state.chat_history.append({"role": "model", "text": response, "model_name": model, "temp": temp, "top_p": top_p})

    st.rerun()  # Force UI update to display new responses
