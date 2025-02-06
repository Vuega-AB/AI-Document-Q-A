import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
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

# Load environment variables
load_dotenv()
API_KEY = os.getenv("TOGETHER_API_KEY")

client = Together(api_key=API_KEY)

# Available Together.AI models
AVAILABLE_MODELS = [
    "deepseek-ai/DeepSeek-V3",
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    "Qwen/Qwen1.5-7B-Chat"
]

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
        "selected_model": AVAILABLE_MODELS[0]  # Default model
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

# Function to initialize or reload FAISS index
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

# Function to update FAISS index and persist it
def update_vector_db(texts):
    if not texts:
        return
    embeddings = embedding_model.encode(texts)
    faiss_index.add(np.array(embeddings).astype("float32"))
    faiss.write_index(faiss_index, "faiss_index.index")  # Save index persistently
    
    st.session_state.config["text_chunks"].extend(texts)
    with open("text_chunks.json", "w") as f:
        json.dump(st.session_state.config["text_chunks"], f)

# PDF Processing Functions
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = "".join([page.extract_text() + "\n" for page in reader.pages if page.extract_text()])
    return text

def process_pdf(file):
    text = extract_text_from_pdf(file)
    chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]  
    update_vector_db(chunks)  
    return chunks

# Together.AI Integration
def generate_response(prompt, context):
    try:
        response = client.chat.completions.create(
            model=st.session_state.config["selected_model"],
            messages=[
                {"role": "system", "content": f"{st.session_state.config['system_prompt']}"},
                {"role": "user", "content": f"Context: {context}. Question: {prompt}"}
            ],
            temperature=st.session_state.config["temperature"],
            top_p=st.session_state.config["top_p"]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"

# RAG Pipeline
def retrieve_context(query, top_k=15):
    query_embedding = embedding_model.encode([query])
    distances, indices = faiss_index.search(query_embedding, top_k)
    valid_indices = [i for i in indices[0] if i < len(st.session_state.config["text_chunks"])]
    return valid_indices

# URL Processing
def get_pdfs_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    return [link.get("href") for link in soup.find_all("a") if link.get("href", "").endswith(".pdf")]

# Streamlit UI
st.title("📄 AI Document Q&A with Together.AI")

# Sidebar Configuration
with st.sidebar:
    st.header("Configuration")
    st.session_state.config["selected_model"] = st.selectbox("Select AI Model", AVAILABLE_MODELS, index=0)
    st.session_state.config["temperature"] = st.slider("Temperature", 0.0, 1.0, st.session_state.config.get("temperature", 0.7))
    st.session_state.config["top_p"] = st.slider("Top-p Sampling", 0.0, 1.0, st.session_state.config.get("top_p", 0.9))
    st.session_state.config["system_prompt"] = st.text_area("System Prompt", value=st.session_state.config.get("system_prompt", ""))
    config_file = st.file_uploader("Upload Configuration", type=['json'])
    if config_file:
        load_config(config_file)
    if st.button("Update and Download Configuration"):
        config_bytes = save_config(st.session_state.config)
        st.download_button("Download Config", data=config_bytes, file_name="config.json", mime="application/json")

# File Upload Section
st.header("Document Management")
uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
if uploaded_files:
    for file in uploaded_files:
        with io.BytesIO(file.getvalue()) as pdf_file:
            process_pdf(pdf_file)
            st.session_state.config["stored_pdfs"].append(file.name)
    st.success(f"Processed {len(uploaded_files)} files")

st.header("Chat with Documents")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question"):
    context_indices = retrieve_context(prompt)
    context = " ".join([st.session_state.config["text_chunks"][i] for i in context_indices]) if context_indices else "No relevant context found."
    with st.spinner("Generating response..."):
        response = generate_response(prompt, context)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
