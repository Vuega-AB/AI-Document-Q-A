import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
import PyPDF2
import io
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from io import BytesIO
import json
from dotenv import load_dotenv
from together import Together
from streamlit_option_menu import option_menu

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
        "selected_models": [],
    }

# Function to initialize or reload FAISS index
def initialize_vector_db():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.IndexFlatL2(384)
    return model, index

# Initialize FAISS and embedding model at startup
embedding_model, faiss_index = initialize_vector_db()

# Function to update FAISS index
def update_vector_db(texts):
    if not texts:
        return
    embeddings = embedding_model.encode(texts)
    faiss_index.add(np.array(embeddings).astype("float32"))
    st.session_state.config["text_chunks"].extend(texts)

# PDF Processing Functions
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    return "".join([page.extract_text() or "" for page in reader.pages])

def process_pdf(file):
    text = extract_text_from_pdf(file)
    chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]  
    update_vector_db(chunks)  
    return chunks

# Together.AI Integration
def generate_response(prompt, context, model):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": st.session_state.config["system_prompt"]},
                {"role": "user", "content": f"Context: {context}. Question: {prompt}"}
            ],
            temperature=st.session_state.config["temperature"],
            top_p=st.session_state.config["top_p"]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"

# RAG Pipeline
def retrieve_context(query, top_k=10):
    query_embedding = embedding_model.encode([query])
    distances, indices = faiss_index.search(query_embedding, top_k)
    valid_indices = [i for i in indices[0] if i < len(st.session_state.config["text_chunks"])]
    return valid_indices

# Streamlit UI
st.title("📄 AI Document Q&A with Multiple Models")

# Sidebar Configuration
with st.sidebar:
    st.header("Configuration")
    selected_models = st.multiselect("Select up to 2 AI Models", AVAILABLE_MODELS, max_selections=2)
    st.session_state.config["selected_models"] = selected_models if selected_models else [AVAILABLE_MODELS[0]]
    st.session_state.config["temperature"] = st.slider("Temperature", 0.0, 1.0, st.session_state.config["temperature"])
    st.session_state.config["top_p"] = st.slider("Top-p Sampling", 0.0, 1.0, st.session_state.config["top_p"])
    st.session_state.config["system_prompt"] = st.text_area("System Prompt", value=st.session_state.config["system_prompt"])

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
if prompt := st.chat_input("Ask a question"):
    context_indices = retrieve_context(prompt)
    context = " ".join([st.session_state.config["text_chunks"][i] for i in context_indices]) if context_indices else "No relevant context found."
    
    # Display responses in tabs
    if len(st.session_state.config["selected_models"]) == 2:
        model1, model2 = st.session_state.config["selected_models"]
        with st.spinner("Generating responses..."):
            response1 = generate_response(prompt, context, model1)
            response2 = generate_response(prompt, context, model2)
        
        tab1, tab2 = st.tabs([f"Response from {model1}", f"Response from {model2}"])
        with tab1:
            st.write(response1)
        with tab2:
            st.write(response2)
    else:
        model = st.session_state.config["selected_models"][0]
        with st.spinner("Generating response..."):
            response = generate_response(prompt, context, model)
        st.write(response)
