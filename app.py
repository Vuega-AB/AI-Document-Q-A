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
import openai
from io import BytesIO
from dotenv import load_dotenv
import dropbox
import hashlib

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
DROPBOX_TOKEN = os.getenv("DROPBOX_TOKEN")

# Initialize models and configurations
INDEX_FILE = "faiss_index.index"
CONFIG_FILENAME = "config.json"
INDEX_FILE_DROPBOX = "/faiss_index.index"  # Path on Dropbox
TEXT_FILE_DROPBOX = "/text_store.json"  # Path on Dropbox

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

def initialize_dropbox():
    try:
        dbx = dropbox.Dropbox(DROPBOX_TOKEN)
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
    chunks = [text[i:i + 2000] for i in range(0, len(text), 2000)]
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

def generate_response(prompt, context):
    try:
        max_context_tokens = 6000
        truncated_context = context[:max_context_tokens]
        
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "developer", "content": st.session_state.config["system_prompt"]},
                {"role": "user", "content": f"Context: {truncated_context}\n\nQuestion: {prompt}"}
            ],
            temperature=st.session_state.config["temperature"],
            top_p=st.session_state.config["top_p"]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

def retrieve_context(query, top_k=3):
    global text_store, faiss_index
    query_embedding = embedding_model.encode([query])
    distances, indices = faiss_index.search(query_embedding, top_k)
    valid_indices = [i for i in indices[0] if i != -1 and i < len(text_store)]
    retrieved_texts = [text_store[idx]["text"] for idx in valid_indices]
    return "\n\n".join(retrieved_texts) if retrieved_texts else "No relevant context found."

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

# Streamlit UI
st.title("📄 AI Document Q&A with OpenAI GPT-4o")

# Sidebar Configuration
with st.sidebar:
    st.header("Configuration")
    
    st.session_state.config["temperature"] = st.slider("Temperature", 0.0, 1.0, 0.7)
    st.session_state.config["top_p"] = st.slider("Top-p Sampling", 0.0, 1.0, 0.9)
    
    st.session_state.config["system_prompt"] = st.text_area("System Prompt", value=st.session_state.config.get("system_prompt", ""))

    config_file = st.file_uploader("Upload Configuration", type=['json'])
    if config_file:
        load_config(config_file)

    if st.button("Update and Download Configuration"):
        config_bytes = save_config(st.session_state.config)
        st.download_button(
            "Download Config",
            data=config_bytes,
            file_name=CONFIG_FILENAME,
            mime="application/json"
        )

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

# File Upload Section
st.header("Document Management")

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

# Chat Interface
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