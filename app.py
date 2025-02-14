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
from io import BytesIO  # For handling file download
from dotenv import load_dotenv
import dropbox
import hashlib


# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
DROPBOX_TOKEN=os.getenv("DROPBOX_TOKEN")

# Initialize models and configurations
INDEX_FILE = "faiss_index.index"
CONFIG_FILENAME = "config.json"
INDEX_FILE_DROPBOX = "/faiss_index.index"  # Path on Dropbox
TEXT_FILE_DROPBOX = "/text_store.json"  # Path on Dropbox


# Initialize session state
# if "file_uploader_key" not in st.session_state:
#     st.session_state.file_uploader_key = 0
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
    text_store = {}

    try:
        with open(INDEX_FILE, "wb") as f_index:  # Use f_index for clarity
            metadata_index, res_index = dbx.files_download(path=INDEX_FILE_DROPBOX)
            f_index.write(res_index.content)
        index = faiss.read_index(INDEX_FILE)

        # Load text store from Dropbox directly into memory
        metadata_text, res_text = dbx.files_download(path=TEXT_FILE_DROPBOX)
        text_store = json.loads(res_text.content.decode('utf-8')) # decode bytes to string before parsing json
        
        st.info("Data loaded from Dropbox.")
        
    except dropbox.exceptions.ApiError:
        st.info("No existing data found in Dropbox. Initializing new index.")

    return model, index, text_store


embedding_model, faiss_index, text_store = initialize_and_load_data()


def save_data_to_dropbox():
    try:
        with open(INDEX_FILE, "rb") as f:
            dbx.files_upload(f.read(), INDEX_FILE_DROPBOX, mode=dropbox.files.WriteMode.overwrite)

        # Save text store directly to Dropbox
        text_json = json.dumps(text_store, ensure_ascii=False, indent=4).encode('utf-8')
        dbx.files_upload(text_json, TEXT_FILE_DROPBOX, mode=dropbox.files.WriteMode.overwrite)

        st.success("Data saved to Dropbox.")
    except Exception as e:
        st.error(f"Error saving data to Dropbox: {e}")


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
    chunks = [text[i:i + 2000] for i in range(0, len(text), 2000)]
    update_vector_db(chunks, file_name, file_hash)  # Pass file_hash as well
    save_data_to_dropbox()
    return chunks


# Function to update the FAISS index and store text mappings
def update_vector_db(texts, file_name, file_hash):
    global text_store, faiss_index
    embeddings = embedding_model.encode(texts)
    embeddings = np.array(embeddings).astype("float32")

    start_idx = len(text_store)
    faiss_index.add(embeddings)

    for i, text in enumerate(texts):
        text_store[str(start_idx + i)] = {
            "text": text,
            "file_name": file_name,
            "file_hash": file_hash  # Store the pre-calculated hash
        }

    # Save FAISS index
    faiss.write_index(faiss_index, INDEX_FILE)




# Function to save config as a downloadable JSON file
def save_config(config):
    """Save configuration as a JSON file."""
    json_bytes = json.dumps(config, indent=4).encode('utf-8')
    return BytesIO(json_bytes)

# Function to load config from an uploaded JSON file
def load_config(uploaded_file):
    """Load configuration from a JSON file uploaded by the user."""
    try:
        config_data = json.load(uploaded_file)
        st.session_state.config.update(config_data)  # Update session state directly
        st.sidebar.success("Configuration loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Failed to load configuration: {e}")


def generate_response(prompt, context):
    try:
        max_context_tokens = 6000  # Leave room for the prompt and system message
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

# RAG Pipeline
def retrieve_context(query, top_k=3):
    global text_store, faiss_index
    query_embedding = embedding_model.encode([query])
    distances, indices = faiss_index.search(query_embedding, top_k)
    valid_indices = [str(i) for i in indices[0] if i != -1 and str(i) in text_store]
    retrieved_texts = [text_store[idx]["text"] for idx in valid_indices]  # Access text from dict
    return "\n\n".join(retrieved_texts) if retrieved_texts else "No relevant context found."

def delete_pdf(file_hash):
    global text_store, faiss_index

    try:
        indices_to_remove = []
        for key, value in text_store.items():
            if value["file_hash"] == file_hash:
                indices_to_remove.append(int(key))

        for index in sorted(indices_to_remove, reverse=True):
            del text_store[str(index)]

        texts = [item["text"] for item in text_store.values()]
        if texts:
            embeddings = embedding_model.encode(texts)
            embeddings = np.array(embeddings).astype("float32")
            faiss_index = faiss.IndexFlatL2(384)
            faiss_index.add(embeddings)
        else:
            faiss_index = faiss.IndexFlatL2(384)

        faiss.write_index(faiss_index, INDEX_FILE)

        save_data_to_dropbox()
        st.success(f"PDF file deleted successfully!")

    except Exception as e:
        st.error(f"Error deleting PDF: {e}")

# URL Processing
# def get_pdfs_from_url(url):
#     response = requests.get(url)
#     soup = BeautifulSoup(response.content, "html.parser")
#     pdf_links = [link.get("href") for link in soup.find_all("a") if link.get("href") and link.get("href").endswith(".pdf")]
#     return pdf_links

# Streamlit UI
st.title("📄 AI Document Q&A with OpenAI GPT-4o")

# Sidebar Configuration
with st.sidebar:
    st.header("Configuration")
    
    # Model Settings
    st.session_state.config["temperature"] = st.slider("Temperature", 0.0, 1.0, 0.7)
    st.session_state.config["top_p"] = st.slider("Top-p Sampling", 0.0, 1.0, 0.9)
    
    # System Prompt
    st.session_state.config["system_prompt"] = st.text_area("System Prompt", value=st.session_state.config.get("system_prompt", ""))


    # Save and Load Configuration
    config_file = st.file_uploader("Upload Configuration", type=['json'])
    if config_file:
        load_config(config_file)  # Load and update config directly

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
        # Get unique file hashes
        unique_file_hashes = set(item["file_hash"] for item in text_store.values())  # Use a set for uniqueness

        for file_hash in unique_file_hashes:  # Loop over unique hashes
            file_name_to_display = "Unknown"
            for item in text_store.values():  # Find a filename associated with the hash
                if item["file_hash"] == file_hash:
                    file_name_to_display = item["file_name"]
                    break  # Stop searching once you find a filename

            # if st.button(f"Delete '{file_name_to_display}'", key=file_hash):  # Key is still essential
            #     delete_pdf(file_hash)
            col1, col2 = st.columns([3, 1])  # Create two columns

            with col1:  # Filename in the first column (non-clickable)
                st.write(file_name_to_display)  # Use st.write for non-clickable text

            with col2:  # Delete button in the second column
                if st.button("🗑️", key=f"delete_{file_hash}"):  # Trash can icon, unique key
                    delete_pdf(file_hash)
                    st.success("File deleted and file uploader cleared.")  
    else:
        st.write("No documents uploaded yet.")

# File Upload Section
st.header("Document Management")
uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True) #,key=st.session_state.file_uploader_key)
# url_input = st.text_input("Or enter URL to scan for PDFs")

# # Process URL PDFs
# if url_input:
#     pdf_links = get_pdfs_from_url(url_input)
#     for link in pdf_links:
#         if st.button(f"Process {link.split('/')[-1]}"):
#             response = requests.get(link)
#             with io.BytesIO(response.content) as pdf_file:
#                 process_pdf(pdf_file)
#                 st.session_state.config["stored_pdfs"].append(link)
#                 st.success("PDF processed successfully!")

# Process Uploaded Files
if uploaded_files:
    for file in uploaded_files:
        file_name = file.name
        unique_file_hashes = set(item["file_hash"] for item in text_store.values())
        # Check if the file has already been processed
        file_hash = hashlib.md5(file.getvalue()).hexdigest()
        if file_hash not in unique_file_hashes: ##use the stored ones
            with io.BytesIO(file.getvalue()) as pdf_file:
                process_pdf(pdf_file, file_name, file_hash)
            st.success(f"Processed '{file_name}'")
        else:
            st.info(f"File '{file_name}' has already been processed.")
    # # **Reset File Uploader**
    # st.session_state.file_uploader_key += 1
    # # st.rerun()

# st.header("Uploaded Documents")
# if text_store:
#     # Get unique file hashes
#     unique_file_hashes = set(item["file_hash"] for item in text_store.values())  # Use a set for uniqueness

#     for file_hash in unique_file_hashes:  # Loop over unique hashes
#         file_name_to_display = "Unknown"
#         for item in text_store.values():  # Find a filename associated with the hash
#             if item["file_hash"] == file_hash:
#                 file_name_to_display = item["file_name"]
#                 break  # Stop searching once you find a filename

#         if st.button(f"Delete '{file_name_to_display}'", key=file_hash):  # Key is still essential
#             delete_pdf(file_hash)
#             # st.session_state.file_uploader_key += 1
# else:
#     st.write("No documents uploaded yet.")

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
