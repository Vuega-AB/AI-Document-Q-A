import streamlit as st
import os
import PyPDF2
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from langdetect import detect
from dotenv import load_dotenv
from together import Together

# Load environment variables
load_dotenv()
API_KEY = os.getenv("TOGETHER_API_KEY")

client = Together(api_key=API_KEY)

AVAILABLE_MODELS = [
    "deepseek-ai/DeepSeek-V3",
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    "Qwen/Qwen1.5-7B-Chat"
]

FAISS_INDEX_PATH = "faiss_index.index"
TEXT_CHUNKS_PATH = "text_chunks.json"
PDF_STORAGE_PATH = "uploaded_pdfs"

# Ensure the folder exists
os.makedirs(PDF_STORAGE_PATH, exist_ok=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "config" not in st.session_state:
    st.session_state.config = {
        "temperature": 0.7,
        "top_p": 0.9,
        "system_prompt": "You are a helpful assistant.",
        "stored_pdfs": os.listdir(PDF_STORAGE_PATH),  # List saved PDFs
        "text_chunks": [],
        "selected_model": AVAILABLE_MODELS[0]
    }

# Load text chunks
def load_text_chunks():
    if os.path.exists(TEXT_CHUNKS_PATH):
        with open(TEXT_CHUNKS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_text_chunks():
    with open(TEXT_CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(st.session_state.config["text_chunks"], f, indent=4)

st.session_state.config["text_chunks"] = load_text_chunks()

# Initialize FAISS Vector Store
def initialize_vector_db():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    if os.path.exists(FAISS_INDEX_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
    else:
        index = faiss.IndexFlatL2(384)
    return model, index

embedding_model, faiss_index = initialize_vector_db()

def save_faiss_index():
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)

def update_vector_db(texts):
    embeddings = embedding_model.encode(texts)
    faiss_index.add(np.array(embeddings).astype("float32"))
    save_faiss_index()

def extract_text_from_pdf(file_path):
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "".join([page.extract_text() + "\n" for page in reader.pages if page.extract_text()])
    return text

def process_pdf(file, filename):
    file_path = os.path.join(PDF_STORAGE_PATH, filename)

    # Avoid processing the same file twice
    if filename in st.session_state.config["stored_pdfs"]:
        return

    text = extract_text_from_pdf(file_path)
    chunks = [{"text": text[i:i+2000], "file": filename} for i in range(0, len(text), 2000)]

    st.session_state.config["text_chunks"].extend(chunks)
    st.session_state.config["stored_pdfs"].append(filename)
    save_text_chunks()
    update_vector_db([chunk["text"] for chunk in chunks])

def retrieve_context(query, top_k=5):
    if not st.session_state.config["text_chunks"]:
        return []

    query_embedding = embedding_model.encode([query])
    distances, indices = faiss_index.search(query_embedding, top_k)

    valid_indices = [i for i in indices[0] if i < len(st.session_state.config["text_chunks"])]
    return [st.session_state.config["text_chunks"][i]["text"] for i in valid_indices]

def generate_response(prompt, context):
    try:
        response = client.chat.completions.create(
            model=st.session_state.config["selected_model"],
            messages=[
                {"role": "system", "content": st.session_state.config["system_prompt"]},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {prompt}"}
            ],
            temperature=st.session_state.config["temperature"],
            top_p=st.session_state.config["top_p"]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"

st.title("📄 AI Document Q&A with Together.AI")

# Sidebar for settings & file management
with st.sidebar:
    st.title("🔧 Settings & Knowledge Base")
    
    st.subheader("AI Model Settings")
    st.session_state.config["selected_model"] = st.selectbox("Select AI Model", AVAILABLE_MODELS, index=0)
    st.session_state.config["temperature"] = st.slider("Temperature", 0.0, 1.0, 0.7)
    st.session_state.config["top_p"] = st.slider("Top-p Sampling", 0.0, 1.0, 0.9)
    st.session_state.config["system_prompt"] = st.text_area("System Prompt", value=st.session_state.config["system_prompt"])
                

st.header("Document Management")
uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    new_files = [file for file in uploaded_files if file.name not in st.session_state.config["stored_pdfs"]]
    
    if new_files:
        for file in new_files:
            process_pdf(file, file.name)
        st.success(f"Processed {len(new_files)} new files")

        # Reload the sidebar and other parts of the interface
        st.rerun()  # Re-run the app to reload the sidebar


st.header("Chat with Documents")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question"):
    try:
        lang = detect(prompt)
    except:
        lang = "en"

    context_texts = retrieve_context(prompt)
    context = " ".join(context_texts) if context_texts else "No relevant context found."

    with st.spinner("Generating response..."):
        response = generate_response(prompt, context)

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        st.markdown(response)
