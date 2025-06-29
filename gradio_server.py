# gradio_server.py

import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import gradio as gr

# Import your application's functions
from gradio_ui import create_gradio_app
from backend import initialize_all_components # <-- We import the main initialization function

# ==============================================================================
#  FastAPI App Setup
# ==============================================================================
# Create a FastAPI app object. This will be our main entry point for the server.
app_fastapi = FastAPI()

# ==============================================================================
#  CORS (Cross-Origin Resource Sharing) Configuration
# ==============================================================================
# This allows your Flask domain to embed the Gradio app in an iframe.
# allowed_origin = os.getenv("FLASK_BASE_URL", "http://localhost:5000")

allowed_origin = os.getenv("FLASK_BASE_URL")

print(f"--- CONFIGURING CORS: Allowing origin '{allowed_origin}' for Gradio server ---")

app_fastapi.add_middleware(
    CORSMiddleware,
    allow_origins=[allowed_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
#  SERVER STARTUP INITIALIZATION
# ==============================================================================
# This event handler runs the initialization function ONCE when the server starts.
# This will cause a delay during deployment while models are loaded.
@app_fastapi.on_event("startup")
async def startup_event():
    print("--- FastAPI Server starting up. Triggering backend initializations... ---")
    default_db = os.getenv("DEFAULT_DB_BACKEND", "MongoDB")
    initialize_all_components(default_db=default_db)
    print("--- Backend initializations complete. Gradio is ready. ---")


# ==============================================================================
#  Gradio UI Creation and Mounting
# ==============================================================================
print("--- Creating Gradio UI and mounting to FastAPI... ---")
gradio_app_instance = create_gradio_app()

# Mount the Gradio Blocks instance onto the FastAPI app.
app_fastapi = gr.mount_gradio_app(
    app=app_fastapi,
    blocks=gradio_app_instance,
    path="/"
)
print("--- Gradio UI mounted successfully. ---")


# ==============================================================================
#  Local Development Runner
# ==============================================================================
# This block is ONLY for running this file directly (e.g., `python gradio_server.py`)
if __name__ == "__main__":
    load_dotenv()
    print("--- (Local Dev): Launching Uvicorn server for Gradio... ---")
    
    # The startup_event will handle initializations when uvicorn.run is called.
    
    gradio_port = int(os.getenv("GRADIO_PORT", 7860))
    print(f"--- (Local Dev): Uvicorn will run on http://localhost:{gradio_port} ---")
    
    # Run the FastAPI app using uvicorn
    uvicorn.run(
        "gradio_server:app_fastapi", # Points to the FastAPI object in this file
        host="0.0.0.0",
        port=gradio_port,
        reload=True # Enables auto-reloading for easy development
    )