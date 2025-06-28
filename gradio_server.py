# gradio_server.py

import os
import uvicorn
from fastapi import FastAPI
from dotenv import load_dotenv
import gradio as gr

# Import only the functions needed to build the UI and the backend gatekeeper.
# We DO NOT import or call initialize_all_components directly anymore.
from gradio_ui import create_gradio_app
from backend import ensure_backend_is_initialized # If you want to prime it on first load

# ==============================================================================
#  FastAPI App Setup
# ==============================================================================
# Create a FastAPI app object. This will be our main entry point for the server.
app_fastapi = FastAPI()

# --- REMOVED: No more automatic startup initialization ---
# The @app_fastapi.on_event("startup") block has been removed entirely.
# This ensures the server starts instantly without loading heavy models.

# ==============================================================================
#  Gradio UI Creation and Mounting
# ==============================================================================
print("--- Creating Gradio UI object... ---")
gradio_app_instance = create_gradio_app()

print("--- Mounting Gradio UI to FastAPI at root path '/'... ---")
# Mount the Gradio Blocks instance onto the FastAPI app.
app_fastapi = gr.mount_gradio_app(
    app=app_fastapi,
    blocks=gradio_app_instance,
    path="/"
)
print("--- Gradio UI mounted successfully. Server is ready. ---")


# ==============================================================================
#  Local Development Runner
# ==============================================================================
# This block is ONLY for running this file directly (e.g., `python gradio_server.py`)
if __name__ == "__main__":
    load_dotenv()
    print("--- (Local Dev): Launching Uvicorn server for Gradio... ---")
    
    # For local development, you might want to pre-load the models
    # to simulate the user's first action immediately.
    # To do this, you would uncomment the following line:
    # ensure_backend_is_initialized()
    
    gradio_port = int(os.getenv("GRADIO_PORT", 7860))
    print(f"--- (Local Dev): Uvicorn will run on http://localhost:{gradio_port} ---")
    
    # Run the FastAPI app using uvicorn
    uvicorn.run(
        "gradio_server:app_fastapi", # Points to the FastAPI object in this file
        host="0.0.0.0",
        port=gradio_port,
        reload=True # Enables auto-reloading for easy development
    )