# gradio_server.py
import os
import uvicorn
from fastapi import FastAPI
from dotenv import load_dotenv

# Import your application's functions
from gradio_ui import create_gradio_app
from backend import initialize_all_components

# --- FastAPI App Setup ---
# Create a FastAPI app object. This will be our main entry point for the server.
app_fastapi = FastAPI()

# --- Deferred Initialization using FastAPI's startup event ---
# This is the key to preventing deployment timeouts. The heavy ML models
# will only load once the server is already live.
@app_fastapi.on_event("startup")
async def startup_event():
    print("--- Gradio Server starting up. Triggering backend initializations... ---")
    default_db = os.getenv("DEFAULT_DB_BACKEND", "MongoDB")
    initialize_all_components(default_db=default_db)
    print("--- Backend initializations complete. Gradio is ready. ---")

# --- Gradio UI Creation and Mounting ---
print("--- Creating Gradio UI and mounting to FastAPI... ---")
gradio_app_instance = create_gradio_app()

# Mount the Gradio Blocks instance onto the FastAPI app.
# This makes the Gradio UI available at the root path ("/").
app_fastapi = gr.mount_gradio_app(
    app=app_fastapi,
    blocks=gradio_app_instance,
    path="/"
)
print("--- Gradio UI mounted successfully. ---")


# This block is ONLY for running this file directly for local development
if __name__ == "__main__":
    load_dotenv()
    print("--- (Local Dev): Launching Uvicorn server for Gradio... ---")
    
    # We call this manually for local testing because the startup event
    # is handled by the uvicorn.run command itself.
    # Note: If Flask is also running locally, only one process should initialize.
    # initialize_all_components(default_db="MongoDB") 
    
    gradio_port = int(os.getenv("GRADIO_PORT", 7860))
    print(f"--- (Local Dev): Uvicorn will run on http://localhost:{gradio_port} ---")
    
    # Run the FastAPI app using uvicorn
    uvicorn.run(
        "gradio_server:app_fastapi", # Points to the FastAPI object in this file
        host="0.0.0.0",
        port=gradio_port,
        reload=True # Enables auto-reloading for easy development
    )