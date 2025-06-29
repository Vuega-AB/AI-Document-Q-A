import os


from dotenv import load_dotenv

# Import the function that creates the Gradio UI
from gradio_ui import create_gradio_app
# Import the backend initialization function
from backend import initialize_all_components

GRADIO_SERVER_INITIALIZED = False

def run_gradio_server_initializations():
    global GRADIO_SERVER_INITIALIZED
    if not GRADIO_SERVER_INITIALIZED:
        print("Running initializations for Gradio Server process...")
        # Example: Get default DB from env or use MongoDB
        default_db = os.getenv("DEFAULT_DB_BACKEND", "MongoDB")
        initialize_all_components(default_db=default_db)
        GRADIO_SERVER_INITIALIZED = True
        print("Gradio Server initializations completed.")
    else:
        print("Gradio Server initializations already run.")


if __name__ == "__main__":
    load_dotenv() # Load .env for local development; Render uses env vars

    print("Starting Gradio server process...")
    run_gradio_server_initializations() # Initialize backend components

    print("Creating Gradio app object...")
    # This 'demo' is the gr.Blocks instance returned by create_gradio_app()
    demo = create_gradio_app()

    # Get the port from the environment variable set by Render (or default for local)
    gradio_port = int(os.getenv("GRADIO_PORT", 7860))
    print(f"Attempting to launch Gradio server on 0.0.0.0:{gradio_port}...")

    try:
        demo.launch(
            server_name="0.0.0.0",
            server_port=gradio_port,
            share=False,
            inbrowser=False,
        )
        print(f"Gradio server successfully launched and should be listening on port {gradio_port}.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to launch Gradio server on port {gradio_port}.")
        print(f"Error details: {e}")
        # Consider exiting if launch fails, so Render knows it's a hard failure
        import sys
        sys.exit(1)