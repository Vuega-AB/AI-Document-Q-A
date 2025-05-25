# gradio_server.py
from gradio_ui import create_gradio_app
from backend import initialize_all_components, MONGO_URI 

GRADIO_SERVER_INITIALIZED = False

def run_gradio_server_initializations():
    global GRADIO_SERVER_INITIALIZED
    if not GRADIO_SERVER_INITIALIZED:
        print("Running initializations for Gradio Server process...")
        if not MONGO_URI: 
            print("CRITICAL: MONGO_URI is not set in .env (for Gradio Server).")
        else:
            print(f"MongoDB URI found for Gradio Server: {MONGO_URI[:20]}...")

        initialize_all_components(default_db="MongoDB") 

        GRADIO_SERVER_INITIALIZED = True
    else:
        print("Gradio Server initializations already run.")


if __name__ == "__main__":
    run_gradio_server_initializations() # <<<< CALL INITIALIZATIONS HERE

    print("Creating Gradio app for Gradio Server process...")
    demo = create_gradio_app() 

    print("Launching Gradio server on 0.0.0.0:7860...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,     
        inbrowser=False,    
    )