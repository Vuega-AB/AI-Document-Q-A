import gradio as gr
import os

from backend import (
    get_current_file_list_md_backend,
    get_unique_filenames_from_text_store,
    switch_db_backend,
    handle_pdf_upload_backend,
    chat_interface_backend,
    run_scraper_backend,
    delete_files_backend,
    apply_uploaded_config_backend,
    generate_config_for_download_backend,
    AVAILABLE_MODELS_NAMES,
    MODEL_NAME_TO_ID_MAP,
    BACKEND_INITIAL_LOAD_MSG,
)

CUSTOM_CSS = """
html, body { /* For the content *within* the iframe */
    height: 100%; /* Allow body to take full iframe height */
    margin: 0;
    padding: 0;
    /* overflow: hidden; */ /* REMOVE this from html,body - let content dictate scroll */
}

.gradio-container { 
    background-color: #f0f4f8 !important; 
    font-family: 'Inter', sans-serif;
    min-height: 100% !important; /* Ensure it at least fills the iframe height */
    height: auto !important;     /* Allow it to grow taller if content is long */
    max-width: none !important; 
    width: 100% !important;
    display: flex !important;
    flex-direction: column !important;
    /* overflow: hidden !important; */ /* REMOVE this from main container */
}

/* Header Row */
#app-header-row {
    padding: 10px 24px !important; 
    margin-bottom: 0 !important; 
    background-color: #ffffff !important; 
    border-bottom: 1px solid #e2e8f0 !important;
    flex-shrink: 0; /* Header should not shrink */
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    position: sticky; /* Make header sticky */
    top: 0;           /* Stick to the top */
    z-index: 100;     /* Ensure it's above other content */
}
/* ... (rest of #app-header-row and its children's styles as before) ... */
#app-title .gr-markdown > div > h1, #app-title .gr-markdown > div > p {
    font-size: 1.4rem !important; 
    font-weight: 600 !important;
    color: #1a202c; 
    margin: 0 !important; 
    line-height: 1.2 !important;
}
#button-column-container { 
    min-width: auto !important; 
    justify-content: flex-end; 
    align-items: center; 
    display: flex; 
    padding: 0 !important;
}
#actual-sign-out-btn.gr-button {
    background-color: #6b7280 !important; 
    color: white !important;
    padding: 5px 10px !important; 
    font-size: 0.8rem !important;   
    font-weight: 500 !important;
    border-radius: 6px !important; 
    border: 1px solid transparent !important;
    line-height: 1.2 !important; 
    min-width: auto !important; 
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
}
#actual-sign-out-btn.gr-button:hover {
    background-color: #4b5563 !important; 
}


/* Main Content Row - This is where the scroll should happen if needed, 
   but ideally, we let the overall page scroll */
#main-content-row.gr-row {
    flex-grow: 1 !important; /* Takes up remaining vertical space */
    /* overflow-y: auto !important; */ /* REMOVE this to let page scroll */
    overflow-x: hidden !important; 
    padding: 20px 24px !important; 
    gap: 24px !important;
    display: flex; /* Ensure it's flex if not already by gr-row */
    flex-direction: row; /* Default, but explicit */
}

/* Let the content within columns define their height */
.controls-column, .chat-column {
    /* Remove fixed heights or overflow settings here unless a component *within* them needs scrolling,
       like the chatbot output itself. */
    display: flex; /* Make columns flex containers for their content */
    flex-direction: column; /* Stack content vertically within columns */
}
.chat-column .gradio-chatbot { /* Specifically the chatbot component */
    flex-grow: 1; /* Allow chatbot to take available space in its column */
    /* Gradio's chatbot has its own internal scrolling if height is constrained and content overflows */
}


/* Chatbot messages & inputs (keep existing styles) */
.gradio-chatbot .message.bot { background-color: #2563eb !important; color: #ffffff !important; border-radius: 1rem 0.25rem 1rem 1rem !important; padding: 0.75rem 1rem !important; box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important; max-width: 80% !important; align-self: flex-start !important; margin-left: 0.5rem; margin-right: auto; }
.gradio-chatbot .message.user { background-color: #e0f2fe !important; color: #0c4a6e !important; border-radius: 0.25rem 1rem 1rem 1rem !important; padding: 0.75rem 1rem !important; box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important; max-width: 80% !important; align-self: flex-end !important; margin-right: 0.5rem; margin-left: auto; }
.gradio-chatbot .messages { padding: 1rem !important; display: flex; flex-direction: column; gap: 0.75rem; }
#chat-input-container textarea { border-radius: 0.75rem !important; border: 1px solid #cbd5e1 !important; padding: 0.75rem !important; }
#chat-input-container button { height: 100%; }
.gr-slider .track, .gr-slider .thumb { height: 1rem !important; }
.gr-slider .thumb { width: 1rem !important; }

/* General Button styles */
.gr-button { border-radius: 0.5rem !important; font-weight: 500 !important; padding: 0.6rem 1.2rem !important; }
.gr-button.gr-button-primary { background-color: #1d4ed8 !important; color: white !important; } 
.gr-button.stop { background-color: #dc2626 !important; color: white !important; } 

/* Responsive adjustments */
@media (max-width: 768px) { 
    #main-content-row.gr-row { 
        flex-direction: column !important; 
        padding: 16px !important;
        gap: 16px !important;
    } 
    .controls-column, .chat-column {{ min-width: 100% !important; }} 
    #app-header-row {{ padding: 8px 12px !important; }}
    #app-title .gr-markdown > div > h1, #app-title .gr-markdown > div > p {{ font-size: 1.2rem !important; }}
}
"""

FLASK_BASE_URL = os.getenv("FLASK_BASE_URL", "http://localhost:5000")

def js_logout_function():
    flask_logout_url = f"{FLASK_BASE_URL}/logout"
    return f"""
    () => {{
        console.log('Attempting to log out by redirecting top window to: {flask_logout_url}');
        window.top.location.href = '{flask_logout_url}';
        return []; 
    }}
    """

def _show_status_popup(message, is_success):
    if message:
        if is_success: gr.Info(message)
        else: gr.Warning(message)

def create_gradio_app():
    original_theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.gray, 
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    )
    # ... (rest of your create_gradio_app function as provided in the previous prompt)
    # The important part is that the CSS above is applied to this structure.
    with gr.Blocks(
        css=CUSTOM_CSS,
        theme=original_theme,
        title="IntelLaw Gradio",
    ) as demo:

        # --- Header Row for Title and Sign Out Button ---
        with gr.Row(elem_id="app-header-row", equal_height=False, variant="compact"):
            with gr.Column(scale=10, min_width=100): 
                gr.Markdown("# 📄 IntelLaw", elem_id="app-title")
            with gr.Column(scale=1, min_width=100, elem_id="button-column-container"): 
                dummy_output_for_js = gr.Textbox(visible=False, label="Dummy JS Output")
                sign_out_btn = gr.Button("Sign Out", elem_id="actual-sign-out-btn", scale=0)

        sign_out_btn.click(
            fn=None, inputs=None, outputs=[dummy_output_for_js], js=js_logout_function()
        )
        
        selected_db_state = gr.State(value="Dropbox")
        initial_selected_model_id = []
        if AVAILABLE_MODELS_NAMES and MODEL_NAME_TO_ID_MAP and AVAILABLE_MODELS_NAMES and MODEL_NAME_TO_ID_MAP.get(AVAILABLE_MODELS_NAMES[0]):
            first_model_name = AVAILABLE_MODELS_NAMES[0]
            initial_selected_model_id = [MODEL_NAME_TO_ID_MAP[first_model_name]]
        
        app_config_state_dict = {
            "selected_models": initial_selected_model_id,
            "vary_temperature": True, "temperature": 0.7,
            "vary_top_p": False, "top_p": 0.9,
            "system_prompt": (
                "You are a helpful assistant. Answer questions strictly based on the provided context. "
                "If there is no context, say 'I don't have enough information to answer that.'"
            )
        }
        app_config_state = gr.State(value=app_config_state_dict)
        status_msg_for_popup = gr.State()
        success_flag_for_popup = gr.State()

        # --- Main Content Row ---
        with gr.Row(equal_height=False, elem_id="main-content-row"): 
            with gr.Column(scale=35, min_width=380, elem_classes=["controls-column"]):
                gr.Markdown("### 🛠️ Controls")
                with gr.Accordion("Database Backend", open=True):
                    db_radio = gr.Radio(label="Choose Database", choices=["Dropbox", "MongoDB"], value=selected_db_state.value)
                    db_status = gr.Textbox(label="DB Status", interactive=False, value=BACKEND_INITIAL_LOAD_MSG)

                with gr.Tabs():
                    with gr.TabItem("🧠 Config"):
                        model_selector_default_value = []
                        if AVAILABLE_MODELS_NAMES and MODEL_NAME_TO_ID_MAP.get(AVAILABLE_MODELS_NAMES[0]):
                             model_selector_default_value = [AVAILABLE_MODELS_NAMES[0]]
                        
                        model_selector = gr.Dropdown(
                            label="AI Models (Max 3)", choices=AVAILABLE_MODELS_NAMES,
                            value=model_selector_default_value,
                            multiselect=True, max_choices=3, interactive=True
                        )
                        vary_temp = gr.Checkbox(label="Vary Temperature", value=app_config_state.value["vary_temperature"], interactive=True)
                        temp_slider = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, step=0.01, value=app_config_state.value["temperature"], interactive=True)
                        vary_top_p = gr.Checkbox(label="Vary Top-P", value=app_config_state.value["vary_top_p"], interactive=True)
                        top_p_slider = gr.Slider(label="Top-P", minimum=0.0, maximum=1.0, step=0.01, value=app_config_state.value["top_p"], interactive=True)
                        system_prompt = gr.Textbox(label="System Prompt", lines=4, value=app_config_state.value["system_prompt"], interactive=True)
                        
                        gr.Markdown("---") 
                        gr.Markdown("#### Configuration File Management") 
                        upload_config_btn = gr.UploadButton("Upload & Apply Config (JSON)", file_types=[".json"])
                        download_config_btn = gr.DownloadButton("Download Current Config")

                    with gr.TabItem("📁 Stored Files"):
                        file_uploader = gr.Files(label="Upload PDFs", file_count="multiple", type="filepath", file_types=[".pdf"])
                        upload_status = gr.Textbox(label="Upload Status", interactive=False, lines=3)
                        initial_filenames = get_unique_filenames_from_text_store()
                        files_to_delete_checkboxgroup = gr.CheckboxGroup(label="Select Files to Delete", choices=initial_filenames, value=[])
                        delete_files_button = gr.Button("Delete Selected Files", variant="stop")
                        delete_status_text = gr.Textbox(label="Deletion Status", interactive=False)
                        stored_files_md = gr.Markdown(value=get_current_file_list_md_backend())

                    with gr.TabItem("🌐 Web Scraper"):
                        base_url_input = gr.Textbox(label="Base URL", value="https://www.imy.se")
                        listing_endpoint_input = gr.Textbox(label="Listing Endpoint", value="tillsyner")
                        pagination_input = gr.Textbox(label="Pagination Format", value="?query=&page=")
                        pages_to_check_input = gr.Number(label="Num Pages to Check", value=1, minimum=1, precision=0)
                        scrape_btn = gr.Button("Start Scraping & Process", variant="secondary")
                        scraper_output = gr.Textbox(label="Scraper Output", lines=5, max_lines=10, interactive=False)

            with gr.Column(scale=65, elem_classes=["chat-column"]): 
                gr.Markdown("### 💬 Chat Interface")
                chatbot = gr.Chatbot(label="IntelLaw Chatbot", height=700, show_copy_button=True, bubble_full_width=False,type="messages" )
                with gr.Row(elem_id="chat-input-container"):
                    chat_input = gr.Textbox(show_label=False, placeholder="Ask anything about the documents...", scale=5, container=False)
                    send_btn = gr.Button("Send", scale=1, variant="primary")
        
        # --- Event Handlers ---
        db_radio.change(
            fn=switch_db_backend,
            inputs=[db_radio, selected_db_state],
            outputs=[selected_db_state, db_status, files_to_delete_checkboxgroup, stored_files_md]
        )
        upload_config_btn.upload(
            fn=apply_uploaded_config_backend,
            inputs=[upload_config_btn, app_config_state],
            outputs=[
                status_msg_for_popup, success_flag_for_popup,
                app_config_state,
                model_selector, vary_temp, temp_slider,
                vary_top_p, top_p_slider, system_prompt
            ]
        ).then(
            fn=_show_status_popup,
            inputs=[status_msg_for_popup, success_flag_for_popup],
            outputs=None
        )
        def _generate_and_get_path_only(app_config_state_value):
            filepath, status_msg, success = generate_config_for_download_backend(app_config_state_value)
            if filepath is None: gr.Warning(status_msg or "Failed to generate file for download.")
            elif success: gr.Info(status_msg or "File ready for download.")
            return filepath
        download_config_btn.click(
            fn=_generate_and_get_path_only,
            inputs=[app_config_state],
            outputs=[download_config_btn]
        )
        file_uploader.upload(
            fn=handle_pdf_upload_backend,
            inputs=[file_uploader, selected_db_state],
            outputs=[upload_status, files_to_delete_checkboxgroup, stored_files_md]
        )
        delete_files_button.click(
            fn=delete_files_backend,
            inputs=[files_to_delete_checkboxgroup, selected_db_state],
            outputs=[delete_status_text, files_to_delete_checkboxgroup, stored_files_md]
        )
        chat_inputs = [chat_input, chatbot, selected_db_state, app_config_state]
        chat_outputs = [chatbot] 
        def clear_input_fn(): return gr.update(value="")
        send_btn.click(fn=chat_interface_backend, inputs=chat_inputs, outputs=chat_outputs).then(fn=clear_input_fn, inputs=None, outputs=chat_input)
        chat_input.submit(fn=chat_interface_backend, inputs=chat_inputs, outputs=chat_outputs).then(fn=clear_input_fn, inputs=None, outputs=chat_input)
        scrape_btn.click(
            fn=run_scraper_backend,
            inputs=[base_url_input, listing_endpoint_input, pagination_input, pages_to_check_input, selected_db_state],
            outputs=[scraper_output, files_to_delete_checkboxgroup, stored_files_md]
        )
        return demo