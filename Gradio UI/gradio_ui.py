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
    MODEL_ID_TO_NAME_MAP,
    get_backend_initial_load_message
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

def js_settings_function():
    flask_settings_url = f"{FLASK_BASE_URL}/settings"
    return f"""
    () => {{
        console.log('Redirecting to settings page: {flask_settings_url}');
        window.top.location.href = '{flask_settings_url}';
        return []; 
    }}
    """

def _show_status_popup(message, is_success):
    if message:
        if is_success: gr.Info(message)
        else: gr.Warning(message)

def get_initial_ui_data_for_gradio():
    print("DEBUG: get_initial_ui_data_for_gradio called for Gradio load event")
    initial_db_status_msg = get_backend_initial_load_message()
    initial_filenames_choices = get_unique_filenames_from_text_store()
    stored_files_md_str = get_current_file_list_md_backend()
    print(f"DEBUG: Initial DB Status from backend: {initial_db_status_msg}")
    return initial_db_status_msg, stored_files_md_str, initial_filenames_choices


def create_gradio_app():
    app_theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.neutral,
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
        radius_size=gr.themes.sizes.radius_sm, # Smaller radius for a tighter look
        spacing_size=gr.themes.sizes.spacing_md,
    ).set(
        button_primary_background_fill="*primary_500", # Standard blue
        button_primary_background_fill_hover="*primary_600",
        button_primary_text_color="white",
        button_secondary_background_fill="*neutral_100", # Light gray for secondary
        button_secondary_background_fill_hover="*neutral_300",
        button_secondary_text_color="*neutral_700",
    )

    with gr.Blocks(
        css=CUSTOM_CSS,
        theme=app_theme, 
        title="IntelLaw Gradio",
    ) as demo:

        # --- Header Row ---
        with gr.Row(elem_id="app-header-row", equal_height=False, variant="panel"):
            with gr.Column(scale=9, min_width=150): 
                gr.Markdown("# 📄 IntelLaw", elem_id="app-title")
            with gr.Column(scale=3, min_width=200, elem_id="button-column-container"): 
                dummy_output_for_js = gr.Textbox(visible=False, label="Dummy JS Output")
                settings_btn = gr.Button("Settings", elem_id="settings-btn", scale=0, visible=False) 
                sign_out_btn = gr.Button("Sign Out", elem_id="actual-sign-out-btn", scale=0)

        sign_out_btn.click(fn=None, inputs=None, outputs=[dummy_output_for_js], js=js_logout_function())
        settings_btn.click(fn=None, inputs=None, outputs=[dummy_output_for_js], js=js_settings_function())
        
        js_to_run_on_load_for_settings = """
            () => { 
                const params = new URLSearchParams(window.location.search);
                const role = params.get('user_role');
                const settingsButton = document.getElementById('settings-btn');
                if (settingsButton) {
                    if (role === 'admin') {
                        settingsButton.style.display = 'inline-flex'; 
                    } else {
                        settingsButton.style.display = 'none';
                    }
                }
                return [];
            }
        """
        
        selected_db_state = gr.State(value="MongoDB") 
        initial_selected_model_id = []
        if AVAILABLE_MODELS_NAMES and MODEL_NAME_TO_ID_MAP and AVAILABLE_MODELS_NAMES and MODEL_NAME_TO_ID_MAP.get(AVAILABLE_MODELS_NAMES[0]):
            first_model_name = AVAILABLE_MODELS_NAMES[0]
            initial_selected_model_id = [MODEL_NAME_TO_ID_MAP[first_model_name]]
        
        app_config_state_dict = {
            "selected_models": initial_selected_model_id,
            "vary_temperature": True, "temperature": 0.7, "vary_top_p": False, "top_p": 0.9,
            "system_prompt": "You are a helpful assistant..."
        }
        app_config_state = gr.State(value=app_config_state_dict)
        status_msg_for_popup = gr.State()
        success_flag_for_popup = gr.State()


        with gr.Row(equal_height=False, elem_id="main-content-row"): 
            with gr.Column(scale=35, elem_classes=["controls-column"]): 
                gr.Markdown("### 🛠️ Controls")
                with gr.Accordion("Database Backend", open=True):
                    db_radio_output = gr.Radio(label="Choose Database", choices=["Dropbox", "MongoDB"], value=selected_db_state.value)
                    db_status_output = gr.Textbox(label="DB Status", interactive=False) 

                with gr.Tabs():
                    with gr.TabItem("🧠 Config"):
                        model_selector_output = gr.Dropdown(label="AI Models (Max 3)", choices=AVAILABLE_MODELS_NAMES, multiselect=True, max_choices=3, interactive=True)
                        vary_temp_output = gr.Checkbox(label="Vary Temperature", interactive=True)
                        temp_slider_output = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, step=0.01, interactive=True)
                        vary_top_p_output = gr.Checkbox(label="Vary Top-P", interactive=True)
                        top_p_slider_output = gr.Slider(label="Top-P", minimum=0.0, maximum=1.0, step=0.01, interactive=True)
                        system_prompt_output = gr.Textbox(label="System Prompt", lines=4, interactive=True) # Increased lines for more content
                        gr.Markdown("---") 
                        gr.Markdown("#### Configuration File Management") 
                        upload_config_btn = gr.UploadButton("Upload & Apply Config (JSON)", file_types=[".json"], variant="secondary", size="sm")
                        download_config_btn = gr.DownloadButton("Download Current Config", variant="secondary", size="sm")

                    with gr.TabItem("📁 Stored Files"):
                        file_uploader = gr.Files(label="Upload PDFs", file_count="multiple", type="filepath", file_types=[".pdf"])
                        upload_status_output = gr.Textbox(label="Upload Status", interactive=False, lines=3)
                        files_to_delete_checkboxgroup_output = gr.CheckboxGroup(label="Select Files to Delete", choices=[], value=[])
                        delete_files_button = gr.Button("Delete Selected", variant="stop", size="sm")
                        delete_status_text_output = gr.Textbox(label="Deletion Status", interactive=False, lines=2) # Increased lines
                        stored_files_md_output = gr.Markdown("Long content to test scrolling.\n" * 50) # Added long content
                        

                    with gr.TabItem("🌐 Web Scraper"):
                        base_url_input = gr.Textbox(label="Base URL", value="https://www.imy.se")
                        listing_endpoint_input = gr.Textbox(label="Listing Endpoint", value="tillsyner")
                        pagination_input = gr.Textbox(label="Pagination Format", value="?query=&page=")
                        pages_to_check_input = gr.Number(label="Pages to Check", value=1, minimum=1, precision=0)
                        scrape_btn = gr.Button("Start Scraping", variant="secondary", size="sm")
                        scraper_output_display = gr.Textbox(label="Scraper Output", lines=5, interactive=False)

            with gr.Column(scale=65, elem_classes=["chat-column"]): 
                gr.Markdown("### 💬 Chat Interface")
                chatbot_output = gr.Chatbot(label="IntelLaw Chatbot", height=600, show_copy_button=True, bubble_full_width=False,type="messages" ) # Adjusted height
                with gr.Row(elem_id="chat-input-container"):
                    chat_input_box = gr.Textbox(show_label=False, placeholder="Ask any question...", scale=5, container=False)
                    send_btn = gr.Button("Send", scale=1, variant="primary")
        
        def update_ui_on_load(current_app_config_dict):
            db_status_val, stored_files_md_str, filenames_choices = get_initial_ui_data_for_gradio()
            
            model_names_for_ui = [MODEL_ID_TO_NAME_MAP[mid] for mid in current_app_config_dict.get("selected_models", []) if mid in MODEL_ID_TO_NAME_MAP]
            if not model_names_for_ui and AVAILABLE_MODELS_NAMES:
                model_names_for_ui = [AVAILABLE_MODELS_NAMES[0]]

            return {
                db_status_output: db_status_val,
                stored_files_md_output: stored_files_md_str,
                files_to_delete_checkboxgroup_output: gr.update(choices=filenames_choices, value=[]),
                model_selector_output: model_names_for_ui,
                vary_temp_output: current_app_config_dict.get("vary_temperature", True),
                temp_slider_output: current_app_config_dict.get("temperature", 0.7),
                vary_top_p_output: current_app_config_dict.get("vary_top_p", False),
                top_p_slider_output: current_app_config_dict.get("top_p", 0.9),
                system_prompt_output: current_app_config_dict.get("system_prompt", "You are a helpful assistant..."),
                db_radio_output: selected_db_state.value 
            }

        demo.load(
            fn=update_ui_on_load, inputs=[app_config_state], 
            outputs=[
                db_status_output, stored_files_md_output, files_to_delete_checkboxgroup_output,
                model_selector_output, vary_temp_output, temp_slider_output,
                vary_top_p_output, top_p_slider_output, system_prompt_output,
                db_radio_output
            ]
        ).then(None, None, None, js=js_to_run_on_load_for_settings)

        # --- Event Handlers ---
        db_radio_output.change(
            fn=switch_db_backend,
            inputs=[db_radio_output, selected_db_state],
            outputs=[selected_db_state, db_status_output, files_to_delete_checkboxgroup_output, stored_files_md_output]
        )
        upload_config_btn.upload(
            fn=apply_uploaded_config_backend,
            inputs=[upload_config_btn, app_config_state],
            outputs=[
                status_msg_for_popup, success_flag_for_popup, app_config_state,
                model_selector_output, vary_temp_output, temp_slider_output, 
                vary_top_p_output, top_p_slider_output, system_prompt_output
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
            outputs=[upload_status_output, files_to_delete_checkboxgroup_output, stored_files_md_output]
        )
        delete_files_button.click(
            fn=delete_files_backend,
            inputs=[files_to_delete_checkboxgroup_output, selected_db_state],
            outputs=[delete_status_text_output, files_to_delete_checkboxgroup_output, stored_files_md_output]
        )
        chat_inputs = [chat_input_box, chatbot_output, selected_db_state, app_config_state]
        chat_outputs = [chatbot_output] 
        def clear_input_fn(): return gr.update(value="")
        send_btn.click(fn=chat_interface_backend, inputs=chat_inputs, outputs=chat_outputs).then(fn=clear_input_fn, inputs=None, outputs=[chat_input_box])
        chat_input_box.submit(fn=chat_interface_backend, inputs=chat_inputs, outputs=chat_outputs).then(fn=clear_input_fn, inputs=None, outputs=[chat_input_box])
        scrape_btn.click(
            fn=run_scraper_backend,
            inputs=[base_url_input, listing_endpoint_input, pagination_input, pages_to_check_input, selected_db_state],
            outputs=[scraper_output_display, files_to_delete_checkboxgroup_output, stored_files_md_output]
        )
        return demo