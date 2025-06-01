# gradio_ui.py

import gradio as gr
import os

# Ensure these imports point to your actual backend.py file and its functions
from backend import (
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
    get_backend_initial_load_message,
)

FLASK_BASE_URL = os.getenv("FLASK_BASE_URL", "http://localhost:5000")

def js_logout_function():
    flask_logout_url = f"{FLASK_BASE_URL}/logout"
    return f"""
    () => {{
        window.top.location.href = '{flask_logout_url}';
        return [];
    }}
    """

def js_settings_function():
    flask_settings_url = f"{FLASK_BASE_URL}/settings"
    return f"""
    () => {{
        window.top.location.href = '{flask_settings_url}';
        return [];
    }}
    """

def _show_status_popup(message, is_success):
    if message:
        if is_success:
            gr.Info(message)
        else:
            gr.Warning(message)

def get_initial_ui_data_for_gradio():
    initial_db_status_msg = get_backend_initial_load_message()
    initial_filenames_choices = get_unique_filenames_from_text_store()
    return initial_db_status_msg, initial_filenames_choices

CUSTOM_CSS = """
/* Structural CSS that might still be relevant with the Miku theme */
/* You'll need to experiment to see what's needed and what conflicts */

#app-header-row {
    /* The Miku theme might style the header, check if this is still needed */
    /* padding: 10px 24px !important; */
    /* background-color: var(--primary-bg-light) !important; */ /* Miku theme will handle this */
    /* border-bottom: 1px solid #dee2e6 !important; */ /* Miku theme will handle this */
    position: sticky !important; /* This might be useful to keep header sticky */
    top: 0;
    z-index: 100;
    /* display: flex !important; align-items: center !important; */ /* Theme should handle flex */
    width: 100% !important; /* This might be useful */
    box-sizing: border-box;
}

#app-title .gr-markdown h1 {
    /* The Miku theme will have its own h1 styling */
    /* You can override if you need a specific font size or color not matching the theme */
    /* margin: 0; padding: 0; font-size: 1.4rem !important; */
}

#main-content-row {
    flex-grow: 1 !important; /* Keep */
    display: flex !important; /* Keep */
    flex-direction: row !important; /* Keep */
    padding: 20px !important; /* Adjust if Miku theme's padding is better/worse */
    gap: 24px !important; /* Adjust based on Miku theme */
    overflow: hidden !important; /* Keep */
    min-height: 0 !important; /* Keep */
    width: 100% !important; /* Keep */
    box-sizing: border-box; /* Keep */
}

.controls-column {
    /* These define the width of your controls column, likely still needed */
    flex: 0 0 450px !important;
    max-width: 500px !important;
    /* The theme will style the column's appearance (border, background) */
}

.chat-column {
    /* This defines how the chat column takes up space, likely still needed */
    flex-grow: 1 !important;
    min-width: 300px !important;
    /* The theme will style the column's appearance */
}

/*
   It's highly recommended to remove most of your previous color, background,
   border, font, and detailed component styling (buttons, inputs, chatbot messages)
   from CUSTOM_CSS when using a comprehensive theme like "NoCrypt/miku".
   The theme is designed to handle these. Your custom CSS will likely
   override and break the theme's intended appearance or cause conflicts.

   Start with minimal structural CSS like above and add back specific overrides
   only if absolutely necessary after observing how the Miku theme renders your UI.
*/
"""

def create_gradio_app():
    # No longer defining a custom theme object if using a Hugging Face theme string
    # theme = gr.themes.Default(...)

    with gr.Blocks(
        css=CUSTOM_CSS,  # Use your (now reduced) CUSTOM_CSS
        theme='lone17/kotaemon',  # Apply the Miku theme
        title="IntelLaw Gradio"
    ) as demo:
        # Header
        with gr.Row(elem_id="app-header-row"):
            gr.Markdown("# 📄 IntelLaw", elem_id="app-title")
            with gr.Row(elem_id="button-column-container"): # Might not need elem_id if theme handles it
                dummy = gr.Textbox(visible=False)
                settings_btn = gr.Button("Settings", elem_id="settings-btn", variant="secondary", visible=False)
                sign_out_btn = gr.Button("Sign Out", elem_id="actual-sign-out-btn", variant="secondary")

        settings_btn.click(fn=None, inputs=None, outputs=[dummy], js=js_settings_function())
        sign_out_btn.click(fn=None, inputs=None, outputs=[dummy], js=js_logout_function())

        # Main content
        with gr.Row(elem_id="main-content-row"):
            # Controls Column
            with gr.Column(elem_classes=["controls-column"], visible=False) as controls_column_element:
                gr.Markdown("### 🛠️ Controls") # Theme will style this
                with gr.Accordion("Database Backend", open=True):
                    db_radio = gr.Radio(label="Choose Database", choices=["Dropbox", "MongoDB"], value="MongoDB")
                    db_status = gr.Textbox(label="DB Status", interactive=False)
                with gr.Tabs():
                    with gr.TabItem("🧠 Config"):
                        model_selector = gr.Dropdown(
                            label="AI Models (Max 3)",
                            choices=AVAILABLE_MODELS_NAMES,
                            multiselect=True,
                            max_choices=3
                        )
                        vary_temp = gr.Checkbox(label="Vary Temperature", value=True)
                        temp_slider = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, step=0.01, value=0.7)
                        vary_top_p = gr.Checkbox(label="Vary Top-P", value=False)
                        top_p_slider = gr.Slider(label="Top-P", minimum=0.0, maximum=1.0, step=0.01, value=0.9)
                        system_prompt = gr.Textbox(label="System Prompt", lines=4, value="Answer questions strictly based on the provided context. If there is no context, say 'I don't have enough information to answer that.'")
                        gr.Markdown("---")
                        gr.Markdown("#### Configuration File Management")
                        upload_config_btn = gr.UploadButton("Upload & Apply Config (JSON)", file_types=[".json"], variant="secondary", size="sm")
                        download_config_btn = gr.DownloadButton("Download Current Config", variant="secondary", size="sm")
                    with gr.TabItem("📁 Stored Files"):
                        file_uploader = gr.Files(label="Upload PDFs", file_count="multiple", type="filepath", file_types=[".pdf"])
                        upload_status = gr.Textbox(label="Upload Status", interactive=False, lines=3)
                        files_to_delete = gr.CheckboxGroup(label="Select Files to Delete", choices=[], value=[])
                        delete_btn = gr.Button("Delete Selected", variant="stop", size="sm")
                        delete_status = gr.Textbox(label="Deletion Status", interactive=False, lines=2)
                    with gr.TabItem("🌐 Web Scraper"):
                        base_url = gr.Textbox(label="Base URL", value="https://www.imy.se")
                        listing_endpoint = gr.Textbox(label="Listing Endpoint", value="tillsyner")
                        pagination = gr.Textbox(label="Pagination Format", value="?query=&page=")
                        pages_to_check = gr.Number(label="Pages to Check", value=1, minimum=1, precision=0)
                        scrape_btn = gr.Button("Start Scraping", variant="secondary", size="sm")
                        scraper_output = gr.Textbox(label="Scraper Output", lines=5, interactive=False)

            # Chat Column
            with gr.Column(elem_classes=["chat-column"]) as chat_column_element:
                gr.Markdown("### 💬 Chat Interface") # Theme will style this
                chatbot = gr.Chatbot(value=[], label="IntelLaw Chatbot", show_copy_button=True, bubble_full_width=False)
                with gr.Row(elem_id="chat-input-container"): # Theme might style this area
                    chat_input = gr.Textbox(show_label=False, placeholder="Ask any question...", container=False)
                    send_btn = gr.Button("Send", variant="primary")

        # States and Load logic (handle_visibility_and_initial_load) remains the same
        selected_db_state = gr.State("MongoDB")
        initial_model_id = MODEL_NAME_TO_ID_MAP.get(AVAILABLE_MODELS_NAMES[0]) if AVAILABLE_MODELS_NAMES and AVAILABLE_MODELS_NAMES[0] in MODEL_NAME_TO_ID_MAP else None
        selected_models_init = [initial_model_id] if initial_model_id else []
        app_config_state = gr.State({
            "selected_models": selected_models_init,
            "vary_temperature": True, "temperature": 0.7,
            "vary_top_p": False, "top_p": 0.9,
            "system_prompt": "You are a helpful assistant..."
        })
        status_msg = gr.State()
        success_flag = gr.State()

        def handle_visibility_and_initial_load(request: gr.Request, current_app_config):
            user_role = "user" 
            if request and hasattr(request, "query_params") and request.query_params:
                user_role = request.query_params.get("user_role", "user")

            db_msg, files_choices = get_initial_ui_data_for_gradio()
            selected_model_ids_from_state = current_app_config.get("selected_models", [])
            initial_selected_model_names = [
                MODEL_ID_TO_NAME_MAP[m_id] for m_id in selected_model_ids_from_state if m_id in MODEL_ID_TO_NAME_MAP
            ]
            if not initial_selected_model_names and AVAILABLE_MODELS_NAMES:
                first_available_model_name = AVAILABLE_MODELS_NAMES[0]
                initial_selected_model_names = [first_available_model_name]
            
            admin_view = (user_role == "admin")

            return (
                gr.update(visible=admin_view),  
                gr.update(visible=admin_view),  
                gr.update(value=db_msg),
                gr.update(choices=files_choices, value=[]), 
                gr.update(value=initial_selected_model_names) 
            )

        demo.load(
            fn=handle_visibility_and_initial_load,
            inputs=[app_config_state], 
            outputs=[
                settings_btn,
                controls_column_element,
                db_status,          
                files_to_delete,    
                model_selector      
            ]
        )

        # Event handlers (remain the same, as they handle logic, not detailed styling)
        db_radio.change(
            fn=switch_db_backend,
            inputs=[db_radio, selected_db_state],
            outputs=[selected_db_state, db_status, files_to_delete]
        )
        def handle_config_change(models_names, vary_t, temp, vary_p, top_p, sys_prompt, current_config_state):
            selected_model_ids = [MODEL_NAME_TO_ID_MAP[name] for name in models_names if name in MODEL_NAME_TO_ID_MAP]
            if len(selected_model_ids) > 3:
                 gr.Warning("Maximum 3 AI models can be selected.")
                 selected_model_ids = selected_model_ids[:3]
            updated_config = current_config_state.copy()
            updated_config.update({
                "selected_models": selected_model_ids,
                "vary_temperature": vary_t, "temperature": temp,
                "vary_top_p": vary_p, "top_p": top_p,
                "system_prompt": sys_prompt
            })
            return updated_config
        config_inputs = [model_selector, vary_temp, temp_slider, vary_top_p, top_p_slider, system_prompt]
        for inp in config_inputs:
            inp.change(
                fn=handle_config_change,
                inputs=config_inputs + [app_config_state],
                outputs=[app_config_state]
            )
        upload_config_btn.upload(
            fn=apply_uploaded_config_backend,
            inputs=[upload_config_btn, app_config_state],
            outputs=[status_msg, success_flag, app_config_state, model_selector, vary_temp, temp_slider, vary_top_p, top_p_slider, system_prompt]
        ).then(fn=_show_status_popup, inputs=[status_msg, success_flag], outputs=None)
        def prepare_download(current_config_state):
            path, msg, ok = generate_config_for_download_backend(current_config_state)
            if not path and not ok: gr.Warning(msg) 
            elif ok and path: gr.Info(msg) 
            return path if (path and ok) else None             
        download_config_btn.click(
            fn=prepare_download,
            inputs=[app_config_state],
            outputs=[download_config_btn] 
        )
        file_uploader.upload(
            fn=handle_pdf_upload_backend,
            inputs=[file_uploader, selected_db_state],
            outputs=[upload_status, files_to_delete]
        )
        delete_btn.click(
            fn=delete_files_backend,
            inputs=[files_to_delete, selected_db_state],
            outputs=[delete_status, files_to_delete]
        )
        chat_inputs = [chat_input, chatbot, selected_db_state, app_config_state]
        send_btn.click(
            fn=chat_interface_backend,
            inputs=chat_inputs,
            outputs=[chatbot]
        ).then(fn=lambda: gr.update(value=""), inputs=None, outputs=[chat_input])
        chat_input.submit(
            fn=chat_interface_backend,
            inputs=chat_inputs,
            outputs=[chatbot]
        ).then(fn=lambda: gr.update(value=""), inputs=None, outputs=[chat_input])
        scrape_btn.click(
            fn=run_scraper_backend,
            inputs=[base_url, listing_endpoint, pagination, pages_to_check, selected_db_state],
            outputs=[scraper_output, files_to_delete]
        )
        return demo