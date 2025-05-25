import gradio as gr
import os

# Ensure these imports point to your actual backend.py file and its functions
from backend import (
    get_unique_filenames_from_text_store,
    switch_db_backend,
    handle_pdf_upload_backend,
    chat_interface_backend, # CRITICAL: This function must be correctly implemented in backend.py
    run_scraper_backend,
    delete_files_backend,
    apply_uploaded_config_backend,
    generate_config_for_download_backend,
    AVAILABLE_MODELS_NAMES,
    MODEL_NAME_TO_ID_MAP,
    MODEL_ID_TO_NAME_MAP,
    get_backend_initial_load_message,
)

# ==== CUSTOM STYLES ====
CUSTOM_CSS = """
:root {
    --primary-bg-light: #f8f9fa;    /* Light grey for header */
    --primary-text-light: #212529;  /* Dark text for light backgrounds */
    --accent-cyan: #0dcaf0;
    --accent-cyan-hover: #0baccc;
    --text-on-cyan: #000000;        /* Black text for good contrast on cyan */

    /* Dark Mode general variables */
    --light-text-dm: #e0e0e0;
    --dark-bg-dm: #121212;
    --dark-surface-dm: #1e1e1e;
    --dark-border-dm: #333333;
    --input-bg-dm: #2a2a2a;
    --button-secondary-bg-dm: #3a3a3a;
    --button-secondary-hover-bg-dm: #4a4a4a;
}

/* Global reset and light backgrounds (default) */
html, body {
    margin: 0;
    padding: 0;
    height: 100%;
    overflow: auto !important;
    background-color: #ffffff !important;
    color: #000000 !important;
    font-family: 'Inter', sans-serif;
}

/* Dark Mode Global Styles */
html.dark, html.dark body {
    background-color: var(--dark-bg-dm) !important;
    color: var(--light-text-dm) !important;
}
html.dark .gradio-container {
    background-color: var(--dark-bg-dm) !important;
    color: var(--light-text-dm) !important;
}

/* Header */
#app-header-row {
    padding: 10px 24px !important;
    background-color: var(--primary-bg-light) !important; /* LIGHT GREY header */
    border-bottom: 1px solid #dee2e6 !important; /* Standard light border */
    position: sticky !important;
    top: 0;
    z-index: 100;
    display: flex !important;
    align-items: center !important;
    width: 100% !important;
    box-sizing: border-box;
}
#app-title .gr-markdown h1 {
    margin: 0;
    padding: 0;
    font-size: 1.4rem !important;
    color: var(--primary-text-light) !important; /* Dark text on light header */
}
html.dark #app-header-row { /* Dark mode header */
    background-color: var(--dark-surface-dm) !important;
    border-bottom: 1px solid var(--dark-border-dm) !important;
}
html.dark #app-title .gr-markdown h1 {
    color: var(--light-text-dm) !important; /* Light text on dark header */
}


/* Main content structure */
#main-content-row {
    flex-grow: 1 !important;
    display: flex !important;
    flex-direction: row !important;
    padding: 20px !important;
    gap: 24px !important;
    overflow: hidden !important;
    min-height: 0 !important;
    width: 100% !important;
    box-sizing: border-box;
}

.controls-column, .chat-column {
    border: 1px solid #e0e0e0 !important;
    border-radius: 8px !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: hidden !important;
    padding: 0 !important;
    min-height: 0 !important;
    box-sizing: border-box;
    background-color: #ffffff !important;
}
html.dark .controls-column, html.dark .chat-column {
    background-color: var(--dark-surface-dm) !important;
    border: 1px solid var(--dark-border-dm) !important;
    color: var(--light-text-dm) !important;
}

.controls-column { flex: 0 0 450px !important; max-width: 500px !important; }
.chat-column    { flex-grow: 1 !important; min-width: 300px !important; }

.controls-column > .gr-form,
.chat-column    > .gr-form {
    display: flex !important;
    flex-direction: column !important;
    flex-grow: 1 !important;
    min-height: 0 !important;
    padding: 16px !important;
    box-sizing: border-box;
}

/* General text color in dark mode for various components */
html.dark .gr-markdown p, html.dark .gr-markdown li, html.dark .gr-markdown h3, html.dark .gr-markdown h4,
html.dark .gr-radio label span, html.dark .gr-checkbox label span,
html.dark .gr-slider > label span, html.dark .gr-textbox > label span,
html.dark .gr-dropdown > label span, html.dark .gr-number > label span,
html.dark .gr-accordion > label span,
html.dark .gr-file > .label span,
html.dark .gr-checkboxgroup > .label span,
html.dark .gr-tabs > .tab-nav > button:not(.selected) {
    color: var(--light-text-dm) !important;
}
html.dark label.label {
    color: var(--light-text-dm) !important;
}


/* Input fields */
.gradio-textbox textarea,
.gradio-dropdown input[type="text"],
.gr-number input[type="number"] {
    background-color: #ffffff !important;
    color: #212529 !important;
    border: 1px solid #ced4da !important;
    box-shadow: none !important;
    border-radius: 4px !important;
}
html.dark .gradio-textbox textarea,
html.dark .gradio-dropdown input[type="text"],
html.dark .gr-number input[type="number"] {
    background-color: var(--input-bg-dm) !important;
    color: var(--light-text-dm) !important;
    border: 1px solid var(--dark-border-dm) !important;
}
html.dark .gradio-dropdown .token-remove {
    color: var(--light-text-dm) !important;
}
/* Dropdown selected item - uses CYAN accent */
.gradio-dropdown ul.options > li.item.selected,
html.dark .gradio-dropdown ul.options > li.item.selected {
    background-color: var(--accent-cyan) !important;
    color: var(--text-on-cyan) !important; /* Black text on cyan */
}
html.dark .gradio-dropdown ul.options > li.item {
    color: #000000 !important;
}


/* Buttons */
/* Primary Button (e.g., Send) - CYAN */
button.gr-button-primary {
    background: var(--accent-cyan) !important;
    color: var(--text-on-cyan) !important; /* Black text on cyan button */
    border: 1px solid var(--accent-cyan) !important;
}
button.gr-button-primary:hover {
    background: var(--accent-cyan-hover) !important;
    border-color: var(--accent-cyan-hover) !important;
}

/* Secondary/Default Buttons (Settings, Sign Out, Upload, Download) */
#settings-btn, #actual-sign-out-btn,
button.gr-button-secondary,
.gr-upload-button > button.gr-button,
.gr-download-button > button.gr-button {
    background-color: #f0f0f0 !important;
    color: #212529 !important;
    border: 1px solid #ced4da !important;
}
#settings-btn:hover, #actual-sign-out-btn:hover,
button.gr-button-secondary:hover,
.gr-upload-button > button.gr-button:hover,
.gr-download-button > button.gr-button:hover {
    background-color: #e2e6ea !important;
}
/* Dark Mode for Secondary/Default Buttons */
html.dark #settings-btn, html.dark #actual-sign-out-btn,
html.dark button.gr-button-secondary,
html.dark .gr-upload-button > button.gr-button,
html.dark .gr-download-button > button.gr-button {
    background-color: var(--button-secondary-bg-dm) !important;
    color: var(--light-text-dm) !important;
    border: 1px solid var(--dark-border-dm) !important;
}
html.dark #settings-btn:hover, html.dark #actual-sign-out-btn:hover,
html.dark button.gr-button-secondary:hover,
html.dark .gr-upload-button > button.gr-button:hover,
html.dark .gr-download-button > button.gr-button:hover {
    background-color: var(--button-secondary-hover-bg-dm) !important;
}

/* Stop/Delete Buttons */
button.gr-button-stop {
    background-color: #dc3545 !important;
    color: white !important;
    border: 1px solid #dc3545 !important;
}
button.gr-button-stop:hover {
    background-color: #c82333 !important;
    border-color: #c82333 !important;
}


/* Tabs - Selected tab uses CYAN accent */
.gradio-tabs > .tab-nav > button.selected {
    border-bottom: 3px solid var(--accent-cyan) !important;
    color: var(--accent-cyan) !important;
}
html.dark .gradio-tabs > .tab-nav > button.selected {
    color: var(--accent-cyan) !important;
    border-bottom-color: var(--accent-cyan) !important;
}
html.dark .gradio-tabs > .tab-nav > button {
    color: var(--light-text-dm) !important;
}


/* Chatbot */
.gradio-chatbot {
    height: calc(100% - 70px) !important;
    min-height: 300px !important;
}
html.dark .gradio-chatbot {
    border: 1px solid var(--dark-border-dm);
}
/* Chat Messages - User message uses CYAN accent */
.gradio-chatbot .message.user {
    background: #cff4fc !important; /* Lighter cyan for user in light mode */
    color: var(--text-on-cyan) !important; /* Black text */
    align-self: flex-end;
}
.gradio-chatbot .message.bot {
    background: #F0F0F0 !important;
    color: #000000 !important;
    align-self: flex-start;
}
html.dark .gradio-chatbot .message.user {
    background: var(--accent-cyan) !important; /* Cyan for user in dark mode */
    color: var(--text-on-cyan) !important; /* Black text on cyan */
}
html.dark .gradio-chatbot .message.bot {
    background: #383838 !important;
    color: var(--light-text-dm) !important;
}
.gradio-chatbot .message.user .text p, html.dark .gradio-chatbot .message.user .text p,
.gradio-chatbot .message.bot .text p, html.dark .gradio-chatbot .message.bot .text p {
    color: inherit !important;
}

/* Chat input container */
#chat-input-container {
    flex-shrink: 0 !important;
    padding-top: 8px !important;
    border-top: 1px solid #e0e0e0 !important;
    box-sizing: border-box;
}
html.dark #chat-input-container {
    border-top: 1px solid var(--dark-border-dm) !important;
}


/* Checkbox and Radio visibility in dark mode - uses CYAN accent */
html.dark input[type="checkbox"]:checked::before,
html.dark input[type="radio"]:checked::before {
    background-color: var(--accent-cyan) !important;
}


/* Toast messages (gr.Info, gr.Warning) */
.toast-wrap .toast-body.success {
    background-color: #d1e7dd !important; color: #0f5132 !important; border-color: #badbcc !important;
}
.toast-wrap .toast-body.warning {
    background-color: #fff3cd !important; color: #664d03 !important; border-color: #ffecb5 !important;
}
html.dark .toast-wrap .toast-body.success {
    background-color: #14472D !important; color: #A3D8BF !important; border-color: #1B6D42 !important;
}
html.dark .toast-wrap .toast-body.warning {
    background-color: #594402 !important; color: #FFDDA1 !important; border-color: #8C6D03 !important;
}
html.dark .toast-wrap .toast-body p {
    color: inherit !important;
}

/* Scrollbar styling for dark mode (optional, webkit only) */
html.dark ::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
html.dark ::-webkit-scrollbar-track {
    background: var(--dark-surface-dm);
}
html.dark ::-webkit-scrollbar-thumb {
    background-color: #555;
    border-radius: 4px;
    border: 2px solid var(--dark-surface-dm);
}
html.dark ::-webkit-scrollbar-thumb:hover {
    background-color: #777;
}
"""

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

def create_gradio_app():
    theme = gr.themes.Default(
        primary_hue=gr.themes.colors.cyan,
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    )

    with gr.Blocks(css=CUSTOM_CSS, theme=theme, title="IntelLaw Gradio") as demo:
        # Header
        with gr.Row(elem_id="app-header-row"):
            gr.Markdown("# 📄 IntelLaw", elem_id="app-title")
            with gr.Row(elem_id="button-column-container"):
                dummy = gr.Textbox(visible=False)
                settings_btn = gr.Button("Settings", elem_id="settings-btn", variant="secondary")
                sign_out_btn = gr.Button("Sign Out", elem_id="actual-sign-out-btn", variant="secondary")
        settings_btn.click(fn=None, inputs=None, outputs=[dummy], js=js_settings_function())
        sign_out_btn.click(fn=None, inputs=None, outputs=[dummy], js=js_logout_function())

        # Main content
        with gr.Row(elem_id="main-content-row"):
            # Controls Column
            with gr.Column(elem_classes=["controls-column"]):
                gr.Markdown("### 🛠️ Controls")
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
                        system_prompt = gr.Textbox(label="System Prompt", lines=4, value="You are a helpful assistant...")
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
            with gr.Column(elem_classes=["chat-column"]):
                gr.Markdown("### 💬 Chat Interface")
                # Initialize chatbot with an empty list to avoid None issues on first load if backend doesn't handle None
                chatbot = gr.Chatbot(value=[], label="IntelLaw Chatbot", show_copy_button=True, bubble_full_width=False)
                with gr.Row(elem_id="chat-input-container"):
                    chat_input = gr.Textbox(show_label=False, placeholder="Ask any question...", container=False)
                    send_btn = gr.Button("Send", variant="primary")

        # States and Load
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

        def update_ui_on_initial_load(current_app_config):
            db_msg, files_choices = get_initial_ui_data_for_gradio()
            selected_model_ids_from_state = current_app_config.get("selected_models", [])
            initial_selected_model_names = [
                MODEL_ID_TO_NAME_MAP[m_id] for m_id in selected_model_ids_from_state if m_id in MODEL_ID_TO_NAME_MAP
            ]
            if not initial_selected_model_names and AVAILABLE_MODELS_NAMES:
                first_available_model_name = AVAILABLE_MODELS_NAMES[0]
                initial_selected_model_names = [first_available_model_name]

            return (
                gr.update(value=db_msg),
                gr.update(choices=files_choices, value=[]),
                gr.update(value=initial_selected_model_names)
            )

        demo.load(
            fn=update_ui_on_initial_load,
            inputs=[app_config_state],
            outputs=[db_status, files_to_delete, model_selector]
        )

        # Event handlers
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

        def prepare_download(cfg):
            path, msg, ok = generate_config_for_download_backend(cfg)
            if not path: gr.Warning(msg)
            elif ok: gr.Info(msg)
            return path
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
