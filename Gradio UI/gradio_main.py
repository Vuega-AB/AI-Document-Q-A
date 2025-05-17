import gradio as gr

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

# Custom CSS (remains the same)
CUSTOM_CSS = """
/* Global background */
body, .gradio-container { background-color: #f0f4f8 !important; }
.gradio-container { font-family: 'Inter', sans-serif; }
.gradio-chatbot .message.bot { background-color: #2563eb !important; color: #ffffff !important; border-radius: 1rem 0.25rem 1rem 1rem !important; padding: 0.75rem 1rem !important; box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important; max-width: 80% !important; align-self: flex-start !important; margin-left: 0.5rem; margin-right: auto; }
.gradio-chatbot .message.user { background-color: #e0f2fe !important; color: #0c4a6e !important; border-radius: 0.25rem 1rem 1rem 1rem !important; padding: 0.75rem 1rem !important; box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important; max-width: 80% !important; align-self: flex-end !important; margin-right: 0.5rem; margin-left: auto; }
.gradio-chatbot .messages { padding: 1rem !important; display: flex; flex-direction: column; gap: 0.75rem; }
#chat-input-container textarea { border-radius: 0.75rem !important; border: 1px solid #cbd5e1 !important; padding: 0.75rem !important; }
#chat-input-container button { height: 100%; }
.gr-slider .track, .gr-slider .thumb { height: 1rem !important; }
.gr-slider .thumb { width: 1rem !important; }
@media (max-width: 768px) { .gr-row { flex-direction: column !important; } .controls-column { min-width: 100% !important; } .gradio-chatbot .message.bot, .gradio-chatbot .message.user { max-width: 90% !important; } }
.gr-button { border-radius: 0.75rem !important; font-weight: 500 !important; padding: 0.6rem 1.2rem !important; }
.gr-button.gr-button-primary { background-color: #1d4ed8 !important; color: white !important; }
.gr-button.stop { background-color: #dc2626 !important; color: white !important; }
"""

# Helper function to show popups
def _show_status_popup(message, is_success):
    if message:
        if is_success:
            gr.Info(message)
        else:
            gr.Warning(message)

with gr.Blocks(
    css=CUSTOM_CSS,
    theme=gr.themes.Soft(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.sky,
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    ),
    title="IntelLaw Gradio"
) as demo:
    selected_db_state = gr.State(value="Dropbox")
    initial_selected_model_id = []
    if AVAILABLE_MODELS_NAMES and MODEL_NAME_TO_ID_MAP:
        first_model_name = AVAILABLE_MODELS_NAMES[0]
        if first_model_name in MODEL_NAME_TO_ID_MAP:
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

    gr.Markdown("# 📄 IntelLaw - Chat with Documents")

    with gr.Row(equal_height=False):
        with gr.Column(scale=2, min_width=420, elem_classes=["controls-column"]):
            gr.Markdown("### 🛠️ Controls")
            with gr.Accordion("Database Backend", open=True):
                db_radio = gr.Radio(label="Choose Database", choices=["Dropbox", "MongoDB"], value=selected_db_state.value)
                db_status = gr.Textbox(label="DB Status", interactive=False, value=BACKEND_INITIAL_LOAD_MSG)

            with gr.Tabs():
                with gr.TabItem("🧠 Config"):
                    model_selector = gr.Dropdown(
                        label="AI Models (Max 3)", choices=AVAILABLE_MODELS_NAMES,
                        value=[AVAILABLE_MODELS_NAMES[0]] if AVAILABLE_MODELS_NAMES and AVAILABLE_MODELS_NAMES[0] in MODEL_NAME_TO_ID_MAP else [],
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

        with gr.Column(scale=3):
            gr.Markdown("### 💬 Chat Interface")
            chatbot = gr.Chatbot(label="IntelLaw Chatbot", height=600, show_copy_button=True, bubble_full_width=False)
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
        if filepath is None:
            gr.Warning(status_msg or "Failed to generate file for download.")
        elif success: # Optionally show success info even in simplified mode
            gr.Info(status_msg or "File ready for download.")
        return filepath

    download_config_btn.click(
        fn=_generate_and_get_path_only,
        inputs=[app_config_state],
        outputs=[download_config_btn] # Only output to the button itself
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
    # MODIFIED: chat_outputs no longer includes responses_html
    chat_outputs = [chatbot] 
    
    def clear_input_fn(): return gr.update(value="")

    send_btn.click(fn=chat_interface_backend, inputs=chat_inputs, outputs=chat_outputs).then(fn=clear_input_fn, inputs=None, outputs=chat_input)
    chat_input.submit(fn=chat_interface_backend, inputs=chat_inputs, outputs=chat_outputs).then(fn=clear_input_fn, inputs=None, outputs=chat_input)

    scrape_btn.click(
        fn=run_scraper_backend,
        inputs=[base_url_input, listing_endpoint_input, pagination_input, pages_to_check_input, selected_db_state],
        outputs=[scraper_output, files_to_delete_checkboxgroup, stored_files_md]
    )

if __name__ == '__main__':
    if BACKEND_INITIAL_LOAD_MSG == "Backend not initialized.":
        print("Backend may not have initialized fully.")
    demo.launch()