# gradio_ui.py

import gradio as gr
import os
import json
from dotenv import load_dotenv

# Ensure these imports point to your actual backend.py file and its functions
from backend import (
    get_unique_filenames_from_text_store,
    switch_db_backend,
    handle_pdf_upload_backend,
    chat_interface_backend,
    delete_files_backend,
    # Main config functions
    backend_get_initial_main_config,
    backend_update_main_config,
    backend_apply_uploaded_main_config,
    backend_generate_main_config_for_download,
    # Existing model info
    AVAILABLE_MODELS_NAMES,
    MODEL_NAME_TO_ID_MAP,
    MODEL_ID_TO_NAME_MAP,
    get_backend_initial_load_message,
    # Scraper config functions (for managing sources)
    get_scraper_ui_initial_data,
    backend_update_scraper_source_in_db,
    backend_remove_scraper_source_from_db,
)
load_dotenv() 
# FLASK_BASE_URL = os.getenv("FLASK_BASE_URL", "http://localhost:5000")
FLASK_BASE_URL = os.getenv("FLASK_BASE_URL")


def js_logout_function():
    flask_logout_url = f"{FLASK_BASE_URL}/logout"
    return f"() => {{ window.top.location.href = '{flask_logout_url}'; return []; }}"

def js_settings_function():
    flask_settings_url = f"{FLASK_BASE_URL}/settings"
    return f"() => {{ window.top.location.href = '{flask_settings_url}'; return []; }}"

def _show_status_popup(message, is_success):
    if message:
        if is_success: gr.Info(message)
        else: gr.Warning(message)

def get_initial_ui_data_for_gradio():
    initial_db_status_msg = get_backend_initial_load_message()
    initial_filenames_choices = get_unique_filenames_from_text_store()
    return initial_db_status_msg, initial_filenames_choices

CUSTOM_CSS = """
#app-header-row { position: sticky !important; top: 0; z-index: 100; width: 100% !important; box-sizing: border-box; }
#main-content-row { flex-grow: 1 !important; display: flex !important; flex-direction: row !important; padding: 20px !important; gap: 24px !important; overflow: hidden !important; min-height: 0 !important; width: 100% !important; box-sizing: border-box; }
.controls-column { flex: 0 0 650px !important; max-width: 700px !important; }
.chat-column { flex-grow: 1 !important; min-width: 300px !important; }
.scraper-section { border: 1px solid #e0e0e0; padding: 15px; border-radius: 8px; margin-bottom:15px; }
html.dark .scraper-section { border: 1px solid var(--dark-border-dm); }
.cron-explanation pre { white-space: pre-wrap; word-wrap: break-word; font-family: monospace; }
"""

CRON_EXPLANATION_MARKDOWN = """
A cron schedule expression has five fields that describe the schedule details.
`*` = always. It is a wildcard for every part of the cron schedule expression.
So `* * * * *` means **every minute** of **every hour** of **every day** of **every month** and **every day** of the **week**.

*The nice drawing above is provided by [wikipedia](https://en.wikipedia.org/wiki/Cron#CRON_expression).*
"""

def create_gradio_app():
    with gr.Blocks(css=CUSTOM_CSS, theme='lone17/kotaemon', title="IntelLaw Gradio") as demo:
        # --- States ---
        main_config_state = gr.State({})
        predefined_sources_state = gr.State([])
        current_db_scraper_config_state = gr.State([])
        add_selected_predefined_source_key_state = gr.State(None)
        edit_selected_db_source_key_state = gr.State(None)
        
        # --- UI Helper Functions ---
        def _get_source_details_by_key(source_key, predefined_list):
            return next((s for s in predefined_list if s['source_key'] == source_key), None)
        def _update_scraper_tab_choices(predefined_list, db_config_list):
            db_source_keys = {s.get('source_key') for s in db_config_list}
            add_choices = [s['display_name'] for s in predefined_list if s.get('source_key') not in db_source_keys]
            manage_choices = [s.get('display_name', 'Unnamed') for s in db_config_list]
            db_config_json_str = json.dumps(db_config_list, indent=2) if db_config_list else "[]"
            return gr.update(choices=add_choices, value=None), gr.update(choices=manage_choices, value=None), gr.update(choices=manage_choices, value=[]), db_config_json_str

        # Header
        with gr.Row(elem_id="app-header-row"):
            gr.Markdown("# 📄 IntelLaw", elem_id="app-title")
            with gr.Row():
                dummy = gr.Textbox(visible=False)
                settings_btn = gr.Button("Settings", elem_id="settings-btn", variant="secondary", visible=False)
                sign_out_btn = gr.Button("Sign Out", elem_id="actual-sign-out-btn", variant="secondary")
        settings_btn.click(fn=None, inputs=None, outputs=[dummy], js=js_settings_function())
        sign_out_btn.click(fn=None, inputs=None, outputs=[dummy], js=js_logout_function())

        # Main content
        with gr.Row(elem_id="main-content-row"):
            with gr.Column(elem_classes=["controls-column"], visible=False) as controls_column_element:
                gr.Markdown("### 🛠️ Controls")
                with gr.Accordion("Database Backend", open=True):
                    db_radio = gr.Radio(label="Choose Database", choices=["Dropbox", "MongoDB"], value="MongoDB")
                    db_status = gr.Textbox(label="DB Status", interactive=False)
                with gr.Tabs() as admin_tabs:
                    with gr.TabItem("⚙️ Main Config"):
                        gr.Markdown("### AI Model Configuration")
                        model_selector = gr.Dropdown(label="AI Models (Max 3)", choices=AVAILABLE_MODELS_NAMES, multiselect=True, max_choices=3, interactive=True)
                        vary_temp = gr.Checkbox(label="Vary Temperature", value=True)
                        temp_slider = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, step=0.01, value=0.7)
                        vary_top_p = gr.Checkbox(label="Vary Top-P", value=False)
                        top_p_slider = gr.Slider(label="Top-P", minimum=0.0, maximum=1.0, step=0.01, value=0.9)
                        system_prompt = gr.Textbox(label="System Prompt", lines=4, value="Answer questions strictly based on the provided context...")
                        gr.Markdown("---")
                        save_model_config_btn = gr.Button("Save Model & Prompt Settings", variant="primary")
                        main_config_feedback_text = gr.Textbox(label="Config Status", interactive=False, lines=2)
                        gr.Markdown("---")
                        gr.Markdown("#### Configuration File Management")
                        upload_main_config_btn = gr.UploadButton("Upload & Apply Main Config (JSON)", file_types=[".json"], variant="secondary", size="sm")
                        download_main_config_btn = gr.DownloadButton("Download Current Main Config", variant="secondary", size="sm")

                    with gr.TabItem("⏰ Scheduler"):
                        gr.Markdown("### Scraper Cron Job Schedule")
                        with gr.Row():
                            cron_minute = gr.Textbox(label="Minute", placeholder="0-59", interactive=True)
                            cron_hour = gr.Textbox(label="Hour", placeholder="0-23", interactive=True)
                            cron_day_month = gr.Textbox(label="Day (Month)", placeholder="1-31", interactive=True)
                            cron_month = gr.Textbox(label="Month", placeholder="1-12", interactive=True)
                            cron_day_week = gr.Textbox(label="Day (Week)", placeholder="0-7", interactive=True)
                        save_schedule_btn = gr.Button("Save Schedule", variant="primary")
                        schedule_feedback_text = gr.Textbox(label="Schedule Status", interactive=False, lines=2)
                        with gr.Accordion("Cron Syntax Help", open=False, elem_classes="cron-explanation"):
                           gr.Markdown(CRON_EXPLANATION_MARKDOWN)

                    with gr.TabItem("📁 Stored Files"):
                        file_uploader = gr.Files(label="Upload PDFs", file_count="multiple", type="filepath", file_types=[".pdf"])
                        upload_status = gr.Textbox(label="Upload Status", interactive=False, lines=3)
                        files_to_delete_cbg = gr.CheckboxGroup(label="Select Files to Delete", choices=[], value=[])
                        delete_btn = gr.Button("Delete Selected", variant="stop", size="sm")
                        delete_status = gr.Textbox(label="Deletion Status", interactive=False, lines=2)
                    
                    with gr.TabItem("🌐 Scraper Sources"):
                        gr.Markdown("### Web Scraper Sources Management")
                        scraper_sources_feedback_text = gr.Textbox(label="Sources Status", interactive=False, lines=2, placeholder="Status messages will appear here...")
                        with gr.Blocks(elem_classes="scraper-section"):
                            gr.Markdown("#### 1. Add New Source to DB Configuration")
                            add_source_dropdown = gr.Dropdown(label="Select Predefined Source to Add", choices=[], interactive=True)
                            with gr.Group(visible=False) as add_source_details_group:
                                add_source_display_name_info = gr.Textbox(label="Source Name", interactive=False)
                                add_source_base_url_info = gr.Textbox(label="Base URL", interactive=False)
                            add_source_max_pages_input = gr.Number(label="Set Max Pages for This Source", value=1, minimum=1, precision=0, interactive=True, visible=False)
                            add_source_to_db_btn = gr.Button("Add to DB Config", variant="primary", visible=False)
                        with gr.Blocks(elem_classes="scraper-section"):
                            gr.Markdown("#### 2. Edit Max Pages for Configured Source")
                            edit_source_dropdown = gr.Dropdown(label="Select Configured Source to Edit", choices=[], interactive=True)
                            with gr.Group(visible=False) as edit_source_details_group:
                                edit_source_params_display = gr.Code(label="Current Parameters (Informational)", language="json", interactive=False)
                                edit_source_max_pages_input = gr.Number(label="New Max Pages", value=1, minimum=1, precision=0, interactive=True)
                            update_max_pages_btn = gr.Button("Update Max Pages in DB", variant="secondary", visible=False)
                        with gr.Blocks(elem_classes="scraper-section"):
                            gr.Markdown("#### 3. Remove Sources from DB Configuration")
                            remove_source_checkbox_group = gr.CheckboxGroup(label="Select Configured Sources to Remove", choices=[], interactive=True)
                            remove_selected_sources_btn = gr.Button("Remove Selected from DB Config", variant="stop")
                        gr.Markdown("---")
                        gr.Markdown("### Current Scraper Sources in Database (JSON View)")
                        current_db_scraper_config_display = gr.Code(label="DB Scraper Config", language="json", interactive=False)

            with gr.Column(elem_classes=["chat-column"]) as chat_column_element:
                gr.Markdown("### 💬 Chat Interface")
                chatbot = gr.Chatbot(value=[], label="IntelLaw Chatbot", show_copy_button=True, bubble_full_width=False, height=500)
                with gr.Row(elem_id="chat-input-container"):
                    chat_input = gr.Textbox(show_label=False, placeholder="Ask any question...", container=False)
                    send_btn = gr.Button("Send", variant="primary")

        # States and Load logic
        selected_db_state = gr.State("MongoDB")
        status_msg = gr.State(); success_flag = gr.State()

        def handle_visibility_and_initial_load(request: gr.Request):
            user_role = request.query_params.get("user_role", "user") if request and hasattr(request, "query_params") else "user"
            admin_view_visible = (user_role == "admin")
            db_msg, files_choices_init = get_initial_ui_data_for_gradio()
            main_config_init = backend_get_initial_main_config()
            model_conf = main_config_init.get('model_config', {})
            sel_model_ids = model_conf.get("selected_models", [])
            initial_selected_model_names = [MODEL_ID_TO_NAME_MAP.get(mid) for mid in sel_model_ids if MODEL_ID_TO_NAME_MAP.get(mid)]
            cron_expr = main_config_init.get('cron_schedule', '0 2 * * *')
            cron_parts = cron_expr.split()
            c_min, c_hour, c_dom, c_mon, c_dow = cron_parts if len(cron_parts) == 5 else ['0', '2', '*', '*', '*']
            predefined_sources_init, db_scraper_config_init = get_scraper_ui_initial_data()
            add_dd, edit_dd, remove_cbg, db_json_str = _update_scraper_tab_choices(predefined_sources_init, db_scraper_config_init)
            return (
                gr.update(visible=admin_view_visible), gr.update(visible=admin_view_visible),
                db_msg, gr.update(choices=files_choices_init, value=[]), main_config_init,
                gr.update(value=initial_selected_model_names),
                model_conf.get("vary_temperature", True), model_conf.get("temperature", 0.7),
                model_conf.get("vary_top_p", False), model_conf.get("top_p", 0.9),
                model_conf.get("system_prompt", ""), c_min, c_hour, c_dom, c_mon, c_dow,
                predefined_sources_init, db_scraper_config_init, add_dd, edit_dd, remove_cbg, db_json_str,
            )
        demo.load(
            fn=handle_visibility_and_initial_load, inputs=None, 
            outputs=[
                settings_btn, controls_column_element, db_status, files_to_delete_cbg, main_config_state,
                model_selector, vary_temp, temp_slider, vary_top_p, top_p_slider, system_prompt,
                cron_minute, cron_hour, cron_day_month, cron_month, cron_day_week,
                predefined_sources_state, current_db_scraper_config_state,
                add_source_dropdown, edit_source_dropdown, remove_source_checkbox_group, current_db_scraper_config_display
            ]
        )

        # Event Handlers
        db_radio.change(fn=switch_db_backend, inputs=[db_radio, selected_db_state], outputs=[selected_db_state, db_status, files_to_delete_cbg])
        
        def handle_save_model_config(current_main_config, sel_models, vary_t, temp, vary_p, top_p, sys_prompt):
            updated_config = current_main_config.copy()
            sel_model_ids = [MODEL_NAME_TO_ID_MAP.get(name) for name in sel_models if MODEL_NAME_TO_ID_MAP.get(name)]
            if 'model_config' not in updated_config: updated_config['model_config'] = {}
            updated_config['model_config'].update({"selected_models": sel_model_ids, "vary_temperature": vary_t, "temperature": temp, "vary_top_p": vary_p, "top_p": top_p, "system_prompt": sys_prompt})
            success, msg, final_config = backend_update_main_config(updated_config)
            return msg, final_config if success else current_main_config
        save_model_config_btn.click(
            fn=handle_save_model_config,
            inputs=[main_config_state, model_selector, vary_temp, temp_slider, vary_top_p, top_p_slider, system_prompt],
            outputs=[main_config_feedback_text, main_config_state]
        )
        
        def handle_save_schedule(current_main_config, c_min, c_hour, c_dom, c_mon, c_dow):
            updated_config = current_main_config.copy()
            updated_config['cron_schedule'] = f"{c_min} {c_hour} {c_dom} {c_mon} {c_dow}"
            success, msg, final_config = backend_update_main_config(updated_config)
            return msg, final_config if success else current_main_config
        save_schedule_btn.click(
            fn=handle_save_schedule,
            inputs=[main_config_state, cron_minute, cron_hour, cron_day_month, cron_month, cron_day_week],
            outputs=[schedule_feedback_text, main_config_state]
        )
        
        def update_ui_from_main_config(new_main_config):
            model_conf = new_main_config.get('model_config', {})
            sel_model_ids = model_conf.get("selected_models", [])
            sel_model_names = [MODEL_ID_TO_NAME_MAP.get(mid) for mid in sel_model_ids if MODEL_ID_TO_NAME_MAP.get(mid)]
            cron_expr = new_main_config.get('cron_schedule', '0 2 * * *')
            cron_parts = cron_expr.split()
            c_min, c_hour, c_dom, c_mon, c_dow = cron_parts if len(cron_parts) == 5 else ['0', '2', '*', '*', '*']
            return (gr.update(value=sel_model_names), model_conf.get("vary_temperature", True), model_conf.get("temperature", 0.7), model_conf.get("vary_top_p", False), model_conf.get("top_p", 0.9), model_conf.get("system_prompt", ""), c_min, c_hour, c_dom, c_mon, c_dow)
        
        upload_main_config_btn.upload(fn=backend_apply_uploaded_main_config, inputs=[upload_main_config_btn], outputs=[status_msg, success_flag, main_config_state]
            ).then(fn=_show_status_popup, inputs=[status_msg, success_flag], outputs=None
            ).then(fn=update_ui_from_main_config, inputs=[main_config_state], outputs=[
                model_selector, vary_temp, temp_slider, vary_top_p, top_p_slider, system_prompt,
                cron_minute, cron_hour, cron_day_month, cron_month, cron_day_week
            ])
            
        download_main_config_btn.click(fn=backend_generate_main_config_for_download, inputs=[main_config_state], outputs=[download_main_config_btn])
        file_uploader.upload(fn=handle_pdf_upload_backend, inputs=[file_uploader, selected_db_state], outputs=[upload_status, files_to_delete_cbg])
        delete_btn.click(fn=delete_files_backend, inputs=[files_to_delete_cbg, selected_db_state], outputs=[delete_status, files_to_delete_cbg])
        
        # --- CORRECTED Event Handlers for Web Scraper Sources Tab ---
        def on_add_source_dropdown_change(selected_display_name, predefined_list):
            if not selected_display_name: return gr.update(visible=False), None, "", "", 1, gr.update(visible=False), gr.update(visible=False)
            source_details = next((s for s in predefined_list if s['display_name'] == selected_display_name), None)
            if not source_details: return gr.update(visible=False), None, "", "", 1, gr.update(visible=False), gr.update(visible=False)
            return (gr.update(visible=True), source_details['source_key'], source_details['display_name'], source_details['parameters'].get('base_url_root', ''), gr.update(visible=True), gr.update(visible=True))
        add_source_dropdown.change(
            fn=on_add_source_dropdown_change,
            inputs=[add_source_dropdown, predefined_sources_state],
            outputs=[add_source_details_group, add_selected_predefined_source_key_state, add_source_display_name_info, add_source_base_url_info, add_source_max_pages_input, add_source_to_db_btn]
        )
        def _get_db_config_details_by_key(source_key, db_config_list):
            return next((s for s in db_config_list if s['source_key'] == source_key), None)
        def on_add_source_to_db_click(s_key, max_p, p_list, db_list):
            if not s_key:
                add, edit, rem, json_str = _update_scraper_tab_choices(p_list, db_list)
                return "No source selected.", db_list, add, edit, rem, json_str, gr.update(visible=False), gr.update(visible=False)
            predefined = _get_source_details_by_key(s_key, p_list)
            if not predefined:
                add, edit, rem, json_str = _update_scraper_tab_choices(p_list, db_list)
                return f"Error: Details for key '{s_key}' not found.", db_list, add, edit, rem, json_str, gr.update(visible=False), gr.update(visible=False)
            new_params = {**predefined['parameters'], 'max_pages': int(max_p)}
            new_db_config, status_msg_backend = backend_update_scraper_source_in_db(s_key, predefined['display_name'], new_params)
            add, edit, rem, json_str = _update_scraper_tab_choices(p_list, new_db_config)
            return status_msg_backend, new_db_config, add, edit, rem, json_str, gr.update(visible=False), gr.update(visible=False)
        add_source_to_db_btn.click(
            fn=on_add_source_to_db_click,
            inputs=[add_selected_predefined_source_key_state, add_source_max_pages_input, predefined_sources_state, current_db_scraper_config_state],
            outputs=[scraper_sources_feedback_text, current_db_scraper_config_state, add_source_dropdown, edit_source_dropdown, remove_source_checkbox_group, current_db_scraper_config_display, add_source_details_group, add_source_to_db_btn]
        )

        def on_edit_source_dropdown_change(selected_name, db_list, p_list):
            if not selected_name: return gr.update(visible=False), None, "{}", 1, gr.update(visible=False)
            db_source = next((s for s in db_list if s.get('display_name') == selected_name), None)
            if not db_source: return gr.update(visible=False), None, "{}", 1, gr.update(visible=False)
            source_key = db_source.get('source_key')
            predefined_params = _get_source_details_by_key(source_key, p_list).get('parameters', {})
            current_params = {**predefined_params, **db_source.get('parameters', {})}
            return gr.update(visible=True), source_key, json.dumps(current_params, indent=2), current_params.get('max_pages', 1), gr.update(visible=True)
        edit_source_dropdown.change(
            fn=on_edit_source_dropdown_change,
            inputs=[edit_source_dropdown, current_db_scraper_config_state, predefined_sources_state],
            outputs=[edit_source_details_group, edit_selected_db_source_key_state, edit_source_params_display, edit_source_max_pages_input, update_max_pages_btn]
        )
        
        def on_update_max_pages_click(s_key, new_max_p, p_list, db_list):
            if not s_key:
                add, edit, rem, json_str = _update_scraper_tab_choices(p_list, db_list)
                return "No source selected to update.", db_list, add, edit, rem, json_str, gr.update(visible=False), gr.update(visible=False)
            db_source = _get_db_config_details_by_key(s_key, db_list)
            predefined = _get_source_details_by_key(s_key, p_list)
            if not db_source or not predefined:
                add, edit, rem, json_str = _update_scraper_tab_choices(p_list, db_list)
                return f"Error: Details for key '{s_key}' not found.", db_list, add, edit, rem, json_str, gr.update(visible=False), gr.update(visible=False)
            updated_params = {**predefined['parameters'], **db_source['parameters'], 'max_pages': int(new_max_p)}
            new_db_config, status_msg_backend = backend_update_scraper_source_in_db(s_key, db_source['display_name'], updated_params)
            add, edit, rem, json_str = _update_scraper_tab_choices(p_list, new_db_config)
            return status_msg_backend, new_db_config, add, edit, rem, json_str, gr.update(visible=False), gr.update(visible=False)
        update_max_pages_btn.click(
            fn=on_update_max_pages_click,
            inputs=[edit_selected_db_source_key_state, edit_source_max_pages_input, predefined_sources_state, current_db_scraper_config_state],
            outputs=[scraper_sources_feedback_text, current_db_scraper_config_state, add_source_dropdown, edit_source_dropdown, remove_source_checkbox_group, current_db_scraper_config_display, edit_source_details_group, update_max_pages_btn]
        )
        
        def on_remove_selected_sources_click(names_to_remove, p_list, db_list):
            if not names_to_remove:
                add, edit, rem, json_str = _update_scraper_tab_choices(p_list, db_list)
                return "No sources selected for removal.", db_list, add, edit, rem, json_str
            keys_to_remove = [s['source_key'] for s in db_list if s.get('display_name') in names_to_remove]
            new_db_config = db_list
            msgs = []
            for key in keys_to_remove:
                new_db_config, msg = backend_remove_scraper_source_from_db(key)
                msgs.append(msg)
            add, edit, rem, json_str = _update_scraper_tab_choices(p_list, new_db_config)
            return f"Removal process finished: {'. '.join(msgs)}", new_db_config, add, edit, rem, json_str
        remove_selected_sources_btn.click(
            fn=on_remove_selected_sources_click,
            inputs=[remove_source_checkbox_group, predefined_sources_state, current_db_scraper_config_state],
            outputs=[scraper_sources_feedback_text, current_db_scraper_config_state, add_source_dropdown, edit_source_dropdown, remove_source_checkbox_group, current_db_scraper_config_display]
        )
        
        def run_chat(user_input, history, db_state_val, main_config_val):
            model_config = main_config_val.get('model_config', {})
            return chat_interface_backend(user_input, history, db_state_val, model_config)
        chat_inputs = [chat_input, chatbot, selected_db_state, main_config_state]
        send_btn.click(fn=run_chat, inputs=chat_inputs, outputs=[chatbot]).then(fn=lambda: gr.update(value=""), inputs=None, outputs=[chat_input])
        chat_input.submit(fn=run_chat, inputs=chat_inputs, outputs=[chatbot]).then(fn=lambda: gr.update(value=""), inputs=None, outputs=[chat_input])
        
        return demo