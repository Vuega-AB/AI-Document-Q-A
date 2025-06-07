# gradio_ui.py

import gradio as gr
import os
import json

# Ensure these imports point to your actual backend.py file and its functions
from backend import (
    get_unique_filenames_from_text_store,
    switch_db_backend,
    handle_pdf_upload_backend,
    chat_interface_backend,
    delete_files_backend,
    apply_uploaded_config_backend,
    generate_config_for_download_backend,
    AVAILABLE_MODELS_NAMES,
    MODEL_NAME_TO_ID_MAP,
    MODEL_ID_TO_NAME_MAP,
    get_backend_initial_load_message,
    get_scraper_ui_initial_data,
    backend_update_scraper_source_in_db,
    backend_remove_scraper_source_from_db,
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
#app-header-row {
    position: sticky !important; top: 0; z-index: 100; width: 100% !important; box-sizing: border-box;
}
#main-content-row {
    flex-grow: 1 !important; display: flex !important; flex-direction: row !important;
    padding: 20px !important; gap: 24px !important; overflow: hidden !important;
    min-height: 0 !important; width: 100% !important; box-sizing: border-box;
}
.controls-column { flex: 0 0 450px !important; max-width: 500px !important; }
.chat-column { flex-grow: 1 !important; min-width: 300px !important; }
.scraper-section { border: 1px solid #e0e0e0; padding: 15px; border-radius: 8px; margin-bottom:15px;}
html.dark .scraper-section { border: 1px solid var(--dark-border-dm); }
"""

def create_gradio_app():
    with gr.Blocks(css=CUSTOM_CSS, theme='lone17/kotaemon', title="IntelLaw Gradio") as demo:
        # --- States needed across UI ---
        predefined_sources_state = gr.State([])  # Full list of predefined source dicts
        current_db_scraper_config_state = gr.State([])  # List of source dicts from DB
        
        # States for scraper tab to hold selected source_key for actions
        add_selected_predefined_source_key_state = gr.State(None)
        edit_selected_db_source_key_state = gr.State(None)

        # --- Helper Python functions for UI updates ---
        def _get_source_details_by_key(source_key, predefined_list):
            return next((s for s in predefined_list if s['source_key'] == source_key), None)

        def _get_db_config_details_by_key(source_key, db_config_list):
            return next((s for s in db_config_list if s['source_key'] == source_key), None)

        def _update_scraper_tab_choices(predefined_list, db_config_list):
            db_source_keys = {s['source_key'] for s in db_config_list}
            
            # For adding: predefined sources NOT YET in DB
            add_choices = [s['display_name'] for s in predefined_list if s['source_key'] not in db_source_keys]
            
            # For editing/removing: sources currently IN DB
            manage_choices = [s['display_name'] for s in db_config_list] # Use display_name for UI
            
            db_config_json_str = json.dumps(db_config_list, indent=2) if db_config_list else "[]"
            
            return (
                gr.update(choices=add_choices, value=None),       # add_source_dropdown
                gr.update(choices=manage_choices, value=None),    # edit_source_dropdown
                gr.update(choices=manage_choices, value=[]),     # remove_source_checkbox_group
                db_config_json_str                               # current_db_config_display
            )

        # Header
        with gr.Row(elem_id="app-header-row"):
            gr.Markdown("# 📄 IntelLaw", elem_id="app-title")
            with gr.Row(elem_id="button-column-container"):
                dummy = gr.Textbox(visible=False)
                settings_btn = gr.Button("Settings", elem_id="settings-btn", variant="secondary", visible=False)
                sign_out_btn = gr.Button("Sign Out", elem_id="actual-sign-out-btn", variant="secondary")
        settings_btn.click(fn=None, inputs=None, outputs=[dummy], js=js_settings_function())
        sign_out_btn.click(fn=None, inputs=None, outputs=[dummy], js=js_logout_function())

        # Main content
        with gr.Row(elem_id="main-content-row"):
            # Controls Column (Admin Only)
            with gr.Column(elem_classes=["controls-column"], visible=False) as controls_column_element:
                gr.Markdown("### 🛠️ Controls")
                with gr.Accordion("Database Backend", open=True):
                    db_radio = gr.Radio(label="Choose Database", choices=["Dropbox", "MongoDB"], value="MongoDB")
                    db_status = gr.Textbox(label="DB Status", interactive=False)
                with gr.Tabs() as admin_tabs:
                    with gr.TabItem("🧠 Model Config"):
                        # ... (Model Config UI remains the same as your last provided version) ...
                        model_selector = gr.Dropdown(label="AI Models (Max 3)", choices=AVAILABLE_MODELS_NAMES, multiselect=True, max_choices=3)
                        vary_temp = gr.Checkbox(label="Vary Temperature", value=True)
                        temp_slider = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, step=0.01, value=0.7)
                        vary_top_p = gr.Checkbox(label="Vary Top-P", value=False)
                        top_p_slider = gr.Slider(label="Top-P", minimum=0.0, maximum=1.0, step=0.01, value=0.9)
                        system_prompt = gr.Textbox(label="System Prompt", lines=4, value="Answer questions strictly based on the provided context. If there is no context, say 'I don't have enough information to answer that.'")
                        gr.Markdown("---"); gr.Markdown("#### Configuration File Management")
                        upload_config_btn = gr.UploadButton("Upload & Apply Model Config (JSON)", file_types=[".json"], variant="secondary", size="sm")
                        download_config_btn = gr.DownloadButton("Download Current Model Config", variant="secondary", size="sm")

                    with gr.TabItem("📁 Stored Files"):
                        # ... (Stored Files UI remains the same as your last provided version) ...
                        file_uploader = gr.Files(label="Upload PDFs", file_count="multiple", type="filepath", file_types=[".pdf"])
                        upload_status = gr.Textbox(label="Upload Status", interactive=False, lines=3)
                        files_to_delete_cbg = gr.CheckboxGroup(label="Select Files to Delete", choices=[], value=[]) # Renamed for clarity
                        delete_btn = gr.Button("Delete Selected", variant="stop", size="sm")
                        delete_status = gr.Textbox(label="Deletion Status", interactive=False, lines=2)
                    
                    with gr.TabItem("🌐 Web Scraper Config") as web_scraper_tab:
                        gr.Markdown("### Web Scraper Configuration Management")
                        scraper_config_feedback_text = gr.Textbox(label="Config Status", interactive=False, lines=2, placeholder="Status messages will appear here...")
                        
                        with gr.Blocks(elem_classes="scraper-section"):
                            gr.Markdown("#### 1. Add New Source to DB Configuration")
                            add_source_dropdown = gr.Dropdown(label="Select Predefined Source to Add", choices=[], interactive=True)
                            with gr.Group(visible=False) as add_source_details_group:
                                add_source_display_name_info = gr.Textbox(label="Source Name", interactive=False)
                                add_source_base_url_info = gr.Textbox(label="Base URL", interactive=False)
                                add_source_listing_segment_info = gr.Textbox(label="Listing Segment", interactive=False)
                                add_source_pagination_info = gr.Textbox(label="Pagination Format", interactive=False)
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
                        gr.Markdown("### Current Scraper Configuration in Database (JSON View)")
                        current_db_config_display = gr.Code(label="DB Config", language="json", interactive=False)

            # Chat Column (Visible to all)
            with gr.Column(elem_classes=["chat-column"]) as chat_column_element:
                # ... (Chat UI remains the same) ...
                gr.Markdown("### 💬 Chat Interface")
                chatbot = gr.Chatbot(value=[], label="IntelLaw Chatbot", show_copy_button=True, bubble_full_width=False)
                with gr.Row(elem_id="chat-input-container"):
                    chat_input = gr.Textbox(show_label=False, placeholder="Ask any question...", container=False)
                    send_btn = gr.Button("Send", variant="primary")

        # States and Load logic
        selected_db_state = gr.State("MongoDB")
        initial_model_id = MODEL_NAME_TO_ID_MAP.get(AVAILABLE_MODELS_NAMES[0]) if AVAILABLE_MODELS_NAMES and AVAILABLE_MODELS_NAMES[0] in MODEL_NAME_TO_ID_MAP else None
        selected_models_init = [initial_model_id] if initial_model_id else []
        app_config_state = gr.State({ "selected_models": selected_models_init, "vary_temperature": True, "temperature": 0.7, "vary_top_p": False, "top_p": 0.9, "system_prompt": "Answer questions strictly based on the provided context..."})
        status_msg = gr.State(); success_flag = gr.State() 

        # --- Main Load Function ---
        def handle_visibility_and_initial_load(request: gr.Request, current_model_app_config_val):
            user_role = request.query_params.get("user_role", "user") if request and hasattr(request, "query_params") else "user"
            admin_view_visible = (user_role == "admin")

            db_msg, files_choices_init = get_initial_ui_data_for_gradio()
            
            selected_model_ids_from_state = current_model_app_config_val.get("selected_models", [])
            initial_selected_model_names = [MODEL_ID_TO_NAME_MAP.get(m_id) for m_id in selected_model_ids_from_state if MODEL_ID_TO_NAME_MAP.get(m_id)]
            if not initial_selected_model_names and AVAILABLE_MODELS_NAMES:
                initial_selected_model_names = [AVAILABLE_MODELS_NAMES[0]]
            
            predefined_sources_init_val, db_scraper_config_init_val = get_scraper_ui_initial_data()
            add_dd, edit_dd, remove_cbg, db_json_str = _update_scraper_tab_choices(predefined_sources_init_val, db_scraper_config_init_val)

            return (
                gr.update(visible=admin_view_visible), gr.update(visible=admin_view_visible), db_msg,
                gr.update(choices=files_choices_init, value=[]), gr.update(value=initial_selected_model_names),
                predefined_sources_init_val, db_scraper_config_init_val,
                add_dd, edit_dd, remove_cbg, db_json_str
            )
        demo.load(
            fn=handle_visibility_and_initial_load, inputs=[app_config_state], 
            outputs=[
                settings_btn, controls_column_element, db_status, files_to_delete_cbg, model_selector,
                predefined_sources_state, current_db_scraper_config_state,
                add_source_dropdown, edit_source_dropdown, remove_source_checkbox_group, current_db_config_display
            ]
        )

        # --- Event Handlers for Tabs other than Scraper ---
        db_radio.change(fn=switch_db_backend, inputs=[db_radio, selected_db_state], outputs=[selected_db_state, db_status, files_to_delete_cbg]) # Pass correct output
        # ... (Model Config event handlers as before) ...
        def handle_model_config_change_event(models_names, vary_t, temp, vary_p, top_p, sys_prompt, current_config_state_val): # Renamed to avoid conflict
            # ... (logic for model config change)
            selected_model_ids = [MODEL_NAME_TO_ID_MAP[name] for name in models_names if name in MODEL_NAME_TO_ID_MAP]
            if len(selected_model_ids) > 3: selected_model_ids = selected_model_ids[:3] # Max 3
            updated_config = current_config_state_val.copy()
            updated_config.update({"selected_models": selected_model_ids, "vary_temperature": vary_t, "temperature": temp, "vary_top_p": vary_p, "top_p": top_p, "system_prompt": sys_prompt})
            return updated_config
        model_config_inputs_list = [model_selector, vary_temp, temp_slider, vary_top_p, top_p_slider, system_prompt] # Renamed
        for inp_widget in model_config_inputs_list: # Renamed
            inp_widget.change(fn=handle_model_config_change_event, inputs=model_config_inputs_list + [app_config_state], outputs=[app_config_state])
        upload_config_btn.upload(fn=apply_uploaded_config_backend, inputs=[upload_config_btn, app_config_state], outputs=[status_msg, success_flag, app_config_state, model_selector, vary_temp, temp_slider, vary_top_p, top_p_slider, system_prompt]).then(fn=_show_status_popup, inputs=[status_msg, success_flag], outputs=None)
        def prepare_model_config_download_event(current_config_state_val): # Renamed
            # ... (logic for model config download)
            path, msg, ok = generate_config_for_download_backend(current_config_state_val)
            if not path and not ok: gr.Warning(msg) 
            elif ok and path: gr.Info(msg) 
            return path if (path and ok) else None  
        download_config_btn.click(fn=prepare_model_config_download_event, inputs=[app_config_state], outputs=[download_config_btn])
        file_uploader.upload(fn=handle_pdf_upload_backend, inputs=[file_uploader, selected_db_state], outputs=[upload_status, files_to_delete_cbg])
        delete_btn.click(fn=delete_files_backend, inputs=[files_to_delete_cbg, selected_db_state], outputs=[delete_status, files_to_delete_cbg])

        # --- Event Handlers for Web Scraper Config Tab ---
        
        # When a predefined source is selected from the "Add" dropdown
        def on_add_source_dropdown_change(selected_display_name, predefined_list):
            if not selected_display_name:
                return gr.update(visible=False), None, "", "", "", "", 1, gr.update(visible=False), gr.update(visible=False) # Hide group, max_pages, button
            
            source_details = next((s for s in predefined_list if s['display_name'] == selected_display_name), None)
            if not source_details:
                return gr.update(visible=False), None, "", "", "", "", 1, gr.update(visible=False), gr.update(visible=False)

            return (
                gr.update(visible=True), # add_source_details_group
                source_details['source_key'], # add_selected_predefined_source_key_state
                source_details['display_name'],
                source_details['parameters'].get('base_url_root', 'N/A'),
                source_details['parameters'].get('listing_path_segment', 'N/A'),
                source_details['parameters'].get('pagination_query_format', 'N/A'),
                source_details['parameters'].get('max_pages', 1), # Default max_pages
                gr.update(visible=True), # add_source_max_pages_input
                gr.update(visible=True)  # add_source_to_db_btn
            )
        add_source_dropdown.change(
            fn=on_add_source_dropdown_change,
            inputs=[add_source_dropdown, predefined_sources_state],
            outputs=[
                add_source_details_group, add_selected_predefined_source_key_state,
                add_source_display_name_info, add_source_base_url_info, add_source_listing_segment_info, add_source_pagination_info,
                add_source_max_pages_input, add_source_max_pages_input, add_source_to_db_btn # Max pages input appears twice intentionally for the return tuple
            ]
        )

        # Add button click
        def on_add_source_to_db_click(s_key_to_add, max_p_val, predefined_list_val, current_db_config_val):
            if not s_key_to_add:
                return "No source selected to add.", current_db_config_val, *_update_scraper_tab_choices(predefined_list_val, current_db_config_val), gr.update(visible=False), gr.update(visible=False) # Hide group and button
            
            predefined_detail = _get_source_details_by_key(s_key_to_add, predefined_list_val)
            if not predefined_detail:
                return f"Error: Details for source key '{s_key_to_add}' not found.", current_db_config_val, *_update_scraper_tab_choices(predefined_list_val, current_db_config_val), gr.update(visible=False), gr.update(visible=False)

            new_params = predefined_detail['parameters'].copy()
            new_params['max_pages'] = int(max_p_val)
            
            new_db_config, status_msg_backend = backend_update_scraper_source_in_db(s_key_to_add, predefined_detail['display_name'], new_params)
            
            add_dd_upd, edit_dd_upd, rem_cbg_upd, db_json_upd = _update_scraper_tab_choices(predefined_list_val, new_db_config)
            return status_msg_backend, new_db_config, add_dd_upd, edit_dd_upd, rem_cbg_upd, db_json_upd, gr.update(visible=False), gr.update(visible=False) # Hide add group
        add_source_to_db_btn.click(
            fn=on_add_source_to_db_click,
            inputs=[add_selected_predefined_source_key_state, add_source_max_pages_input, predefined_sources_state, current_db_scraper_config_state],
            outputs=[
                scraper_config_feedback_text, current_db_scraper_config_state,
                add_source_dropdown, edit_source_dropdown, remove_source_checkbox_group, current_db_config_display,
                add_source_details_group, add_source_to_db_btn # Hide these after adding
            ]
        )
        
        # When a configured source is selected from the "Edit" dropdown
        def on_edit_source_dropdown_change(selected_display_name_edit, db_config_list_val, predefined_list_val):
            if not selected_display_name_edit:
                return gr.update(visible=False), None, "{}", 1, gr.update(visible=False) # Hide group and button
            
            # Find by display name first, then get source_key
            selected_db_source = next((s for s in db_config_list_val if s['display_name'] == selected_display_name_edit), None)
            if not selected_db_source:
                 return gr.update(visible=False), None, "{}", 1, gr.update(visible=False)
            
            source_key = selected_db_source['source_key']
            # Get full predefined params as base, then overlay DB params
            predefined_params = _get_source_details_by_key(source_key, predefined_list_val)['parameters'] if _get_source_details_by_key(source_key, predefined_list_val) else {}
            current_params = {**predefined_params, **selected_db_source['parameters']} # Merge, DB overrides predefined for max_pages
            
            return (
                gr.update(visible=True), # edit_source_details_group
                source_key, # edit_selected_db_source_key_state
                json.dumps(current_params, indent=2), # display all current params
                current_params.get('max_pages', 1), # editable max_pages
                gr.update(visible=True)  # update_max_pages_btn
            )
        edit_source_dropdown.change(
            fn=on_edit_source_dropdown_change,
            inputs=[edit_source_dropdown, current_db_scraper_config_state, predefined_sources_state],
            outputs=[
                edit_source_details_group, edit_selected_db_source_key_state,
                edit_source_params_display, edit_source_max_pages_input, update_max_pages_btn
            ]
        )

        # Update Max Pages button click
        def on_update_max_pages_click(s_key_to_edit, new_max_p_val, predefined_list_val, current_db_config_val):
            if not s_key_to_edit:
                return "No source selected to update.", current_db_config_val, *_update_scraper_tab_choices(predefined_list_val, current_db_config_val), gr.update(visible=False), gr.update(visible=False)
            
            db_source_detail = _get_db_config_details_by_key(s_key_to_edit, current_db_config_val)
            predefined_detail = _get_source_details_by_key(s_key_to_edit, predefined_list_val)

            if not db_source_detail or not predefined_detail: # Should exist if selected
                return f"Error: Details for source key '{s_key_to_edit}' not found for update.", current_db_config_val, *_update_scraper_tab_choices(predefined_list_val, current_db_config_val), gr.update(visible=False), gr.update(visible=False)

            # Start with predefined params, overlay existing DB params, then update max_pages
            updated_params = predefined_detail['parameters'].copy() # Base
            updated_params.update(db_source_detail['parameters'])   # Overlay DB specifics
            updated_params['max_pages'] = int(new_max_p_val)        # Apply edit

            new_db_config, status_msg_backend = backend_update_scraper_source_in_db(s_key_to_edit, db_source_detail['display_name'], updated_params)
            
            add_dd_upd, edit_dd_upd, rem_cbg_upd, db_json_upd = _update_scraper_tab_choices(predefined_list_val, new_db_config)
            return status_msg_backend, new_db_config, add_dd_upd, edit_dd_upd, rem_cbg_upd, db_json_upd, gr.update(visible=False), gr.update(visible=False) # Hide edit group
        update_max_pages_btn.click(
            fn=on_update_max_pages_click,
            inputs=[edit_selected_db_source_key_state, edit_source_max_pages_input, predefined_sources_state, current_db_scraper_config_state],
            outputs=[
                scraper_config_feedback_text, current_db_scraper_config_state,
                add_source_dropdown, edit_source_dropdown, remove_source_checkbox_group, current_db_config_display,
                edit_source_details_group, update_max_pages_btn # Hide these after update
            ]
        )

        # Remove Selected Sources button click
        def on_remove_selected_sources_click(display_names_to_remove, predefined_list_val, current_db_config_val):
            if not display_names_to_remove:
                return "No sources selected for removal.", current_db_config_val, *_update_scraper_tab_choices(predefined_list_val, current_db_config_val)

            source_keys_to_remove = []
            for dn in display_names_to_remove:
                s = next((src for src in current_db_config_val if src['display_name'] == dn), None)
                if s: source_keys_to_remove.append(s['source_key'])
            
            if not source_keys_to_remove: # Should not happen if display_names_to_remove is not empty and mapping works
                 return "Could not map selected names to source keys.", current_db_config_val, *_update_scraper_tab_choices(predefined_list_val, current_db_config_val)

            new_db_config = current_db_config_val # Start with current
            all_msgs = []
            for s_key_to_remove in source_keys_to_remove:
                new_db_config, status_msg_backend = backend_remove_scraper_source_from_db(s_key_to_remove) # Backend handles one by one, updates its internal state
                all_msgs.append(status_msg_backend)
            
            # After all removals, get the final state of choices
            add_dd_upd, edit_dd_upd, rem_cbg_upd, db_json_upd = _update_scraper_tab_choices(predefined_list_val, new_db_config) # new_db_config here is the final one from last backend_remove call
            
            return f"Removal process finished: {', '.join(all_msgs)}", new_db_config, add_dd_upd, edit_dd_upd, rem_cbg_upd, db_json_upd
        remove_selected_sources_btn.click(
            fn=on_remove_selected_sources_click,
            inputs=[remove_source_checkbox_group, predefined_sources_state, current_db_scraper_config_state],
            outputs=[
                scraper_config_feedback_text, current_db_scraper_config_state,
                add_source_dropdown, edit_source_dropdown, remove_source_checkbox_group, current_db_config_display
            ]
        )

        # --- Chat Event Handlers ---
        chat_inputs = [chat_input, chatbot, selected_db_state, app_config_state]
        send_btn.click(fn=chat_interface_backend, inputs=chat_inputs, outputs=[chatbot]).then(fn=lambda: gr.update(value=""), inputs=None, outputs=[chat_input])
        chat_input.submit(fn=chat_interface_backend, inputs=chat_inputs, outputs=[chatbot]).then(fn=lambda: gr.update(value=""), inputs=None, outputs=[chat_input])
        
        return demo