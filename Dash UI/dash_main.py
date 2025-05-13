import dash
from dash import dcc, html, Input, Output, State, no_update, callback_context, ALL
import dash_bootstrap_components as dbc
import base64
import os
import json

import backend

# --- Initial Backend System Setup & Data Load ---
backend.initialize_api_clients_backend()
backend.initialize_dropbox_client_backend()
backend.initialize_mongodb_client_backend()

INITIAL_DB_CHOICE = backend.get_default_db_choice_backend()
INITIAL_LOAD_STATUS_MESSAGE = "System Initialized. Select a database or upload files."
if INITIAL_DB_CHOICE:
    INITIAL_LOAD_STATUS_MESSAGE = backend.load_data_from_db(INITIAL_DB_CHOICE)
    print(f"Initial load from '{INITIAL_DB_CHOICE}': {INITIAL_LOAD_STATUS_MESSAGE}")
else:
    INITIAL_LOAD_STATUS_MESSAGE = "No primary database (Dropbox/MongoDB) configured. Please check .env settings. Uploads will be in-memory only unless a DB is configured and selected."
    print(INITIAL_LOAD_STATUS_MESSAGE)

AVAILABLE_MODELS_OPTIONS_FRONTEND = backend.get_available_models_options_backend()
IS_MONGO_CONFIGURED_FRONTEND = backend.get_mongo_uri_status_backend()
IS_DROPBOX_CONFIGURED_FRONTEND = backend.get_is_dropbox_configured_backend()

# --- Dash App Initialization ---
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.LUX, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)
server = app.server

# --- Layout Definition ---

def make_form_group(label_text, component, label_html_for=None, margin_bottom="mb-3"):
    return dbc.Form([
        dbc.Label(label_text, html_for=label_html_for, className="fw-semibold small text-muted"),
        component
    ], className=margin_bottom)

sidebar_header = dbc.Row(
    dbc.Col(
        html.H4(
            [html.I(className="fas fa-brain me-2"), "IntelLaw AI Suite"],
            className="text-white my-3 text-center fw-light"
        )
    ),
    className="bg-primary sticky-top",
    style={"boxShadow": "0 2px 4px rgba(0,0,0,0.1)"}
)

sidebar_tabs = dbc.Tabs(
    id="sidebar-tabs",
    active_tab="tab-config",
    children=[
        dbc.Tab(
            label="AI Settings",
            tab_id="tab-config",
            children=[
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(
                            dcc.Upload(
                                id='upload-config-json',
                                children=html.Div([html.I(className="fas fa-upload me-1"), ' Upload JSON']),
                                style={
                                    'width': '100%', 'height': '38px', 'lineHeight': '36px',
                                    'borderWidth': '1px', 'borderStyle': 'dashed',
                                    'borderRadius': '0.25rem', 'textAlign': 'center', 'cursor': 'pointer',
                                    'borderColor': '#ced4da', 'fontSize': '0.85em', 'color': '#6c757d'
                                }, multiple=False, accept='.json'
                            ), width=6, className="pe-1"
                        ),
                        dbc.Col(
                            dbc.Button(
                                [html.I(className="fas fa-download me-1"), "Download JSON"],
                                id="btn-download-config", color="secondary", outline=True,
                                className="w-100", style={'height': '38px', 'fontSize': '0.85em'}
                            ), width=6, className="ps-1"
                        )
                    ], className="mb-3"),
                    dcc.Download(id="download-config-json"),
                    html.Div(id="config-upload-status", className="mb-2 small text-muted"),
                    html.Hr(className="my-3"),
                    make_form_group(
                        "AI Models (Max 3)",
                        dcc.Dropdown(
                            id='model-selector', options=AVAILABLE_MODELS_OPTIONS_FRONTEND, multi=True,
                            value=[AVAILABLE_MODELS_OPTIONS_FRONTEND[0]['value']] if AVAILABLE_MODELS_OPTIONS_FRONTEND else []
                        ), label_html_for='model-selector'
                    ),
                    dbc.Row([
                        dbc.Col(dbc.Checkbox(id='vary-temp-checkbox', label="Vary Temperature", value=True), width="auto"),
                        dbc.Col(make_form_group("Base Temp.", dcc.Slider(id='temperature-slider', min=0, max=1, step=0.05, value=0.7, marks=None, tooltip={"placement": "top", "always_visible": False}), margin_bottom="mb-2"))
                    ]),
                     dbc.Row([
                        dbc.Col(dbc.Checkbox(id='vary-top-p-checkbox', label="Vary Top-P", value=False), width="auto"),
                        dbc.Col(make_form_group("Base Top-P", dcc.Slider(id='top-p-slider', min=0, max=1, step=0.05, value=0.9, marks=None, tooltip={"placement": "top", "always_visible": False}), margin_bottom="mb-2"))
                    ]),
                    make_form_group("System Prompt", dbc.Textarea(id='system-prompt-area', value="You are a helpful assistant. Answer questions strictly based on the provided context. If the context is insufficient or irrelevant to the question, state that you don't have enough information to answer based on the documents. Do not use outside knowledge.", rows=6, style={"fontSize": "0.85em"}), margin_bottom="mb-0")
                ], className="py-3 px-3")
            ],
            label_style={"fontSize": "0.85em", "padding": "0.5rem 0.75rem"},
            active_label_style={"fontWeight": "600"}
        ),
        dbc.Tab(
            label="Web Scraper", tab_id="tab-scraper",
            children=[
                dbc.CardBody([
                    make_form_group("Base URL", dbc.Input(id='scrape-base-url', placeholder="e.g., https://www.imy.se", value="https://www.imy.se"), label_html_for='scrape-base-url'),
                    make_form_group("Listing Endpoint", dbc.Input(id='scrape-listing-endpoint', placeholder="e.g., tillsyner", value="tillsyner"), label_html_for='scrape-listing-endpoint'),
                    make_form_group("Pagination Format", dbc.Input(id='scrape-pagination-format', placeholder="e.g., ?page= or page/", value="?page="), label_html_for='scrape-pagination-format'),
                    make_form_group("Number of Pages", dbc.Input(id='scrape-num-pages', placeholder="e.g., 3", type="number", value=1, min=1, step=1), label_html_for='scrape-num-pages'),
                    dbc.Button([html.I(className="fas fa-search-location me-2"), "Scrape & Process PDFs"], id='start-scraping-button', color="info", className="w-100 mt-3 btn-sm"),
                    html.Div(id='scraper-status-output', className="mt-3 p-2 border rounded small", style={'maxHeight': '150px', 'overflowY': 'auto', "backgroundColor": "#f0f0f0"})
                ], className="py-3 px-3")
            ],
            label_style={"fontSize": "0.85em", "padding": "0.5rem 0.75rem"}, active_label_style={"fontWeight": "600"}
        ),
        dbc.Tab(
            label="Manage Files", tab_id="tab-files",
            children=[
                dbc.CardBody([
                    dcc.Upload(
                        id='upload-pdf-sidebar',
                        children=html.Div([html.I(className="fas fa-file-arrow-up me-2"), 'Upload Local PDF(s)']),
                        style={
                            'width': '100%', 'height': '60px', 'lineHeight': '58px', 'borderWidth': '1px',
                            'borderStyle': 'dashed', 'borderRadius': '0.25rem', 'textAlign': 'center',
                            'cursor': 'pointer', 'borderColor': '#ced4da', 'color': '#6c757d', 'backgroundColor': '#f8f9fa'
                        }, multiple=True, className="mb-3"
                    ),
                    html.Div(id='sidebar-upload-status-display', className="mb-3 small", style={'maxHeight': '100px', 'overflowY': 'auto'}),
                    dbc.Label("Indexed Documents:", className="fw-semibold mb-2 small text-muted"),
                    dbc.ListGroup(id='stored-files-display-list', flush=True, style={'maxHeight': 'calc(100vh - 500px)', 'overflowY': 'auto'})
                ], className="py-3 px-3")
            ],
            label_style={"fontSize": "0.85em", "padding": "0.5rem 0.75rem"}, active_label_style={"fontWeight": "600"}
        ),
    ],
    className="mb-3 custom-sidebar-tabs"
)

sidebar_layout = html.Div(
    [
        sidebar_header,
        dbc.Container(
            [
                make_form_group(
                    "Active Database",
                    dbc.RadioItems(
                        id='db-selection-radioitems',
                        options=[
                            {'label': html.Span([html.I(className="fab fa-dropbox me-2"), " Dropbox"]), 'value': 'Dropbox', 'disabled': not IS_DROPBOX_CONFIGURED_FRONTEND},
                            {'label': html.Span([html.I(className="fas fa-database me-2"), " MongoDB"]), 'value': 'MongoDB', 'disabled': not IS_MONGO_CONFIGURED_FRONTEND}
                        ],
                        value=INITIAL_DB_CHOICE if (INITIAL_DB_CHOICE == "Dropbox" and IS_DROPBOX_CONFIGURED_FRONTEND) or \
                                                  (INITIAL_DB_CHOICE == "MongoDB" and IS_MONGO_CONFIGURED_FRONTEND) else \
                                                  ("Dropbox" if IS_DROPBOX_CONFIGURED_FRONTEND else ("MongoDB" if IS_MONGO_CONFIGURED_FRONTEND else "Dropbox")),
                        className="mt-1", inputClassName="me-2", labelCheckedClassName="fw-bold text-primary",
                        inline=True
                    ),
                    margin_bottom="mb-3"
                ),
                html.Hr(className="my-2"),
                sidebar_tabs,
            ],
            fluid=True,
            className="p-3 sidebar-content-area"
        )
    ],
    style={
        'position': 'fixed', 'top': 0, 'left': 0, 'bottom': 0,
        'width': '36rem',
        'boxShadow': '0 4px 12px rgba(0,0,0,0.15)',
        'overflowY': 'auto',
        'backgroundColor': '#FCFCFC'
    },
    className="border-end"
)

main_content_layout = html.Div([
    dbc.Container([
        dbc.Row(
            dbc.Col(
                html.H2([html.I(className="fas fa-comments me-2"), "Document Conversation"], className="my-4 text-center display-6 fw-light text-muted"),
            )
        ),
        dbc.Alert(
            id='main-notifications-alert',
            children=INITIAL_LOAD_STATUS_MESSAGE,
            is_open=True if INITIAL_LOAD_STATUS_MESSAGE else False,
            duration=8000,
            dismissable=True,
            color="primary",
            className="shadow-sm mb-3"
        ),
        dbc.Card(
            dbc.CardBody(
                html.Div(
                    id='chat-history-container',
                    style={'minHeight': 'calc(100vh - 280px)', 'maxHeight': 'calc(100vh - 230px)', 'overflowY': 'auto', 'padding': '20px 15px'}
                ), className="p-0"
            ),
            className="mb-3 shadow-lg chat-container-card",
            style={"backgroundColor": "white", "borderRadius":"0.5rem"}
        ),
        dbc.Row([
            dbc.Col(
                dbc.InputGroup([
                    dbc.Textarea(id='chat-user-input', placeholder="Type your question...", rows=2, className="shadow-sm-inset chat-input-area"),
                    dbc.Button([html.I(className="fas fa-paper-plane"), ""], id='send-chat-msg-button', color="primary", className="shadow-sm send-button")
                ]),
            )
        ], className="mb-3 align-items-center chat-input-row"),
    ], fluid=True, className="py-3 px-lg-5 px-md-4 px-sm-3")
], style={'marginLeft': '36rem', 'padding': '0', 'backgroundColor': '#f4f7f6'})


app.layout = html.Div([
    dcc.Store(id='selected-db-store', data=INITIAL_DB_CHOICE),
    dcc.Store(id='chat-history-store', data=[]),
    dcc.Store(id='app-configuration-store', data={
        "selected_models": [AVAILABLE_MODELS_OPTIONS_FRONTEND[0]['value']] if AVAILABLE_MODELS_OPTIONS_FRONTEND else [],
        "vary_temperature": True, "temperature": 0.7,
        "vary_top_p": False, "top_p": 0.9,
        "system_prompt": "You are a helpful assistant. Answer questions strictly based on the provided context. If the context is insufficient or irrelevant to the question, state that you don't have enough information to answer based on the documents. Do not use outside knowledge."
    }),
    dcc.Location(id='url', refresh=False),
    sidebar_layout,
    main_content_layout
])


# --- Callbacks ---
@app.callback(
    Output("download-config-json", "data"),
    Input("btn-download-config", "n_clicks"),
    State("app-configuration-store", "data"),
    prevent_initial_call=True,
)
def download_config(n_clicks, current_config_data):
    if n_clicks is None:
        return no_update
    return dict(content=json.dumps(current_config_data, indent=4), filename="ai_config.json")

@app.callback(
    [Output('model-selector', 'value', allow_duplicate=True),
     Output('vary-temp-checkbox', 'value', allow_duplicate=True),
     Output('temperature-slider', 'value', allow_duplicate=True),
     Output('vary-top-p-checkbox', 'value', allow_duplicate=True),
     Output('top-p-slider', 'value', allow_duplicate=True),
     Output('system-prompt-area', 'value', allow_duplicate=True),
     Output('app-configuration-store', 'data', allow_duplicate=True),
     Output('config-upload-status', 'children')],
    Input('upload-config-json', 'contents'),
    State('upload-config-json', 'filename'),
    State('app-configuration-store', 'data'),
    prevent_initial_call=True
)
def upload_and_apply_config(contents, filename, current_store_data):
    if contents is None:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update, ""

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        uploaded_config = json.loads(decoded.decode('utf-8'))

        models = uploaded_config.get("selected_models", current_store_data.get("selected_models", []))
        vary_t = uploaded_config.get("vary_temperature", current_store_data.get("vary_temperature", True))
        temp = uploaded_config.get("temperature", current_store_data.get("temperature", 0.7))
        vary_p = uploaded_config.get("vary_top_p", current_store_data.get("vary_top_p", False))
        top_p_val = uploaded_config.get("top_p", current_store_data.get("top_p", 0.9))
        sys_prompt = uploaded_config.get("system_prompt", current_store_data.get("system_prompt", ""))

        new_store_data = {
            "selected_models": models,
            "vary_temperature": vary_t,
            "temperature": temp,
            "vary_top_p": vary_p,
            "top_p": top_p_val,
            "system_prompt": sys_prompt
        }
        
        status_msg = dbc.Alert(f"Successfully loaded configuration from '{filename}'.", color="success", duration=4000, className="mt-2")
        return models, vary_t, temp, vary_p, top_p_val, sys_prompt, new_store_data, status_msg

    except Exception as e:
        print(f"Error processing uploaded config file: {e}")
        status_msg = dbc.Alert(f"Error loading config from '{filename}': {e}. Please ensure it's a valid JSON.", color="danger", duration=6000, className="mt-2")
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update, status_msg


@app.callback(
    Output('app-configuration-store', 'data', allow_duplicate=True),
    Input('model-selector', 'value'),
    Input('vary-temp-checkbox', 'value'), Input('temperature-slider', 'value'),
    Input('vary-top-p-checkbox', 'value'), Input('top-p-slider', 'value'),
    Input('system-prompt-area', 'value'),
    State('app-configuration-store', 'data'),
    prevent_initial_call=True
)
def update_app_config_from_ui(models, vary_t, temp, vary_p_ui, top_p_val_ui, sys_prompt, current_config):
    triggered_input_ids = [trigger['prop_id'].split('.')[0] for trigger in callback_context.triggered if trigger['value'] is not None]
    
    ui_inputs = ['model-selector', 'vary-temp-checkbox', 'temperature-slider', 
                 'vary-top-p-checkbox', 'top-p-slider', 'system-prompt-area']
    
    is_ui_trigger = any(ui_element_id in triggered_input_ids for ui_element_id in ui_inputs)

    if not is_ui_trigger and triggered_input_ids:
        return no_update 
    if not triggered_input_ids and not callback_context.triggered:
        return no_update

    if models is not None and len(models) > 3:
        models = models[:3]

    updated_data = {
        "selected_models": models if models is not None else current_config.get("selected_models", []),
        "vary_temperature": vary_t,
        "temperature": temp,
        "vary_top_p": vary_p_ui,
        "top_p": top_p_val_ui,
        "system_prompt": sys_prompt
    }
    if updated_data != current_config:
        return updated_data
    return no_update

@app.callback(
    Output('selected-db-store', 'data'),
    Output('main-notifications-alert', 'children', allow_duplicate=True),
    Output('main-notifications-alert', 'is_open', allow_duplicate=True),
    Output('main-notifications-alert', 'color', allow_duplicate=True),
    Output('stored-files-display-list', 'children', allow_duplicate=True),
    Input('db-selection-radioitems', 'value'),
    prevent_initial_call=True
)
def switch_database_and_reload(selected_db_val):
    if not selected_db_val:
        return no_update, "No database selected.", True, "warning", []

    status_message = backend.load_data_from_db(selected_db_val)
    files_list_backend = backend.get_stored_files_list_backend()
    files_list_items_display = []
    if files_list_backend:
        for f_item in files_list_backend:
            files_list_items_display.append(
                dbc.ListGroupItem(
                    [
                        html.Div(
                            [
                                html.I(className="fas fa-file-pdf me-2 text-danger"),
                                html.Span(f_item["name"], title=f_item["name"], style={"flexGrow": 1, "overflow": "hidden", "textOverflow": "ellipsis", "whiteSpace": "nowrap"}),
                            ], className="d-flex align-items-center"
                        ),
                        dbc.Button(html.I(className="fas fa-trash-alt"), id={'type': 'delete-file-btn', 'index': f_item["hash"]}, color="danger", size="sm", outline=True, className="ms-auto", title="Delete File")
                    ],
                    className="d-flex justify-content-between align-items-center p-2"
                )
            )
    else:
        files_list_items_display.append(dbc.ListGroupItem("No files in this database.", className="text-muted"))
    return selected_db_val, status_message, True, "primary", files_list_items_display


@app.callback(
    Output('sidebar-upload-status-display', 'children'),
    Output('stored-files-display-list', 'children', allow_duplicate=True),
    Output('main-notifications-alert', 'children', allow_duplicate=True),
    Output('main-notifications-alert', 'is_open', allow_duplicate=True),
    Output('main-notifications-alert', 'color', allow_duplicate=True),
    Input('upload-pdf-sidebar', 'contents'),
    State('upload-pdf-sidebar', 'filename'),
    State('selected-db-store', 'data'),
    prevent_initial_call=True
)
def handle_sidebar_pdf_upload_files(list_of_contents, list_of_names, selected_db):
    if list_of_contents is None:
        return no_update, no_update, no_update, False, no_update
    upload_alerts = []
    processed_any_new_successfully = False
    main_alert_msg = ""
    main_alert_color = "info"
    for content, name in zip(list_of_contents, list_of_names):
        try:
            content_type, content_string = content.split(',')
            decoded_content = base64.b64decode(content_string)
            status_msg, success = backend.process_uploaded_pdf_backend(decoded_content, name, selected_db)
            upload_alerts.append(dbc.Alert(status_msg, color="success" if success else "warning", dismissable=True, duration=8000, className="small"))
            if success:
                processed_any_new_successfully = True
                main_alert_msg = status_msg
                main_alert_color = "success"
            elif not main_alert_msg:
                main_alert_msg = status_msg
                main_alert_color = "warning"
        except Exception as e:
            upload_alerts.append(dbc.Alert(f"Error processing {name}: {e}", color="danger", dismissable=True, className="small"))
            if not main_alert_msg:
                main_alert_msg = f"Error processing {name}: {e}"
                main_alert_color = "danger"

    if processed_any_new_successfully:
        files_list_backend = backend.get_stored_files_list_backend()
        files_list_items_display = []
        if files_list_backend:
            for f_item in files_list_backend:
                 files_list_items_display.append(
                    dbc.ListGroupItem(
                        [
                            html.Div(
                                [
                                    html.I(className="fas fa-file-pdf me-2 text-danger"),
                                    html.Span(f_item["name"], title=f_item["name"], style={"flexGrow": 1, "overflow": "hidden", "textOverflow": "ellipsis", "whiteSpace": "nowrap"}),
                                ], className="d-flex align-items-center"
                            ),
                            dbc.Button(html.I(className="fas fa-trash-alt"), id={'type': 'delete-file-btn', 'index': f_item["hash"]}, color="danger", size="sm", outline=True, className="ms-auto", title="Delete File")
                        ],
                        className="d-flex justify-content-between align-items-center p-2"
                    )
                )
        else:
            files_list_items_display.append(dbc.ListGroupItem("No files after upload.", className="text-muted"))
        return upload_alerts, files_list_items_display, main_alert_msg, True, main_alert_color
    return upload_alerts, no_update, main_alert_msg if main_alert_msg else "No new files processed.", True if main_alert_msg else False, main_alert_color if main_alert_msg else "info"


@app.callback(
    Output('stored-files-display-list', 'children', allow_duplicate=True),
    Output('main-notifications-alert', 'children', allow_duplicate=True),
    Output('main-notifications-alert', 'is_open', allow_duplicate=True),
    Output('main-notifications-alert', 'color', allow_duplicate=True),
    Input({'type': 'delete-file-btn', 'index': ALL}, 'n_clicks'),
    State('selected-db-store', 'data'),
    prevent_initial_call=True
)
def delete_stored_file(n_clicks_list, selected_db_state):
    ctx = callback_context
    if not ctx.triggered_id or not any(click for click in n_clicks_list if click is not None):
        return no_update, no_update, False, no_update

    button_id = ctx.triggered_id
    file_hash_to_delete = button_id['index']
    status_message, success = backend.delete_file_from_store_backend(file_hash_to_delete, selected_db_state)
    files_list_backend = backend.get_stored_files_list_backend()
    files_list_items_display = []
    if files_list_backend:
        for f_item in files_list_backend:
            files_list_items_display.append(
                dbc.ListGroupItem(
                    [
                        html.Div(
                            [
                                html.I(className="fas fa-file-pdf me-2 text-danger"),
                                html.Span(f_item["name"], title=f_item["name"], style={"flexGrow": 1, "overflow": "hidden", "textOverflow": "ellipsis", "whiteSpace": "nowrap"}),
                            ], className="d-flex align-items-center"
                        ),
                        dbc.Button(html.I(className="fas fa-trash-alt"), id={'type': 'delete-file-btn', 'index': f_item["hash"]}, color="danger", size="sm", outline=True, className="ms-auto", title="Delete File")
                    ],
                    className="d-flex justify-content-between align-items-center p-2"
                )
            )
    else:
        files_list_items_display.append(dbc.ListGroupItem("No files remaining.", className="text-muted"))
    return files_list_items_display, status_message, True, "success" if success else "warning"


@app.callback(
    Output('stored-files-display-list', 'children', allow_duplicate=True),
    Input('url', 'pathname'),
    prevent_initial_call='initial_duplicate'
)
def populate_initial_file_list(_):
    files_list_backend = backend.get_stored_files_list_backend()
    files_list_items_display = []
    if files_list_backend:
        for f_item in files_list_backend:
            files_list_items_display.append(
                dbc.ListGroupItem(
                    [
                        html.Div(
                            [
                                html.I(className="fas fa-file-pdf me-2 text-danger"),
                                html.Span(f_item["name"], title=f_item["name"], style={"flexGrow": 1, "overflow": "hidden", "textOverflow": "ellipsis", "whiteSpace": "nowrap"}),
                            ], className="d-flex align-items-center"
                        ),
                        dbc.Button(html.I(className="fas fa-trash-alt"), id={'type': 'delete-file-btn', 'index': f_item["hash"]}, color="danger", size="sm", outline=True, className="ms-auto", title="Delete File")
                    ],
                    className="d-flex justify-content-between align-items-center p-2"
                )
            )
    else:
        files_list_items_display.append(dbc.ListGroupItem("No files found or database not loaded.", className="text-muted"))
    return files_list_items_display


@app.callback(
    Output('scraper-status-output', 'children'),
    Output('stored-files-display-list', 'children', allow_duplicate=True),
    Output('main-notifications-alert', 'children', allow_duplicate=True),
    Output('main-notifications-alert', 'is_open', allow_duplicate=True),
    Output('main-notifications-alert', 'color', allow_duplicate=True),
    Input('start-scraping-button', 'n_clicks'),
    State('scrape-base-url', 'value'), State('scrape-listing-endpoint', 'value'),
    State('scrape-pagination-format', 'value'), State('scrape-num-pages', 'value'),
    State('selected-db-store', 'data'),
    prevent_initial_call=True
)
def handle_web_scraping(n_clicks, base_url, endpoint, pagination, num_pages_str, selected_db):
    if n_clicks is None:
        return no_update, no_update, no_update, False, no_update
    if not all([base_url, endpoint, pagination, num_pages_str]):
        return [dbc.Alert("All scraper fields are required.", color="warning", className="small")], no_update, no_update, False, no_update
    try:
        num_pages = int(num_pages_str)
        if num_pages < 1:
            return [dbc.Alert("Number of pages must be at least 1.", color="warning", className="small")], no_update, no_update, False, no_update
    except ValueError:
        return [dbc.Alert("Invalid number for 'Num Pages'.", color="warning", className="small")], no_update, no_update, False, no_update

    status_updates_from_backend = backend.run_web_scraping_and_processing_backend(base_url, endpoint, pagination, num_pages, selected_db)
    scraper_status_display = [html.P(msg, className="mb-1") for msg in status_updates_from_backend]
    files_list_backend = backend.get_stored_files_list_backend()
    files_list_items_display = []
    if files_list_backend:
        for f_item in files_list_backend:
             files_list_items_display.append(
                dbc.ListGroupItem(
                    [
                        html.Div(
                            [
                                html.I(className="fas fa-file-pdf me-2 text-danger"),
                                html.Span(f_item["name"], title=f_item["name"], style={"flexGrow": 1, "overflow": "hidden", "textOverflow": "ellipsis", "whiteSpace": "nowrap"}),
                            ], className="d-flex align-items-center"
                        ),
                        dbc.Button(html.I(className="fas fa-trash-alt"), id={'type': 'delete-file-btn', 'index': f_item["hash"]}, color="danger", size="sm", outline=True, className="ms-auto", title="Delete File")
                    ],
                    className="d-flex justify-content-between align-items-center p-2"
                )
            )
    else:
        files_list_items_display.append(dbc.ListGroupItem("No files after scraping.", className="text-muted"))
    main_alert_msg = status_updates_from_backend[-1] if status_updates_from_backend else "Scraping process initiated."
    main_alert_color = "success" if "finished" in main_alert_msg.lower() and "error" not in main_alert_msg.lower() else "info"
    if "error" in main_alert_msg.lower() or "fail" in main_alert_msg.lower():
        main_alert_color = "danger"
    return scraper_status_display, files_list_items_display, main_alert_msg, True, main_alert_color


@app.callback(
    [Output('chat-history-container', 'children'),
     Output('chat-user-input', 'value'),
     Output('chat-history-store', 'data')],
    Input('send-chat-msg-button', 'n_clicks'),
    State('chat-user-input', 'value'),
    State('chat-history-store', 'data'),
    State('app-configuration-store', 'data'),
    State('selected-db-store', 'data'),
    prevent_initial_call=True
)
def handle_chat_interaction(n_clicks, user_input_val, current_chat_history_tuples, app_config, selected_db):
    if n_clicks is None or not user_input_val or not user_input_val.strip():
        return no_update, no_update, no_update

    current_chat_history_tuples.append({"sender": "User", "message": user_input_val})
    ai_response_list_of_dicts = backend.generate_chat_responses_backend(user_input_val, app_config, selected_db)
    for resp_data in ai_response_list_of_dicts:
        current_chat_history_tuples.append({
            "sender": "AI", "message": resp_data['response'], "model_info": resp_data['model_info_str']
        })

    chat_display_elements = []
    for item_idx, item in enumerate(current_chat_history_tuples):
        is_user = item['sender'] == "User"
        
        bubble_specific_class = "user-bubble" if is_user else "ai-bubble"
        alignment_class = "justify-content-end" if is_user else "justify-content-start" 

        message_card = dbc.Card(
            dbc.CardBody(html.Pre(item['message'])),
            className=f"chat-bubble-card {bubble_specific_class}",
        )
        
        message_block_elements = []

        if not is_user and item.get('model_info'):
            message_block_elements.append(
                 html.Div(
                    html.Small(item['model_info']),
                    className="model-info-header text-muted mb-1",
                )
            )
        
        message_block_elements.append(message_card)

        chat_display_elements.append(
            dbc.Row(
                dbc.Col(
                    html.Div(message_block_elements, className="d-inline-flex flex-column"),
                    width="auto",
                    style={'maxWidth': '80%'} 
                ),
                className=f"d-flex {alignment_class} mb-3",
            )
        )
    return chat_display_elements, "", current_chat_history_tuples


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)