# app_ui.py
import gradio as gr
import os
import sys
import asyncio

# Assuming your backend logic is in backend.py
from backend import (
    get_current_file_list_md_backend,
    update_app_config_backend,
    switch_db_backend,
    handle_pdf_upload_backend,
    chat_interface_backend,
    run_scraper_backend,
    AVAILABLE_MODELS_NAMES,
    MODEL_NAME_TO_ID_MAP,
    BACKEND_INITIAL_LOAD_MSG
)

# Custom CSS for styling and responsiveness
CUSTOM_CSS = """
/* Global background */
body, .gradio-container { background-color: #f0f4f8 !important; } /* Ensure body and container get the color */

/* Main layout styling */
.gradio-container { font-family: 'Inter', sans-serif; } /* Example: Using a nice sans-serif font */

/* Controls Column Styling */
.controls-column .gr-panel { /* Target panels within the controls column if needed */
    /* background-color: #ffffff; */ /* Example: if you want a different bg for panels */
    /* border-radius: 1rem; */
    /* box-shadow: 0 4px 12px rgba(0,0,0,0.05); */
}

/* Chat bubbles styling */
.gradio-chatbot .message.bot {
    background-color: #2563eb !important; /* Slightly adjusted blue */
    color: #ffffff !important;
    border-radius: 1rem 0.25rem 1rem 1rem !important; /* Asymmetric rounding for bot */
    padding: 0.75rem 1rem !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
    max-width: 80% !important;
    align-self: flex-start !important; /* Explicitly align left */
    margin-left: 0.5rem;
    margin-right: auto;
}
.gradio-chatbot .message.user {
    background-color: #e0f2fe !important; /* Lighter blue for user */
    color: #0c4a6e !important; /* Darker text for contrast */
    border-radius: 0.25rem 1rem 1rem 1rem !important; /* Asymmetric rounding for user */
    padding: 0.75rem 1rem !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
    max-width: 80% !important;
    align-self: flex-end !important; /* Explicitly align right */
    margin-right: 0.5rem;
    margin-left: auto;
}

/* Chat container padding and message spacing */
.gradio-chatbot .messages {
    padding: 1rem !important;
    display: flex;
    flex-direction: column;
    gap: 0.75rem; /* Space between messages */
}

/* Chat Input Area */
#chat-input-container {
    margin-top: 1rem;
    padding: 0.5rem;
    /* background-color: #ffffff; */ /* Optional: if you want a distinct bg for input bar */
    /* border-radius: 1rem; */
    /* box-shadow: 0 -2px 8px rgba(0,0,0,0.05); */ /* Subtle top shadow */
}
#chat-input-container textarea {
    border-radius: 0.75rem !important; /* Rounded corners for textbox */
    border: 1px solid #cbd5e1 !important; /* Subtle border */
    padding: 0.75rem !important;
}
#chat-input-container button { /* Style the send button specifically if needed */
    height: 100%; /* Make button same height as textbox */
}


/* Slider bar size - adjust if too chunky */
.gr-slider .track, .gr-slider .thumb {
    height: 1rem !important; /* Reduced slightly */
}
.gr-slider .thumb {
    width: 1rem !important; /* Ensure thumb is proportional */
}

/* Responsive layout for small screens */
@media (max-width: 768px) {
  .gr-row {
    flex-direction: column !important;
  }
  .controls-column { /* Ensure controls column takes full width on small screens */
    min-width: 100% !important;
  }
  .gradio-chatbot .message.bot, .gradio-chatbot .message.user {
    max-width: 90% !important; /* Allow slightly wider bubbles on mobile */
  }
}

/* General Button rounding and font */
.gr-button {
    border-radius: 0.75rem !important;
    font-weight: 500 !important;
    padding: 0.6rem 1.2rem !important; /* Adjust padding for better feel */
}
.gr-button.gr-button-primary { /* Primary button specific styling */
    background-color: #1d4ed8 !important; /* Ensure primary color from theme is strong */
    color: white !important;
}

/* Accordion and Tabs styling for clarity */
.gr-accordion .gr-button { /* Accordion header */
    /* background-color: #eef2ff !important; */
    /* border-radius: 0.5rem !important; */
}
.gr-tabs .gr-button { /* Tab buttons */
    /* border-bottom: 2px solid transparent !important; */
}
.gr-tabs .gr-button.selected {
    /* border-bottom-color: #1d4ed8 !important; */ /* Highlight selected tab */
    /* color: #1d4ed8 !important; */
}
"""

with gr.Blocks(
    css=CUSTOM_CSS,
    theme=gr.themes.Soft(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.sky,
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"], # Adding a nice font
    ),
    title="IntelLaw Gradio"
) as demo:
    # --- State Variables ---
    selected_db_state = gr.State(value="Dropbox")
    initial_selected_model_id = []
    if AVAILABLE_MODELS_NAMES:
        first = AVAILABLE_MODELS_NAMES[0]
        if first in MODEL_NAME_TO_ID_MAP:
            initial_selected_model_id = [MODEL_NAME_TO_ID_MAP[first]]
    app_config_state = gr.State(value={
        "selected_models": initial_selected_model_id,
        "vary_temperature": True, "temperature": 0.7,
        "vary_top_p": False, "top_p": 0.9,
        "system_prompt": (
            "You are a helpful assistant. Answer questions strictly based on the provided context. "
            "If there is no context, say 'I don't have enough information to answer that.'"
        )
    })

    # --- Header ---
    gr.Markdown("# 📄 IntelLaw - Chat with Documents")

    # --- Layout ---
    with gr.Row(equal_height=False): # Set to False if columns have vastly different content heights
        with gr.Column(scale=2, min_width=420, elem_classes=["controls-column"]): # Increased scale and min_width
            gr.Markdown("### 🛠️ Controls")
            with gr.Accordion("Database Backend", open=True):
                db_radio = gr.Radio(
                    label="Choose Database", choices=["Dropbox", "MongoDB"], value="Dropbox"
                )
                db_status = gr.Textbox(label="DB Status", interactive=False, value=BACKEND_INITIAL_LOAD_MSG)

            with gr.Tabs():
                with gr.TabItem("🧠 Config"):
                    model_selector = gr.Dropdown(
                        label="AI Models (Max 3)", choices=AVAILABLE_MODELS_NAMES,
                        value=[AVAILABLE_MODELS_NAMES[0]] if AVAILABLE_MODELS_NAMES else [],
                        multiselect=True, max_choices=3
                    )
                    vary_temp = gr.Checkbox(label="Vary Temperature", value=True)
                    temp_slider = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, step=0.01, value=0.7)
                    vary_top_p = gr.Checkbox(label="Vary Top-P", value=False)
                    top_p_slider = gr.Slider(label="Top-P", minimum=0.0, maximum=1.0, step=0.01, value=0.9)
                    system_prompt = gr.Textbox(label="System Prompt", lines=4, value=app_config_state.value["system_prompt"])
                    update_config = gr.Button("Update Config", variant="secondary")
                    config_status = gr.Textbox(label="Config Status", interactive=False)

                with gr.TabItem("📁 Stored Files"):
                    file_uploader = gr.Files(label="Upload PDFs", file_count="multiple", type="filepath") # Default uses temp paths
                    upload_status = gr.Textbox(label="Upload Status", interactive=False, lines=3)
                    stored_files_md = gr.Markdown(value=get_current_file_list_md_backend())

                with gr.TabItem("🌐 Web Scraper"):
                    base_url = gr.Textbox(label="Base URL", value="https://www.imy.se") # Example default
                    listing_endpoint = gr.Textbox(label="Listing Endpoint (e.g., 'tillsyner')", value="tillsyner")
                    pagination = gr.Textbox(label="Pagination Format (e.g. '?page=')", value="?query=&page=")
                    pages_to_check = gr.Number(label="Num Pages to Check", value=1, minimum=1, precision=0)
                    scrape_btn = gr.Button("Start Scraping & Process", variant="secondary")
                    scraper_output = gr.Textbox(label="Scraper Output", lines=5, max_lines=10, interactive=False)

        with gr.Column(scale=3):
            gr.Markdown("### 💬 Chat Interface")
            chatbot = gr.Chatbot(
                label="IntelLaw Chatbot",
                height=600, # Increased height
                show_copy_button=True,
                bubble_full_width=False # Crucial for custom bubble styling and alignment
            )
            with gr.Row(elem_id="chat-input-container"):
                chat_input = gr.Textbox(
                    show_label=False,
                    placeholder="Ask anything about the documents...",
                    scale=5, # Textbox takes more relative space
                    container=False # Important for tight layout with button
                )
                send_btn = gr.Button("Send", scale=1, variant="primary") # Primary variant for emphasis

            gr.Markdown("### 🤖 Model Responses (Details)")
            responses_html = gr.HTML()

    # --- Event Handlers ---
    db_radio.change(
        fn=switch_db_backend,
        inputs=[db_radio, selected_db_state],
        outputs=[selected_db_state, db_status, stored_files_md]
    )

    update_config.click(
        fn=update_app_config_backend,
        inputs=[model_selector, vary_temp, temp_slider, vary_top_p, top_p_slider, system_prompt, app_config_state],
        outputs=[config_status, app_config_state]
    )

    file_uploader.upload(
        fn=handle_pdf_upload_backend,
        inputs=[file_uploader, selected_db_state],
        outputs=[upload_status, stored_files_md]
    )

    chat_inputs = [chat_input, chatbot, selected_db_state, app_config_state]
    chat_outputs = [chatbot, responses_html]

    def clear_input_fn(): return gr.update(value="")

    send_btn.click(
        fn=chat_interface_backend,
        inputs=chat_inputs, outputs=chat_outputs
    ).then(
        fn=clear_input_fn, outputs=chat_input # Use the new way of referencing outputs for .then()
    )
    chat_input.submit(
        fn=chat_interface_backend, inputs=chat_inputs, outputs=chat_outputs
    ).then(
        fn=clear_input_fn, outputs=chat_input
    )

    scrape_btn.click(
        fn=run_scraper_backend,
        inputs=[base_url, listing_endpoint, pagination, pages_to_check, selected_db_state],
        outputs=[scraper_output, stored_files_md]
    )

if __name__ == '__main__':
    demo.launch()