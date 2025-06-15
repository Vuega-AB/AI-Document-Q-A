#!/bin/sh

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Start Gradio Worker in the Background ---
echo "INFO: Starting Gradio/Uvicorn worker process..."
uvicorn gradio_server:fast_api_app --host 0.0.0.0 --port 7860 --workers 1 &
echo "INFO: Gradio worker started in the background."


# --- Start Flask Web Server in the Foreground ---
echo "INFO: Starting Flask/Gunicorn web process..."
# This is the main process. It will bind to the port Render provides.
# The --access-logfile - flag prints access logs to standard output, which is helpful for debugging.
gunicorn --workers 3 --bind 0.0.0.0:$PORT --access-logfile - "main_flask_app:app"