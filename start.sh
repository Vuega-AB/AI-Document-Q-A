#!/bin/bash

# This script starts the two processes for the application.

# 1. Start the Gradio worker process in the background.
#    It will run on port 7860 internally. The --workers 1 flag is crucial for free plans.
echo "Starting Gradio worker..."
uvicorn gradio_server:fast_api_app --host 0.0.0.0 --port 7860 --workers 1 &

# 2. Start the Flask web server in the foreground.
#    This is the main process that Render will monitor. It will use the
#    $PORT environment variable provided by Render.
echo "Starting Flask web server..."
gunicorn app:app --bind 0.0.0.0:$PORT