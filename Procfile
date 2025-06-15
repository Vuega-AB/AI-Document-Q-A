web: gunicorn app:app
worker: uvicorn gradio_server:fast_api_app --host 0.0.0.0 --port $PORT --workers 1