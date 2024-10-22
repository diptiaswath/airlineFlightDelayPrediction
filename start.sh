#!/bin/sh

# Start FastAPI server on port 8000
uvicorn app_interface.main:skyflow --reload --host 0.0.0.0 --port 8000 &

# Start StreamLit app on port 8501
streamlit run app_interface/skyflow_app.py \
    --server.port 8501 \
    --server.sslCertFile /code/ssl_certs/fullchain.pem \
    --server.sslKeyFile /code/ssl_certs/privkey.pem

# Wait indefinitely to keep the container running
wait