FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y git make curl jq && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir dvc[s3] mlflow fastapi uvicorn pytest pandas scikit-learn
WORKDIR /workspace
