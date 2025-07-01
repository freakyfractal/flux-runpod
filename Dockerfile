# CUDA 12.4 • PyTorch 2.3 • Python 3.11
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/data/hf-cache

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt /tmp/req.txt
RUN pip install --no-cache-dir -r /tmp/req.txt

# Handler code (only)
WORKDIR /workspace
COPY runpod_handler.py .

CMD ["python", "-u", "runpod_handler.py"]

