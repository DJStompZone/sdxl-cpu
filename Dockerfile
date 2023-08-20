FROM python:3.9-slim

LABEL maintainer="dj@deepai.org"

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev && \
    rm -rf /var/lib/apt/lists/* && \
    pip install diffusers transformers accelerate invisible_watermark mediapy

COPY sdxl.py /workspace/

CMD ["python", "/workspace/sdxl.py"]
