FROM nvcr.io/nvidia/pytorch:21.09-py3

LABEL maintainer="dj@deepai.org"

WORKDIR /workspace

RUN apt-get update && apt-get install git ffmpeg libsm6 libxext6 libgl1-mesa-glx -y && pip install -r requirements.txt

COPY sdxl.py /workspace/

CMD ["python", "/workspace/sdxl.py"]
