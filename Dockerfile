FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

LABEL maintainer="dj@deepai.org"

WORKDIR /workspace

RUN apt-get update && apt-get install git ffmpeg libsm6 libxext6 libgl1-mesa-glx -y && pip3 install -r requirements.txt

COPY sdxl.py /workspace/

CMD ["python", "/workspace/sdxl.py"]
