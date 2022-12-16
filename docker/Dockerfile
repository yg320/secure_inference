ARG PYTORCH="1.11.0"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

RUN conda install torchvision -y

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir mmcv-full==1.5.3 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11/index.html && \
    pip install mmsegmentation==0.27.0

RUN apt-get update; apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /usr/local/bin
COPY docker_entry_point.py .
CMD ["python", "docker_entry_point.py"]
