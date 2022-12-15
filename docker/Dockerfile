
ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

RUN conda install torchvision=0.7.0=py37_cu101

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir mmcv-full==1.5.3 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html && \
    pip install mmsegmentation==0.27.0

RUN apt-get update; apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /usr/local/bin
COPY docker_entry_point.py .
CMD ["python", "docker_entry_point.py"]

#    4  conda install torchvision=0.7.0=py37_cu101
#    5  pip install mmcv-full==1.5.3 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html
#    6  pip install mmsegmentation==0.27.0
#   10  apt-get update; apt-get install ffmpeg libsm6 libxext6  -y

