FROM python:3.8.12

RUN mkdir -p /work
WORKDIR /work
COPY requirements.txt .
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y libgl1-mesa-dev
RUN cat requirements.txt | xargs  -n 1 -L 1 pip install
COPY *.py yolov5s.onnx ./