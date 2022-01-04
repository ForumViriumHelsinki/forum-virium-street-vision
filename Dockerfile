FROM python:3.8-slim-buster
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY 2021-12-31-light-augments-val-loss-0_085 2021-12-31-light-augments-val-loss-0_085
COPY predict.py predict.py
