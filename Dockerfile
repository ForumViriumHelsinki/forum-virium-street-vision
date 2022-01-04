FROM python:3.8-slim-buster
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY 2021-12-31-light-augments-val-loss-0_085 2021-12-31-light-augments-val-loss-0_085
COPY predict.py predict.py
