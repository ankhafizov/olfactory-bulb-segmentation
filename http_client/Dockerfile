FROM python:3.8.12

COPY requirements.txt .
RUN apt-get update &&\
    /usr/local/bin/python -m pip install --upgrade pip &&\
    pip install -r requirements.txt
RUN apt-get update &&\
    apt-get install ffmpeg libsm6 libxext6  -y

COPY . /usr/src/http_client
WORKDIR /usr/src/http_client
RUN ls

CMD [ "python", "run.py"]
