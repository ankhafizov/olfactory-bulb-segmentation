FROM python:3.8.12

COPY requirements.txt .
RUN apt-get update &&\
    /usr/local/bin/python -m pip install --upgrade pip &&\
    pip install -r requirements.txt
RUN apt-get update &&\
    apt-get install ffmpeg libsm6 libxext6  -y

COPY . /usr/src/sample_segmentation
WORKDIR /usr/src/sample_segmentation
RUN ls

EXPOSE 5000

CMD [ "python", "run.py"]
