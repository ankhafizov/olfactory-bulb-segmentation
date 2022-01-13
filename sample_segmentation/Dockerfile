FROM python:3.8.12

COPY requirements.txt .
RUN apt-get update &&\
    /usr/local/bin/python -m pip install --upgrade pip &&\
    pip install -r requirements.txt

COPY . /usr/src/sample_segmentation
WORKDIR /usr/src/sample_segmentation

CMD [ "python", "predict.py"]
