# Purpose

The goal of this project is to show the implementation of the paper _Deep Leaning-based segmentation of post-mortem human's olfactory bulb structures in X-ray phase-contrast tomography_.

![web](![alt text](https://github.com/ankhafizov/olfactory-bulb-segmentation/blob/master/image.jpg?raw=true))

# Launch

## AWS server

Disclaimer: may be tmporary not available.

https://ec2-52-73-12-215.compute-1.amazonaws.com/

## Locally, sing Docker

Pull this repository and execute^

```
docker-compose up
```
open:

http://0.0.0.0:5000

## Locally, without Docker

Install python (3.8.12 is recomended) neccessary dependencies:

```
pip install -r http_client/requirements.txt -r sample_segmentation/requirements.txt -r layers_segmentation/requirements.txt
```

than open 3 console tab and execute in each tap separately

```
python http_client run.py
```
```
layers_segmentation server.py
```
```
sample_segmentation server.py
```
open:

http://0.0.0.0:5000

# Architecture



# Futher investigations

- enable CUDA for inference
- add link to published paper and citation column
