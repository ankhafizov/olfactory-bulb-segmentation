# Olfactory Bulb Segmentation

<p align="left">
  <a href="https://ec2-52-73-12-215.compute-1.amazonaws.com/">
    <img alt="shield_aws" src="https://img.shields.io/badge/AWS-service-yellow">
  </a>
</p>
</br>

# Purpose

The goal of this project is to show the implementation of the paper _Deep Leaning-based segmentation of post-mortem human's olfactory bulb structures in X-ray phase-contrast tomography_.

![web](https://github.com/ankhafizov/olfactory-bulb-segmentation/blob/master/ob_screen_web.png?raw=true)

# Launch

## AWS server

Disclaimer: may be tmporary not available.

https://ec2-52-73-12-215.compute-1.amazonaws.com/

## Locally, using Docker

Pull this repository and execute:

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

The app is developed using the microservice approach. Each server is launched in Docker (blue containers on the image below), which are interconnected in one docker-compose file (red one). 

![web](https://github.com/ankhafizov/olfactory-bulb-segmentation/blob/master/architecture.png?raw=true)

In AWS also Apache 2.4 was utilised as a reverse proxy for port 5000.

We hope that this scheme will be helpful if new features should be implemented. In that case, the corresponding code should be put in the Docker container, and connected to Central Processor (see http_client/app/views.py) afterwards.

# Futher investigations

- enable CUDA for inference
- add link to published paper and citation column
- run flask WSGI as gunicorn
