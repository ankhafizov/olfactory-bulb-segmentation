import os
from flask import Flask, redirect, request

from predict import *
import numpy as np
import cv2
import matplotlib.pyplot as plt


HEADERS = {'Content-type': 'application/json', 'Accept': 'text/plain'}

def decode_image_file_to_Image(image_file):
    nparr = np.frombuffer(image_file.read(), np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    return Image.fromarray(img_np)


def flask_app(configs):
    app = Flask(__name__)

    @app.route('/upload-raw-image', methods=['POST'])
    def start():
        image = request.files['image']
        image = decode_image_file_to_Image(image)
        mask = predict(image, configs)

        plt.imshow(mask)
        plt.show()

        print(image)
        return redirect(request.url, 200)

    return app

if __name__ == '__main__':
    configs = load_configs()
    app = flask_app(configs)
    app.run(port=configs["port"], host=configs["host"])