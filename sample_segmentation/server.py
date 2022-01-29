from flask import Flask, send_file, request
import logging

from predict import *
import numpy as np
import cv2
import io
import logging


def decode_image_file_to_Image(image_file):
    nparr = np.frombuffer(image_file.read(), np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    return Image.fromarray(img_np)


def encode_prediction_to_byte_stream(image):
    image = Image.fromarray(image)
    frame_in_bytes = io.BytesIO()
    image.save(frame_in_bytes, format = "PNG")
    frame_in_bytes.seek(0)
    return frame_in_bytes


def flask_app(configs):
    app = Flask(__name__)

    @app.route('/upload-raw-image', methods=['POST'])
    def start():
        image_file = request.files['image']
        image = decode_image_file_to_Image(image_file)
        logging.info("recieved image")
        mask = predict(image, configs)

        mask = encode_prediction_to_byte_stream(mask)

        logging.info("posting mask")
        return send_file(mask, as_attachment=False,
                         download_name='mask.png',
                         mimetype='image/png')

    return app

if __name__ == '__main__':
    configs = load_configs()
    app = flask_app(configs)
    log = logging.getLogger('werkzeug')
    log.disabled = True
    app.run(debug=configs["debug_mode"], port=configs["port"], host=configs["host"])