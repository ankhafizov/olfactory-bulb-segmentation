import os
from app import app
from flask import render_template, request
import requests as request_http
import numpy as np
import cv2

import matplotlib.pyplot as plt
from skimage import exposure


APP_ROOT = "app"
OUTPUT_SAVE_PATH = f"static/img/output.png"


def decode_image_file_to_numpy(image_file):
    nparr = np.frombuffer(image_file, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    return img_np


@app.route('/', methods=["POST", "GET"])
def upload_file():
    if request.method == "POST":
        if request.files:

            image = request.files["image"]
            image_orig = decode_image_file_to_numpy(image.read())
            image = image.seek(0)

            response = request_http.post("http://127.0.0.1:5001/upload-raw-image", files={"image" : request.files["image"]})

            image_mask = decode_image_file_to_numpy(response.content)
            image_orig = exposure.equalize_adapthist(image_orig, clip_limit=0.03)
            # print(image)

            fig, ax = plt.subplots(1, 1, figsize=(10, 10), constrained_layout=True)
            image_mask = np.array(image_mask, dtype=bool)
            ax.imshow(image_orig, cmap="gray")
            ax.imshow(~image_mask, alpha=0.3, cmap="Reds")
            ax.axis("off")
            fig.savefig(os.path.join(APP_ROOT, OUTPUT_SAVE_PATH))

            return render_template("main.html", output_img_url=OUTPUT_SAVE_PATH)
    
    return render_template("main.html")
