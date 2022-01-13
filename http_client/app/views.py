import os
from app import app
from flask import render_template, request
import requests as request_http
import numpy as np
import cv2
from PIL import Image


APP_ROOT = "app"
OUTPUT_SAVE_PATH = f"static/img/output.png"


def decode_image_file_to_Image(image_file):
    nparr = np.frombuffer(image_file, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    return Image.fromarray(img_np)


@app.route('/', methods=["POST", "GET"])
def upload_file():
    if request.method == "POST":
        if request.files:

            image = request.files["image"]
            response = request_http.post("http://127.0.0.1:5001/upload-raw-image", files={"image" : image})

            image_mask = decode_image_file_to_Image(response.content)
            image_mask.save(os.path.join(APP_ROOT, OUTPUT_SAVE_PATH))

            return render_template("main.html", output_img_url=OUTPUT_SAVE_PATH)
    
    return render_template("main.html")
