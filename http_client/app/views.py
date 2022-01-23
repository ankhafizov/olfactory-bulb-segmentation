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


def read_bytes(buffer):
    byte_stream = buffer.read()
    buffer = buffer.seek(0)
    return byte_stream


def normalize_0_1_numpy(img):
    return (img - img.min()) / (img.max() - img.min())


def decode_buf_image_file_to_numpy(image_file_bytes, dtype):
    nparr = np.frombuffer(image_file_bytes, dtype=np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    img_np = img_np.astype(dtype)
    return img_np


def request_server(buffered_image_file, host, ip, filetype="image"):
    response = request_http.post(f"http://{host}:{ip}/upload-raw-image",
                                 files={filetype: buffered_image_file})
    buffered_image_file = buffered_image_file.seek(0)
    return response


def enhance_contrast(img_numpy):
    img_numpy = (img_numpy - img_numpy.min()) / (img_numpy.max() - img_numpy.min())
    return exposure.equalize_adapthist(img_numpy, clip_limit=0.03)


def highlight_background(mask, ax):
    ax.imshow(~mask, alpha=0.3, cmap="Reds")


def find_background(flask_request):
    buffered_image_file = flask_request.files["image"]

    image_orig = decode_buf_image_file_to_numpy(
        read_bytes(buffered_image_file), dtype=np.float32)

    image_orig = enhance_contrast(image_orig)

    response = request_server(
        buffered_image_file, "127.0.0.1", "5001")
    print("respondse recieved successfully!")
    image_mask = decode_buf_image_file_to_numpy(
        response.content, dtype=bool)

    return image_orig, image_mask


@app.route('/', methods=["POST", "GET"])
def upload_file():
    if request.method == "POST":
        if request.files:
            requested_out_info = request.form.getlist("out-info")

            if not "layers" in requested_out_info:
                image_orig, image_mask = find_background(request)

                fig, ax = plt.subplots(1, 1, figsize=(
                    10, 10), constrained_layout=True)
                ax.imshow(image_orig, cmap="gray")
                highlight_background(image_mask, ax)
                ax.axis("off")
                fig.savefig(os.path.join(APP_ROOT, OUTPUT_SAVE_PATH))
            else:
                pass
                # image = request.files["image"]
                # image_orig = decode_image_file_to_numpy(image.read())
                # image = image.seek(0)

                # response = request_http.post(
                #     "http://127.0.0.1:5002/upload-raw-image", files={"image": request.files["image"]})

                # image_mask = decode_image_file_to_numpy(response.content)
                # # image_orig = exposure.equalize_adapthist(image_orig, clip_limit=0.03)
                # # print(image)

                # fig, ax = plt.subplots(1, 1, figsize=(
                #     10, 10), constrained_layout=True)
                # #image_mask = np.array(image_mask, dtype=bool)
                # #ax.imshow(image_orig, cmap="gray")
                # #ax.imshow(~image_mask, alpha=0.3, cmap="Reds")

                # ax.imshow(image_mask)
                # ax.axis("off")
                # fig.savefig(os.path.join(APP_ROOT, OUTPUT_SAVE_PATH))

            return render_template("main.html", output_img_url=OUTPUT_SAVE_PATH)

    return render_template("main.html")
