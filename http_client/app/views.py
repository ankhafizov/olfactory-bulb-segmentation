from asyncio.log import logger
import os, io
import re
from app import app
from flask import render_template, request
import requests as request_http
import numpy as np
import cv2
from PIL import Image
import matplotlib.patches as mpatches

import matplotlib.pyplot as plt
from skimage import exposure


APP_ROOT = "app"
OUTPUT_SAVE_PATH = f"static/img/output.png"
ALPHA_BACKGROUND = 0.3

BACKGROUND_SEGMENTATOR_IP = "127.0.0.1"
BACKGROUND_SEGMENTATOR_HOST = "5001"

LAYER_SEGMENTATOR_IP = "127.0.0.1"
LAYER_SEGMENTATOR_HOST = "5002"


# =========================== Buffering functions ================================


def read_bytes(buffer):
    byte_stream = buffer.read()
    buffer = buffer.seek(0)
    return byte_stream


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


def encode_array_to_byte_stream(image_arr):
    image_arr = (image_arr - image_arr.min()) / (image_arr.max() - image_arr.min())
    image_arr = (image_arr * 255).astype(np.uint8)
    image_arr = Image.fromarray(image_arr)
    frame_in_bytes = io.BytesIO()
    image_arr.save(frame_in_bytes, format = "PNG")
    frame_in_bytes.seek(0)
    return frame_in_bytes

# ======================== Suplementary functions ================================

def enhance_contrast(img_numpy):
    img_numpy = (img_numpy - img_numpy.min()) / (img_numpy.max() - img_numpy.min())
    return exposure.equalize_adapthist(img_numpy, clip_limit=0.03)


def highlight_background(mask, ax, color="Red", alpha=ALPHA_BACKGROUND):
    #color = "Red" or "Blue"
    im = ~mask
    ax.imshow(im, alpha=alpha, cmap=color+"s")
    ax.plot(0, 0, "-", c=color, label="Background")

    background_patch = mpatches.Patch(color=color, label="Background")
    return background_patch


def draw_layer_contours(layer_mask, ax, color):
    #color = "Red" or "Blue"
    ax.contour(layer_mask, colors=[color])
    contour_patch = mpatches.Patch(color=color, label='Layers')
    return contour_patch


def find_sample(flask_request):
    buffered_image_file = flask_request.files["image"]
    image_orig = decode_buf_image_file_to_numpy(
        read_bytes(buffered_image_file), dtype=np.float32)
    image_orig = enhance_contrast(image_orig)

    response = request_server(buffered_image_file,
                              BACKGROUND_SEGMENTATOR_IP,
                              BACKGROUND_SEGMENTATOR_HOST)

    logger.info("response find_sample received successfully!")
    image_mask = decode_buf_image_file_to_numpy(response.content, dtype=bool)

    return image_orig, image_mask


def find_layers(sample_image):
    buffered_image_file = encode_array_to_byte_stream(sample_image)

    response = request_server(buffered_image_file,
                              LAYER_SEGMENTATOR_IP,
                              LAYER_SEGMENTATOR_HOST)

    logger.info("response find_layers received successfully!")
    image_mask = decode_buf_image_file_to_numpy(response.content, dtype=np.uint8)

    return image_mask

# ==================================== Main =================================

@app.route('/', methods=["POST", "GET"])
def upload_file():
    if request.method == "POST":
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), constrained_layout=True)
        ax.axis("off")
        legend_patches = []

        if request.files:
            requested_out_info = request.form.getlist("out-info")

            if "background" in requested_out_info and len(requested_out_info) == 1:
                image_orig, sample_mask = find_sample(request)

                ax.imshow(image_orig, cmap="gray")
                bp = highlight_background(sample_mask, ax)
                legend_patches.append(bp)
            elif "layers" in requested_out_info:
                image_orig, sample_mask = find_sample(request)
                sample_image = image_orig * sample_mask
                layer_mask = find_layers(sample_image)

                ax.imshow(image_orig, cmap="gray", label="Background")
                lp = draw_layer_contours(layer_mask, ax, "red")
                bp = highlight_background(sample_mask, ax, color="Blue")

                # appending
                legend_patches += ([bp] + [lp])

                pass

            ax.legend(handles=legend_patches)
            fig.savefig(os.path.join(APP_ROOT, OUTPUT_SAVE_PATH))
            return render_template("main.html", output_img_url=OUTPUT_SAVE_PATH)

    return render_template("main.html")
