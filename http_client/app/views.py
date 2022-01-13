from app import app
from flask import render_template, request, redirect
import requests as request_http

@app.route('/', methods=["POST", "GET"])
def upload_file():
    if request.method == "POST":
        if request.files:

            image = request.files["image"]
            request_http.post("http://127.0.0.1:5001/upload-raw-image", files={"image" : image})

            return redirect(request.url, 200)
    
    return render_template("main.html")