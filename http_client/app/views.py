from app import app
from flask import render_template, request, redirect
import requests as request_http

@app.route('/', methods=["POST", "GET"])
def upload_file():
    if request.method == "POST":
        if request.files:

            image = request.files["image"]

            image.save(image.filename)

            people = request.form.getlist('people')
            print(people)

            return redirect(request.url)
    
    return render_template("main.html")