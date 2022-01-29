from app import app, views

HOST = views.set_ip("HOST")

if __name__ == "__main__":
    app.run(host=HOST, debug=False)