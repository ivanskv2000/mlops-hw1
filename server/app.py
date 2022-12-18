from flask import Flask, render_template, request
from ml_rest_api import api, controller
from werkzeug.exceptions import HTTPException
import os
import psycopg2

api.add_namespace(controller.api)

app = Flask(__name__)
app.config["ERROR_404_HELP"] = False
app.config["PROPAGATE_EXCEPTIONS"] = False


@app.route("/ml_rest_api")
def home():
    base_url = request.base_url
    return render_template(
        "landing.html",
        title=api.title,
        description=api.description,
        version=api.version,
        contact=api.contact,
        doc_url=base_url + "/doc",
    )


@app.errorhandler(HTTPException)
def handle_http_exception(error):
    return {"message": error.description}, error.code


api.init_app(app)

if __name__ == "__main__":
    app.run(debug=True)
