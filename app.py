from flask import Flask, render_template, request
from ml_rest_api import api
from werkzeug.exceptions import HTTPException

app = Flask(__name__)
app.config["ERROR_404_HELP"] = False


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
def handle_http_exception(e):
    return {"message": e.description}, e.code


api.init_app(app)
app.run(debug=True)
