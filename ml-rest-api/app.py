from flask import Flask, render_template, request
from ml_rest_api import api

app = Flask(__name__)


@app.route('/ml_rest_api')
def home():
    base_url = request.base_url
    return render_template(
        'landing.html',
        title=api.title,
        description=api.description,
        version=api.version,
        contact=api.contact,
        doc_url=base_url + '/doc'
        )


api.init_app(app)
app.run(debug=True)
