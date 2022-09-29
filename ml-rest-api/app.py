from flask import Flask
#from pathlib import Path
#import os
#import shutil
#from werkzeug.middleware.proxy_fix import ProxyFix

#models_dir = Path('./models')
#if models_dir.exists() and models_dir.is_dir():
#    shutil.rmtree(models_dir)
#Path('./models').mkdir(parents=True, exist_ok=True)

from ml_rest_api import api

app = Flask(__name__)
#app.wsgi_app = ProxyFix(app.wsgi_app)

api.init_app(app)

app.run(debug=True)