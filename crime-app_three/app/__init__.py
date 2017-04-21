"""
Create Application Object
"""
from flask import Flask, request
import pickle
import xgboost as xgb
import os
from sklearn.externals import joblib
app = Flask(__name__)
app.config.from_object("app.config")

# unpickle my model
mod = pickle.load( open( "models/crime_model_three.pkl", "rb" ))
MODEL = {'est': mod}
from .views import *
# Handle Bad Requests
@app.errorhandler(404)
def page_not_found(e):
    """Page Not Found"""
    return render_template('404.html'), 404
# Static cache buster
@app.url_defaults
def hashed_url_for_static_file(endpoint, values):
    if 'static' == endpoint or endpoint.endswith('.static'):
        filename = values.get('filename')
        if filename:
            if '.' in endpoint:  # has higher priority
                blueprint = endpoint.rsplit('.', 1)[0]
            else:
                blueprint = request.blueprint  # can be None too

            if blueprint:
                static_folder = app.blueprints[blueprint].static_folder
            else:
                static_folder = app.static_folder

            param_name = 'h'
            while param_name in values:
                param_name = '_' + param_name
            values[param_name] = static_file_hash(os.path.join(static_folder, filename))
            
def static_file_hash(filename):
    return int(os.stat(filename).st_mtime) # or app.config['last_build_timestamp'] or md5(filename) or etc...