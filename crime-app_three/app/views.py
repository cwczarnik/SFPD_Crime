"""
Contains main routes for the Prediction App
"""
from flask import render_template
from flask_wtf import Form
from wtforms import fields
from wtforms.validators import Required
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xgboost as xgb
import pickle
from StringIO import *
from . import app, MODEL
from flask import send_file

from flask import make_response
from functools import wraps, update_wrapper
from datetime import datetime

class PredictForm(Form):
    """Fields for Predict"""
    Latitude = fields.DecimalField('Latitude (-122.51 to -122.35):', places=3, validators=[Required()])
    Longitude = fields.DecimalField('Longitude (37.68 to 37.82):', places=5, validators=[Required()])
    Day = fields.DecimalField('Day (0-6) Monday-Sunday:', places=2, validators=[Required()])
    Hour = fields.DecimalField('Hour (0-23) (1AM to 12PM):', places=2, validators=[Required()])
    #Week = fields.DecimalField('Week:', places=2, validators=[Required()])
    submit = fields.SubmitField('Submit')

# def check_user_sf(X_user,Y_user):
#     X_limit_right = -122.35
#     X_limit_left = -122.51
#     Y_limit_top = 37.82
#     Y_limit_bottom = 37.68
#     if ((X_user  < X_limit_right) & (X_user> X_limit_left)& (Y_user < Y_limit_top) & (Y_user>Y_limit_bottom )):
#     # print('It looks like you\'re in San Francisco, unfortunately...')
#         return True
#     else:
#         # print('You\'re not in San Francisco!')
#         return False

def plot_probs_user(X_user, Y_user,hour_user,day_user):
    # if check_user_sf(X_user,Y_user) == True:
    array_user = pd.Series(data=[X_user, Y_user,hour_user,day_user],index=['X', 'Y','Hour','day'])
    array_user=pd.DataFrame(array_user).T
    estimator = MODEL['est']
    plt.figure()
    prob_user = estimator.predict_proba(array_user)
    objects = ('non_violent','property','violent')
    sort_obj = pd.DataFrame(prob_user[0],objects).sort_values(by=0).T.columns
    y_pos = np.arange(len(objects))
    y_prob_val = pd.DataFrame(prob_user[0],objects).sort_values(by=0).T.values
    plt.barh(y_pos, y_prob_val[0], align='center', alpha=0.8)
    plt.yticks(y_pos, sort_obj)
    plt.xlabel('Probability')
    plt.title('Crime Probability')
    plt.savefig('app/static/fig/myfigcopy.png')
    plt.show()
    # img = StringIO()
    # plt.savefig(img)
    # # img.seek(0)
    return prob_user,#send_file(img, mimetype='image/png')

@app.route('/', methods=('GET', 'POST'))
def index():
    """Index page"""
    form = PredictForm()
    my_prediction = None
    if form.validate_on_submit():
        # store the submitted values
        submitted_data = form.data
        # print(submitted_data)
        # Retrieve values from form
        Latitude= float(submitted_data['Latitude'])
        Longitude = float(submitted_data['Longitude'])
        Day = float(submitted_data['Day'])
        # Week = float(submitted_data['Week'])
        Hour = float(submitted_data['Hour'])
        # Create array from values
        my_prediction = plot_probs_user(Latitude,Longitude,Hour,Day)
        # Return only the Predicted iris species
        # plt.savefig('app/static/fig/myfig.png')
        #my_prediction = [('Murder', .4), ('Robbery', .2)]
    return render_template('index.html',form=form, prediction_vals=my_prediction)
