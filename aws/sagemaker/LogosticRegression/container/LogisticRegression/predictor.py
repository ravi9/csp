# file: predictor.py
#===============================================================================
# Copyright 2014-2018 Intel Corporation.
#
# This software and the related documents are Intel copyrighted  materials,  and
# your use of  them is  governed by the  express license  under which  they were
# provided to you (License).  Unless the License provides otherwise, you may not
# use, modify, copy, publish, distribute,  disclose or transmit this software or
# the related documents without Intel's prior written permission.
#
# This software and the related documents  are provided as  is,  with no express
# or implied  warranties,  other  than those  that are  expressly stated  in the
# License.
#===============================================================================

# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import json
import pickle
from io import StringIO
import sys
import signal
import traceback

import flask

import pandas as pd
import numpy as np
from LogisticRegression import LogisticRegression

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')
# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class ScoringService(object):
    model     = None             # Where we keep the model when it's loaded
    nFeatures = 0
    nClasses  = 0
    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            with open(os.path.join(model_path, 'daal-log-reg-train-model.pkl'), 'rb') as inp:
                cls.model = pickle.load(inp)
        return cls.model
    @classmethod
    def get_num_classes_and_features(cls):
        """Get number of class and features """
        data_loaded = np.genfromtxt(os.path.join(model_path, 'daal-log-reg-train-features-classes.csv'), delimiter=",")
        fptype = np.loadtxt(os.path.join(model_path, 'daal-log-reg-dtype.txt'),dtype=str)
        nClasses = data_loaded[0]
        nFeatures = data_loaded[1]
        return nClasses, nFeatures, fptype
    @classmethod
    def predict(cls, input, nClasses, dtype, resultsToCompute):
        """For the input, do the predictions and return them."""
        model = cls.get_model()
        logistic_regression = LogisticRegression(nClasses=nClasses, dtype = dtype, resultsToCompute=resultsToCompute)
        return logistic_regression.predict(predict_data=input, model=model)

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    if health != None:
        return flask.Response(response='model is found\n', status=200, mimetype='application/json')
    else:
        return flask.Response(response='model is not found\n', status=404, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None

    # Convert from CSV to pandas
    if flask.request.content_type == 'text/csv':
        data = flask.request.data.decode('utf-8')
        s = StringIO(data)
        data = pd.read_csv(s, header=None)
    else:
        return flask.Response(response='This predictor only supports CSV data', status=415, mimetype='text/plain')

    nClasses, nFeatures, fptype = ScoringService.get_num_classes_and_features()

    if data.shape[1] != nFeatures:
        return flask.Response(response='Number of features is incorrect. %d is required\n' % nFeatures, status=415, mimetype='text/plain')
    resultsToCompute = "computeClassesLabels|computeClassesProbabilities"
    print('Invoked with {} records'.format(data.shape[0]))
    if fptype == "float":
        fptype = "float"
    else:
        fptype = "double"
    dtype = (np.float64 if fptype == "double" else np.float32)
    # Do the prediction
    predictions = ScoringService.predict(input=np.ascontiguousarray(data.values, dtype=dtype), nClasses=nClasses,
                                         dtype=fptype, resultsToCompute=resultsToCompute)
    prediction = "".join(str(x[0])+ ' ' for x in predictions.prediction)
    probabilities = ""
    if nClasses > 2:
        i = 0
        while i < nClasses:
            probabilities += "".join(str(x[i])+ ' ' for x in predictions.probabilities) + "\n"
            i = i + 1
    else:
        probabilities = "".join(str(x[0])+ ' ' for x in predictions.probabilities) + "\n"

    result = prediction + "\n" + probabilities

    return flask.Response(response=result, status=200, mimetype='text/csv')
