import os
import json
import pickle
from io import StringIO
import sys
import signal
import traceback

import flask

import numpy as np
from daal4py import decision_forest_regression_prediction

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')
param_path = os.path.join(prefix, 'model/parameters.json')

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class ScoringService(object):
	model = None                # Where we keep the model when it's loaded

	@classmethod
	def get_model(cls):
		"""Get the model object for this instance, loading it if it's not already loaded."""
		if cls.model == None:
			with open(os.path.join(model_path, 'decision-tree-regression-model.pkl'), 'rb') as inp:
				cls = pickle.load(inp)
		return cls.model

	@classmethod
	def predict(cls, input):
		"""For the input, do the predictions and return them.
		Args:
			input (a pandas dataframe): The data on which to do the predictions. There will be
				one prediction per row in the dataframe"""
		with open(param_path, "r") as pf:
			params = json.load(pf)			
			predict_algo=decision_forest_regression_prediction(fptype=params["fptype"],
																	method=params["method"],
																	distributed=(True if params["distributed"] == "True" else False))	
			dtype = (np.float64 if params["fptype"] == "double" else np.float32)
			clf = cls.get_model()
		return predict_algo.compute(input, clf)

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
	"""Determine if the container is working and healthy. In this sample container, we declare
	it healthy if we can load the model successfully."""
	health =  os.path.isfile(param_path) # You can insert a health check here

	status = 200 if health else 404
	return flask.Response(response='\n', status=status, mimetype='application/json')

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
		data = np.genfromtxt(s, delimiter=",")
	else:
		return flask.Response(response='This predictor only supports CSV data', status=415, mimetype='text/plain')

	print('Invoked with {} records'.format(data.shape[0]))

	# Do the prediction
	prediction_result = ScoringService.predict(data)

	# Convert from numpy back to CSV
	out = StringIO()
	np.savetxt(out, prediction_result.prediction)
	np.savetxt(os.path.join(prefix, 'output/predictions.txt'), prediction_result.prediction)
	result = out.getvalue()

	return flask.Response(response=result, status=200, mimetype='text/csv')
