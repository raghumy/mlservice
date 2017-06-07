from api.handlers.LogisticRegressionHandler import LogisticRegressionHandler

"""
This module reads parameters from the request and runs LogisticRegression.
"""

def post(payload):
	# Call the handler to process the payload
	return LogisticRegressionHandler().post(payload)
