from api.handlers.RandomForestHandler import RandomForestHandler

"""
This module reads parameters from the request and runs LogisticRegression.
"""

def post(payload):
	# Call the handler to process the payload
	return RandomForestHandler().post(payload)
