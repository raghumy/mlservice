from api.handlers.SVMHandler import SVMHandler

"""
This module reads parameters from the request and runs LogisticRegression.
"""

def post(payload):
	# Call the handler to process the payload
	return SVMHandler().post(payload)
