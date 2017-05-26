import pandas as pd
import os
from urllib.parse import unquote, urlparse
import json
from model.logisticregression import *
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

data_dir = 'data'

def post(filename, headers=None, penalty=None):
	headers = [unquote(h) for h in headers]
	print('FileName: {}, Headers: {} Penalty: {}'.format(filename, headers, penalty))

	# Validate the filename
	if filename is None:
		raise Exception('FileName is empty')

	# Check if filename is a URL
	url=urlparse(filename)
	if not url.netloc:
		# This must be a local file
		filename = os.path.join(data_dir, filename)

	# We have the fields. Now run the model
	logger.info('Calling LogisticRegression for file {}'.format(filename))
	m = LogisticRegression(filename=filename, columns=headers, penalty=penalty)
	m.run()

	logger.info('LogisticRegression completed with accuracy {}/{}'.format(m.train_accuracy, m.test_accuracy))

	return {'fileName': filename, 'headers': headers, 'penalty': penalty, 
		'train_accuracy': m.train_accuracy, 
		'test_accuracy': m.test_accuracy}