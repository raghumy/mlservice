import pandas as pd
import os
from urllib.parse import unquote, urlparse
import json
from model.randomforest import RandomForest
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

data_dir = 'data'

def post(filename, headers=None, hasHeader=True, classLabel=None, classLabelColumn=0, n_estimators=10):
	if headers:
		headers = [unquote(h).strip() for h in headers]
	logger.info('FileName: {}, Headers: {} n_estimators: {}'.format(filename, headers, n_estimators))

	# Validate the filename
	if filename is None:
		raise Exception('FileName is empty')

	# Check if filename is a URL
	url=urlparse(filename)
	if not url.netloc:
		# This must be a local file
		filename = os.path.join(data_dir, filename)

	# We have the fields. Now run the model
	logger.info('Calling RandomForest for file {}'.format(filename))
	m = RandomForest(filename=filename, columns=headers, n_estimators=n_estimators, has_header=hasHeader, class_label=classLabel, class_label_column=classLabelColumn)
	m.run()

	logger.info('RandomForest completed with accuracy {}/{}'.format(m.train_accuracy, m.test_accuracy))

	return {'fileName': filename, 'headers': headers, 'n_estimators': n_estimators, 
		'train_accuracy': m.train_accuracy, 
		'test_accuracy': m.test_accuracy}