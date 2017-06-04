import pandas as pd
import os
from urllib.parse import unquote, urlparse
import json
from model.svm import SVM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

data_dir = 'data'

"""
This module reads parameters from the request and runs SVM.
"""

def post(filename, headers=None, hasHeader=False, classLabel=None, classLabelColumn=0, kernel=None, C=None):
	"""
	Post request handler for SVM

	Parameters:

	filename: Name of the file or fully defined path to the filename

	headers: Column headings to use for this data

	hasHeader: Does the file have header data

	classLabel: Column to use as class class_label

	classLabelColumn: Index of the column to use as class_label

	kernel: Type of kernel
	"""
	if headers:
		headers = [unquote(h).strip() for h in headers]
	logger.info('FileName: {}, Headers: {} Kernel: {}, C: {}'.format(filename, headers, kernel, C))

	# Validate the filename
	if filename is None:
		raise Exception('FileName is empty')

	if classLabel is not None:
		classLabel = classLabel.strip()

	# Check if filename is a URL
	# If it isn't, expect it in the local directory
	url=urlparse(filename)
	if not url.netloc:
		# This must be a local file
		filename = os.path.join(data_dir, filename)

	# We have the fields. Now run the model
	logger.info('Calling SVM for file {}'.format(filename))
	m = SVM(filename=filename, columns=headers, has_header=hasHeader, class_label=classLabel, class_label_column=classLabelColumn, kernel=kernel, C=C)
	m.run()

	logger.info('SVM completed with accuracy {}/{}'.format(m.train_accuracy, m.test_accuracy))

	return {'fileName': filename, 'headers': headers, 'kernel': m.kernel, 'C': m.C, 
		'train_accuracy': m.train_accuracy, 
		'test_accuracy': m.test_accuracy}