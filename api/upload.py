import pandas as pd
import os
import urllib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
This module reads parameters from the request and saves the file.
"""

def post(upfile):
	# TODO: Add validation of the input
	logger.info('FileName: {}'.format(upfile.filename))
	upfile.save(os.path.join('data', upfile.filename))
	return "OK"