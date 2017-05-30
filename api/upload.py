import pandas as pd
import os
import urllib

def post(upfile):
	print('FileName: {}'.format(upfile.filename))
	upfile.save(os.path.join('data', upfile.filename))
	return "OK"