import pandas as pd
import os
import urllib

def post(upfile, headers):
	headers = [urllib.parse.unquote(h) for h in headers] 
	print('File {}: {}'.format(upfile.filename, headers))
	upfile.save(os.path.join('data', upfile.filename))
	return "OK"