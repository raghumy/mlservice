import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Model:
	"""
	Base class for all models. The base class performs the following functions:
	Reading CSV files
	Splitting the data into test and train_test_split
	Scaling the data
	"""
	def __init__(self, filename, columns=None, test_size=0.3, scale_type='std', *args, **kwargs):
		self.filename = filename
		self.columns = columns
		self.test_size = test_size
		self.scale_type = scale_type

	def get_data(self):
		self.df = pd.read_csv(self.filename)
		if len(self.columns) > 0:
			self.df.columns = self.columns

	def test_train_split(self):
		X, y = self.df.iloc[:, 1:].values, self.df.iloc[:, 0].values

		self.X_train, self.X_test, self.y_train, self.y_test = \
        	train_test_split(X, y, test_size=self.test_size, random_state=0)

	def scale(self):
		if self.scale_type == 'std':
			stdsc = StandardScaler()
			self.X_scale_train = stdsc.fit_transform(self.X_train)
			self.X_scale_test = stdsc.transform(self.X_test)
		else:
			self.X_scale_train = self.X_train
			self.X_scale_test = self.X_test

	def run(self):
		# Run standard processes to get the data
		self.get_data()
		self.test_train_split()
		self.scale()


