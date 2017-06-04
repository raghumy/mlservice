from model.model import Model
import sklearn.linear_model as lr
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LogisticRegression(Model):
	"""
	Class to perform LogisticRegression
	"""
	def __init__(self, *args, **kwargs):
		"""
		Initialize local parameters and then call the base class for the remaining
		"""
		if 'penalty' in kwargs and kwargs['penalty'] is not None:
			self.penalty = kwargs['penalty']
		else:
			self.penalty = 'l2'

		# Penalty for mis-classification
		# Increasing the value of C increases the bias and lowers the variance of the model
		if 'C' in kwargs and kwargs['C'] is not None:
			self.C = kwargs['C']
		else:
			self.C = 0.1

		logger.info('penalty: {}, C: {}'.format(self.penalty, self.C))

		super(LogisticRegression,self).__init__(*args, **kwargs)

	def run(self):
		# Run standard processes to get the data
		super(LogisticRegression,self).run()

		# Run logistic regression
		m = lr.LogisticRegression(penalty=self.penalty, C=self.C)
		m.fit(self.X_scale_train, self.y_train)
		self.train_accuracy = m.score(self.X_scale_train, self.y_train)
		self.test_accuracy = m.score(self.X_scale_test, self.y_test)
