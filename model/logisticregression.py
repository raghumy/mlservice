from model.model import Model
import sklearn.linear_model as lr

class LogisticRegression(Model):
	"""
	Class to perform LogisticRegression
	"""
	def __init__(self, *args, **kwargs):
		if 'penalty' in kwargs:
			self.penalty = kwargs['penalty']
		else:
			self.penalty = 'l2'
		if 'C' in kwargs:
			self.C = kwargs['C']
		else:
			self.C = 0.1

		super(LogisticRegression,self).__init__(*args, **kwargs)

	def run(self):
		# Run standard processes to get the data
		super(LogisticRegression,self).run()

		m = lr.LogisticRegression(penalty=self.penalty, C=self.C)
		m.fit(self.X_scale_train, self.y_train)
		self.train_accuracy = m.score(self.X_scale_train, self.y_train)
		self.test_accuracy = m.score(self.X_scale_test, self.y_test)
