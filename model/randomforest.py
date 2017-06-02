from model.model import Model
from sklearn.ensemble import RandomForestClassifier

class RandomForest(Model):
	"""
	Class to perform RandomForest
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

		super(RandomForest,self).__init__(*args, **kwargs)

	def run(self):
		# Run standard processes to get the data
		super(RandomForest,self).run()

		forest = RandomForestClassifier(criterion='entropy',
       		n_estimators=10,
            random_state=1,
            n_jobs=2)
		forest.fit(self.X_scale_train, self.y_train)
		self.train_accuracy = forest.score(self.X_scale_train, self.y_train)
		self.test_accuracy = forest.score(self.X_scale_test, self.y_test)
