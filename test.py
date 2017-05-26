import os
from model.logisticregression import *

data_dir = 'data'

if __name__ == '__main__':
	c = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 
		'Alcalinity of ash', 'Magnesium', 'Total phenols', 
		'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 
		'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
	m = LogisticRegression(filename=os.path.join(data_dir, 'wine.data'), columns=c, penalty='l1', scale_type='std')
	m.run()
	print(m.X_train.shape)
	print(m.X_train[:3, :3])
	print(m.X_scale_train[:3, :3])
	print(m.train_accuracy)
	print(m.test_accuracy)