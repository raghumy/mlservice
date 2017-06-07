from model.model import Model
from sklearn.ensemble import RandomForestClassifier
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RandomForest(Model):
    """
    Class to perform RandomForest
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize local parameters and then call the base class for the remaining
        """
        if 'n_estimators' in kwargs:
            self.n_estimators = kwargs['n_estimators']
        else:
            self.n_estimators = 10

        logger.info('n_estimators: {}'.format(self.n_estimators))

        super(RandomForest, self).__init__(*args, **kwargs)

    def run(self):
        # Run standard processes to get the data
        super(RandomForest, self).run()

        # Run Random Forest
        forest = RandomForestClassifier(criterion='entropy',
                                        n_estimators=self.n_estimators,
                                        random_state=1,
                                        n_jobs=2)

        forest.fit(self.X_scale_train, self.y_train)
        self.train_accuracy = forest.score(self.X_scale_train, self.y_train)
        self.test_accuracy = forest.score(self.X_scale_test, self.y_test)
