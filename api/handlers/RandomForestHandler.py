from api.handlers.BaseHandler import BaseHandler
from model.randomforest import RandomForest
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
This module reads parameters from the request and runs Random Forest.
"""


class RandomForestHandler(BaseHandler):

    def __init__(self):
        self.n_estimators = None

    def post(self, payload):
        """
        Post request handler for Random Forest

        Parameters:

        penalty: Type of penalty to use - L1 or L2

        C: Control bias and variance
        """

        super(RandomForestHandler, self).post(payload)

        self.n_estimators = payload['n_estimators'] if 'n_estimators' in payload else None

        # We have the fields. Now run the model
        logger.info('Calling RandomForest for file {}'.format(self.filename))
        m = RandomForest(filename=self.filename, columns=self.headers, has_header=self.hasHeader,
                         class_label=self.classLabel, class_label_column=self.classLabelColumn,
                         n_estimators=self.n_estimators)
        m.run()

        logger.info('RandomForest completed with accuracy {}/{}'.format(m.train_accuracy, m.test_accuracy))

        return {'fileName': self.filename, 'headers': self.headers, 'n_estimators': m.n_estimators,
                'train_accuracy': m.train_accuracy,
                'test_accuracy': m.test_accuracy}