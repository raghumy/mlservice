from api.handlers.BaseHandler import BaseHandler
from model.logisticregression import LogisticRegression
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
This module reads parameters from the request and runs LogisticRegression.
"""


class LogisticRegressionHandler(BaseHandler):

    def __init__(self):
        self.penalty = None
        self.C = None

    def post(self, payload):
        """
        Post request handler for LogisticRegression

        Parameters:

        penalty: Type of penalty to use - L1 or L2

        C: Control bias and variance
        """

        super(LogisticRegressionHandler, self).post(payload)

        self.penalty = payload['penalty'] if 'penalty' in payload else None
        self.C = payload['C'] if 'C' in payload else None

        # We have the fields. Now run the model
        logger.info('Calling LogisticRegression for file {}'.format(self.filename))
        m = LogisticRegression(filename=self.filename, columns=self.headers, has_header=self.hasHeader,
                               class_label=self.classLabel, class_label_column=self.classLabelColumn,
                               penalty=self.penalty, C=self.C)
        m.run()

        logger.info('LogisticRegression completed with accuracy {}/{}'.format(m.train_accuracy, m.test_accuracy))

        return {'fileName': self.filename, 'headers': self.headers, 'penalty': m.penalty, 'C': m.C,
                'train_accuracy': m.train_accuracy,
                'test_accuracy': m.test_accuracy}