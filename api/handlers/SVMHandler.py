from api.handlers.BaseHandler import BaseHandler
from model.svm import SVM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
This module reads parameters from the request and runs SVM.
"""


class SVMHandler(BaseHandler):

    def __init__(self):
        self.kernel = None
        self.C = None

    def post(self, payload):
        """
        Post request handler for SVM

        Parameters:

        penalty: Type of penalty to use - L1 or L2

        C: Control bias and variance
        """

        super(SVMHandler, self).post(payload)

        self.kernel = payload['kernel'] if 'kernel' in payload else None
        self.C = payload['C'] if 'C' in payload else None

        # We have the fields. Now run the model
        logger.info('Calling SVM for file {}'.format(self.filename))
        m = SVM(filename=self.filename, columns=self.headers, has_header=self.hasHeader,
                class_label=self.classLabel, class_label_column=self.classLabelColumn,
                kernel=self.kernel, C=self.C)
        m.run()

        logger.info('SVM completed with accuracy {}/{}'.format(m.train_accuracy, m.test_accuracy))

        return {'fileName': self.filename, 'headers': self.headers, 'kernel': m.kernel, 'C': m.C,
                'train_accuracy': m.train_accuracy,
                'test_accuracy': m.test_accuracy}