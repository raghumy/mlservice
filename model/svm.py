from model.model import Model
from sklearn.svm import SVC
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SVM(Model):
    """
    Class to perform svm
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize local parameters and then call the base class for the remaining
        """
        if 'kernel' in kwargs:
            self.kernel = kwargs['kernel']
        else:
            self.kernel = 'linear'

        # Penalty for mis-classification
        # Increasing the value of C increases the bias and lowers the variance of the model
        if 'C' in kwargs and kwargs['C'] is not None:
            self.C = kwargs['C']
        else:
            self.C = 1.0

        logger.info('kernel: {}, C: {}'.format(self.kernel, self.C))

        super(SVM, self).__init__(*args, **kwargs)

    def run(self):
        # Run standard processes to get the data
        super(SVM, self).run()

        # Run SVM regression
        svm = SVC(kernel=self.kernel, C=self.C, random_state=0)
        svm.fit(self.X_scale_train, self.y_train)
        self.train_accuracy = svm.score(self.X_scale_train, self.y_train)
        self.test_accuracy = svm.score(self.X_scale_test, self.y_test)
