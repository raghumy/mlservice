import os
from urllib.parse import unquote, urlparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

data_dir = 'data'

class BaseHandler:
    def __init__(self):
        self.filename = None
        self.headers = None
        self.classLabel = None
        self.hasHeader = None
        self.classLabelColumn = None

    """
    This module reads parameters from the request and runs LogisticRegression.
    """

    def post(self, payload):
        """
        Post request handler for LogisticRegression

        Parameters:

        filename: Name of the file or fully defined path to the filename

        headers: Column headings to use for this data

        hasHeader: Does the file have header data

        classLabel: Column to use as class class_label

        classLabelColumn: Index of the column to use as class_label
        """

        logger.info('Payload: {}'.format(payload))

        # Expand parameters
        self.filename = payload['filename'] if 'filename' in payload else None
        self.headers = payload['headers'] if 'headers' in payload else None
        self.classLabel = payload['classLabel'] if 'classLabel' in payload else None
        self.hasHeader = payload['hasHeader'] if 'hasHeader' in payload else None
        self.classLabelColumn = None

        if self.headers:
            self.headers = [unquote(h).strip() for h in self.headers]
        logger.info('Headers: {}'.format(self.headers))

        # Validate the filename
        if self.filename is None:
            raise Exception('FileName is empty')

        # Strip the class label
        if self.classLabel is not None:
            self.classLabel = self.classLabel.strip()

        # Check if filename is a URL
        # If it isn't, expect it in the local directory
        url=urlparse(self.filename)
        if not url.netloc:
            # This must be a local file
            self.filename = os.path.join(data_dir, self.filename)
