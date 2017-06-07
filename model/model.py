import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Model:
    """
    Base class for all models. The base class performs the following functions:
    Reading CSV files
    Splitting the data into test and train_test_split
    Scaling the data
    Encoding Labels
    """

    def __init__(self, filename, columns=None, test_size=0.3, scale_type='std', has_header=True, class_label=None,
                 class_label_column=0, *args, **kwargs):
        self.filename = filename
        self.columns = columns
        self.test_size = test_size
        self.scale_type = scale_type
        self.has_header = has_header
        self.class_label = class_label
        self.class_label_column = class_label_column

    def get_data(self):
        """
        Retrieve CSV data from the location specified in the filename.
        """
        logger.debug('Has Header: {}'.format(self.has_header))
        if self.has_header:
            self.df = pd.read_csv(self.filename)
        else:
            self.df = pd.read_csv(self.filename, header=None)

        logger.info('Columns: {}'.format(self.columns));

        if self.columns and len(self.columns) > 0:
            self.df.columns = self.columns

    def encode_data(self):
        """
        Encode data for columns of type object
        """
        le = LabelEncoder()
        for col in [c for c in self.df.columns if self.df[c].dtype == 'object']:
            # Replace the column with values from the LabelEncoder
            logger.info('Encoding column {}:{}'.format(col, self.df[col].dtype))
            self.df[col] = le.fit_transform(self.df[col])

    def test_train_split(self):
        """
        Split the data into train and test

        TODO: Expose this parameter
        """
        logger.info('Columns: {}'.format(self.df.columns))

        # If the columns are not of type object and only has integers
        # then make sure the index passed is an integer
        if self.df.columns.dtype != 'object':
            if self.class_label is not None and self.class_label.isdigit():
                self.class_label_column = int(self.class_label)
                self.class_label = None
            else:
                raise Exception('Class label {} is incompatible with the column type {}'.format(self.class_label,
                                                                                                self.df.columns.dtype))

        logger.info('Class Label Column: {}, Class Label: {}'.format(self.class_label_column, self.class_label))
        if self.class_label:
            X, y = self.df.drop(self.class_label, axis=1).values, self.df[self.class_label].values
        else:
            # Make a new list that contains all the columns
            l = list(range(len(self.df.columns)))
            l.remove(self.class_label_column)

            # Remove the class label column
            X, y = self.df.iloc[:, l].values, self.df.iloc[:, self.class_label_column].values

        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=self.test_size, random_state=0)

    def scale(self):
        """
        Scale the data using StandardScaler

        TODO: Expose as parameter and support other types of Scaling
        """
        if self.scale_type == 'std':
            stdsc = StandardScaler()
            self.X_scale_train = stdsc.fit_transform(self.X_train)
            self.X_scale_test = stdsc.transform(self.X_test)
        else:
            self.X_scale_train = self.X_train
            self.X_scale_test = self.X_test

    def run(self):
        # Run standard processes to get the data
        self.get_data()
        self.encode_data()
        self.test_train_split()
        self.scale()
