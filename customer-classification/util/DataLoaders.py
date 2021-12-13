import logging
from abc import ABC, abstractmethod
import os.path
import pandas as pd

class AbstractDataLoader(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def load_data(self):
        NotImplementedError()

class FileDataLoader(AbstractDataLoader):

    # Initialization
    def __init__(self, filename: str):
        super().__init__()
        logging.info('Initializing Data Loading')
        self.filename = filename

    # Load data from file and return data
    def load_data(self):
        # Check file exists
        logging.info('Checking file exists.')
        if not os.path.isfile(self.filename):
            raise FileNotFoundError('File does not exist')
        else:
            logging.info('Found file: ' + self.filename)
        # Load data
        logging.info('Loading data using pandas')
        data = pd.read_csv(self.filename)
        return data
