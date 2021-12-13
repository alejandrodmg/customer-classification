import unittest
import importlib.util
import pandas as pd

def load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class TestDataLoader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_loader = DataLoaders.FileDataLoader('../data/customers.csv')

    def test_create_loader(self):
        # Make sure the object is a class
        self.assertTrue(isinstance(self.data_loader, DataLoaders.FileDataLoader))
        # Check the filename is correct
        self.assertEqual(self.data_loader.filename, '../data/customers.csv')

    def test_load_data(self):
        data = self.data_loader.load_data()
        # Check the object is not empty
        self.assertIsNotNone(data)
        # Ensure the data is a pandas DataFrame
        self.assertTrue(isinstance(data, pd.DataFrame))
        # Check the total number of observations
        self.assertEqual(data.shape[0], 13599)
        # Make sure it loads all the features
        self.assertEqual(data.shape[1], 15)


if __name__ == '__main__':
    # Load custom module
    DataLoaders = load_module('DataLoaders', '../customer-classification/util/DataLoaders.py')
    # Run tests
    unittest.main()
