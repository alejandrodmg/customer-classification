import logging
import json
import os
import sys
import pandas as pd
from collections import namedtuple

from sklearn.model_selection import train_test_split

from util.DataLoaders import FileDataLoader
from util.Predictors import RandomForest, LogitRegression
from util.Report import ClassificationReport

def sort_file_paths(project_name: str):
    runpath = os.path.realpath(__file__)
    rundir = runpath[:runpath.find(project_name) + len(project_name) + 1]
    os.chdir(rundir + project_name)

def load_config():
    run_configuration_file = '../resources/customer-classification.json'
    with open(run_configuration_file) as json_file:
        json_string = json_file.read()
        run_configuration = json.loads(json_string,
                                       object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
    return run_configuration

def split_X_y(data: pd.DataFrame):
    udata = data.drop_duplicates().reset_index(drop=True)
    X = udata.loc[:, udata.columns != 'great_customer_class']
    y = udata[['great_customer_class']]
    return X, y

if __name__ == '__main__':
    # Initialize logging
    logging.basicConfig(format="%(asctime)s;%(levelname)s;%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    logging.info('Starting classification program')

    # Actions: get into working directory, load project config, create dated directories
    sort_file_paths(project_name='customer-classification')
    run_configuration = load_config()

    # Import and split data
    data_loader = FileDataLoader('../data/customers.csv')
    data = data_loader.load_data()
    X, y = split_X_y(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True, random_state=0, stratify=y)

    # Create an instance of the class that evaluates the models on the test set
    report = ClassificationReport(y_test)

    # Load models
    rf_model = RandomForest()
    lr_model = LogitRegression()

    # Read user's option
    arg = sys.argv[1]
    # Execute user's option
    # (i - default) Train and test using a default configuration of ML models
    # (ii - cv) Perform stratified K-fold cross-validation on training data
    # (iii - tuning) Perform parameter tuning while cross-validating the models.
    #                NOTE: It requires setting the grid of hyper-parameters
    #                in config/hyper-params.yaml
    if arg == 'default':
        logging.info('== Random Forest ==')
        rf_model.train(X_train, y_train)
        y_pred = rf_model.predict(X_test)
        report.evaluate(y_pred)

        logging.info('== Logistic Regression ==')
        lr_model.train(X_train, y_train)
        y_pred = lr_model.predict(X_test)
        report.evaluate(y_pred)

    elif arg == 'cv':
        logging.info('== Random Forest ==')
        rf_model.cross_validate(X_train, y_train, 3, 3)

        logging.info('== Logistic Regression ==')
        lr_model.cross_validate(X_train, y_train, 3, 3)

    elif arg == 'tuning':
        logging.info('== Random Forest ==')
        rf_model.hyper_tuning(X_train, y_train, 2, 'f1')
        rf_model.train(X_train, y_train)
        y_pred = rf_model.predict(X_test)
        report.evaluate(y_pred)

        logging.info('== Logistic Regression ==')
        lr_model.hyper_tuning(X_train, y_train, 2, 'f1')
        lr_model.train(X_train, y_train)
        y_pred = lr_model.predict(X_test)
        report.evaluate(y_pred)
    else:
        logging.error('Please enter a valid argument - default, cv or tuning.')

    logging.info('Done.')
