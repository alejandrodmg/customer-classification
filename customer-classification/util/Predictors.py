import logging
import yaml
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

from util.CustomTransformers import FeatureSelector, DummyTransformer, LogTransformer

class Model(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        NotImplementedError()

    @abstractmethod
    def cross_validate(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        NotImplementedError()

    @abstractmethod
    def hyper_tuning(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        NotImplementedError()

    @abstractmethod
    def predict(self, X_test: pd.DataFrame):
        NotImplementedError()

class MLWorkflow():

    def __init__(self):
        self.pipeline = Pipeline(
            steps= [
                ('feature_selector', FeatureSelector([
                    'user_id', 'sex', 'race'])),
                ('one_hot', DummyTransformer([
                    'occupation', 'marital-status', 'workclass'])),
                ('log', LogTransformer('salary')),
                ('scaler', MinMaxScaler(feature_range=(0, 1))),
                ('imputer', KNNImputer(n_neighbors=10, weights='distance')),
                ('smote', SMOTE(sampling_strategy=0.25, k_neighbors=10, random_state=0)),
                ('under', RandomUnderSampler(sampling_strategy=0.40, random_state=0))
            ])

class RandomForest(Model):

    def __init__(self):
        super().__init__()
        logging.info('Initializing model')
        self.classifier = RandomForestClassifier(random_state=0, n_estimators=100)
        self.model = MLWorkflow()
        self.model.pipeline.steps.append(('model', self.classifier))
        self.best_params = None

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        logging.info('Training model')
        # Train on best parameters, if possible
        if self.best_params:
            params_dict = {k.replace('model__', ''): v for k, v in self.best_params.items()}
            self.model.pipeline['model'].set_params(**params_dict)
        self.model.pipeline.fit(X_train, y_train.values.ravel())
        # Calculate feature importance
        col_names = self.model.pipeline[0:3].transform(X_train).columns
        importance = sorted(zip(np.round(self.model.pipeline['model'].feature_importances_,3), col_names), reverse=True)
        print('='*5, 'Feature Importance', '='*5)
        print(pd.DataFrame(importance, columns=['score', 'feature']).to_markdown(tablefmt="grid"))

    def cross_validate(self, X_train: pd.DataFrame, y_train: pd.DataFrame, n_splits: int, n_repeats: int):
        logging.info('Cross-validating model')
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)
        scores = []
        iters = 0
        total_iters = n_splits * n_repeats
        # Train and test the model on k folds of the training data
        for train_index, test_index in cv.split(X_train, y_train):
            X_train_cv, X_test_cv = X_train.iloc[train_index], X_train.iloc[test_index]
            y_train_cv, y_test_cv = y_train.iloc[train_index], y_train.iloc[test_index]
            self.model.pipeline.fit(X_train_cv, y_train_cv.values.ravel())
            y_pred = self.model.pipeline.predict(X_test_cv)
            scores.append(f1_score(y_test_cv, y_pred))
            iters+=1
            logging.info('[CV {}/{}] => Test F1-score {:.2f}'.format(iters, total_iters, scores[-1]))
        logging.info('Average CV F1-score {:.2f}'.format(np.mean(scores)))

    def hyper_tuning(self, X_train: pd.DataFrame, y_train: pd.DataFrame, n_splits: int, evaluate: str):
        logging.info('Hyper-parameter tuning')
        # Load hyper parameters from yaml file
        with open('config/hyper-params.yaml') as file:
            param_grid = yaml.load(file, Loader=yaml.FullLoader)
        params = GridSearchCV(
            self.model.pipeline, param_grid['RandomForest'],
            cv=n_splits, scoring=evaluate, verbose=3).fit(X_train, y_train.values.ravel())
        # Store best parameters
        self.best_params = params.best_params_
        logging.info('Best hyper-parameters: '.format(params.best_params_))
        logging.info('CV F1-score: {:.2f}'.format(params.best_score_))

    def predict(self, X_test: pd.DataFrame):
        logging.info('Doing predictions')
        # Run the pipeline on unseen data
        y_pred = self.model.pipeline.predict(X_test)
        return y_pred

class LogitRegression(Model):

    def __init__(self):
        super().__init__()
        logging.info('Initializing model')
        self.classifier = LogisticRegression(max_iter=3000)
        self.model = MLWorkflow()
        self.model.pipeline.steps.append(('model', self.classifier))
        self.best_params = None

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        logging.info('Training model')
        # Train on best parameters, if possible
        if self.best_params:
            params_dict = {k.replace('model__', ''): v for k, v in self.best_params.items()}
            self.model.pipeline['model'].set_params(**params_dict)
        self.model.pipeline.fit(X_train, y_train.values.ravel())
        # Get coefficients
        col_names = self.model.pipeline[0:3].transform(X_train).columns
        importance = sorted(zip(np.round(self.model.pipeline['model'].coef_[0],3), col_names), reverse=True)
        print('='*5, 'Coefficients', '='*5)
        print(pd.DataFrame(importance, columns=['coef', 'feature']).to_markdown(tablefmt="grid"))

    def cross_validate(self, X_train: pd.DataFrame, y_train: pd.DataFrame, n_splits: int, n_repeats: int):
        logging.info('Cross-validating model')
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)
        scores = []
        iters = 0
        total_iters = n_splits * n_repeats
        # Train and test the model on k folds of the training data
        for train_index, test_index in cv.split(X_train, y_train):
            X_train_cv, X_test_cv = X_train.iloc[train_index], X_train.iloc[test_index]
            y_train_cv, y_test_cv = y_train.iloc[train_index], y_train.iloc[test_index]
            self.model.pipeline.fit(X_train_cv, y_train_cv.values.ravel())
            y_pred = self.model.pipeline.predict(X_test_cv)
            scores.append(f1_score(y_test_cv, y_pred))
            iters+=1
            logging.info('[CV {}/{}] => Test F1-score {:.2f}'.format(iters, total_iters, scores[-1]))
        logging.info('Average CV F1-score {:.2f}'.format(np.mean(scores)))

    def hyper_tuning(self, X_train: pd.DataFrame, y_train: pd.DataFrame, n_splits: int, evaluate: str):
        logging.info('Hyper-parameter tuning')
        # Load hyper parameters from yaml file
        with open('config/hyper-params.yaml') as file:
            param_grid = yaml.load(file, Loader=yaml.FullLoader)
        params = GridSearchCV(
            self.model.pipeline, param_grid['LogitRegression'],
            cv=n_splits, scoring=evaluate, verbose=3).fit(X_train, y_train.values.ravel())
        # Store best parameters
        self.best_params = params.best_params_
        logging.info('Best hyper-parameters:', params.best_params_)
        logging.info('CV F1-score {:.2f}:'.format(params.best_score_))

    def predict(self, X_test: pd.DataFrame):
        logging.info('Doing predictions')
        # Run the pipeline on unseen data
        y_pred = self.model.pipeline.predict(X_test)
        return y_pred
