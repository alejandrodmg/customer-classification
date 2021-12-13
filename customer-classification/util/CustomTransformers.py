import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(self, drop: list):
        super().__init__()
        self.drop = drop

    def fit(self, X, y=None):
        # Stateless transformer
        return self

    def transform(self, X, y=None):
        # Drop a given set of features
        X_ = X.drop(self.drop, axis=1)
        return X_

class DummyTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, features: list):
        super().__init__()
        self.features = features

    def fit(self, X, y=None):
        # Stateless transformer
        return self

    def transform(self, X, y=None):
        # Copy dataset to not modify the original
        X_ = X.copy(deep=True)
        # Replace soldier and estate agent by other
        X_['occupation'].replace({'soldier': 'other',
                                  'estate_agent': 'other'}, inplace=True)
        # Generate dummy variables
        X_ = pd.get_dummies(X_, columns=self.features)
        return X_

class LogTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, feature: str):
        super().__init__()
        self.feature = feature

    def fit(self, X, y=None):
        # Stateless transformer
        return self

    def transform(self, X, y=None):
        # Copy dataset to not modify the original
        X_ = X.copy(deep=True)
        # Log transform a feature
        X_.loc[:, 'log_' + self.feature] = np.log(1 + X_[self.feature].values)
        # Drop the original feature
        X_ = X_.drop(self.feature, axis=1)
        return X_
