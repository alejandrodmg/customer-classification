
## Description

This is an into example to class imbalance in customer classification problems. It mainly focuses on using preprocessing techniques to solve the problem. In particular it covers:

- Feature Selector
- Dummy Transformer
- Log Transformer
- MinMax Scaler
- KNN Imputer
- SMOTE
- Random UnderSampler

Random Forest and Logistic Regression are used to solve a binary classification problem. The code supports K-fold cross validation and hyper-parameter tuning.

## Main Program Execution

Please open a clean terminal window, set your directory to customer-classification/customer_classification and run the file `customer_classification.py` followed by one of these arguments:

- `default`: Train, test and evaluate Random Forest and Logistic Regression using a default configuration.
- `cv`: Perform K-fold cross validation on training data using Random Forest and Logistic Regression.
- `tuning`: Perform parameter tuning for Random Forest and Logistic Regression while cross-validating the models, it requires setting the hyper-parameters in config/hyper-params.yaml. 

Running the code returns a nicely formatted classification report.

Examples:

```
~/$ python3 customer_classification.py default
~/$ python3 customer_classification.py cv
~/$ python3 customer_classification.py tuning
```

## Unit Testing

The code in `test_DataLoaders.py` located in folder customer-classification/tests performs a number of tests to the DataLoader class. To run the tests please move to the directory and execute the following command in a clean terminal window:

```
~/$ python3 test_DataLoaders.py
```