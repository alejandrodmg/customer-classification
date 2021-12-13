--------------------
Exploratory Analysis
--------------------

The jupyter notebook customer_classification.ipynb contains a step-by-step analysis of the dataset and the preprocessing techniques applied prior to training ML models. 

---------
Unit Test
---------

The code in test_DataLoaders.py located in folder customer-classification/tests performs a number of tests to the DataLoader class. To run the tests please move to the directory and execute the following command in a clean terminal window:

~/$ python3 test_DataLoaders.py

----------------------
Main Program Execution
----------------------

Please open a clean terminal window, set your directory to customer-classification/customer_classification and run the file customer_classification.py followed by one of these arguments:

(i) ‘default’: Train, test and evaluate Random Forest and Logistic Regression using a default configuration.
(ii) ‘cv’: Perform K-fold cross validation on training data using Random Forest and Logistic Regression.
(iii) ‘tuning’: Perform parameter tuning for Random Forest and Logistic Regression while cross-validating the models, it requires setting the hyper-parameters in config/hyper-params.yaml. 

=> Examples:

~/$ python3 customer_classification.py default
~/$ python3 customer_classification.py cv
~/$ python3 customer_classification.py tuning