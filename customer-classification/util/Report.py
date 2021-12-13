import logging
import pandas as pd
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix, f1_score,\
                            precision_score, recall_score, roc_auc_score

class ClassificationReport():

    def __init__(self, y_test: pd.DataFrame):
        self.y_test = y_test

    def evaluate(self, y_pred: np.array):
        # Print a full classification report
        print('='*5, 'Evaluation Report', '='*5)
        dict_report = classification_report(self.y_test, y_pred, output_dict=True)
        print(pd.DataFrame(dict_report)[['0', '1', 'macro avg', 'weighted avg']].to_markdown(tablefmt="grid"))
        print(pd.DataFrame(confusion_matrix(self.y_test, y_pred)).to_markdown(tablefmt="grid"))
        print('Accuracy: {:.2f}'.format(dict_report['accuracy']))
        print('F1 Score: {:.2f}'.format(f1_score(self.y_test, y_pred)))
        print('Precision Score: {:.2f}'.format(precision_score(self.y_test, y_pred)))
        print('Recall Score: {:.2f}'.format(recall_score(self.y_test, y_pred)))
        print('ROC AUC Score: {:.2f}'.format(roc_auc_score(self.y_test, y_pred)))
        print('='*30)
