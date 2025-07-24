import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import json
import pandas as pd

def evaluation_metrics(y_true, classes, predicted_test):
    '''Calculate evaluation metrics '''

    accuracy = accuracy_score(y_true, classes)
    print(f'Accuracy: {accuracy}')

    cnf_matrix = confusion_matrix(y_true, classes)

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    # true positive rate - TPR
    TPR = TP/(TP+FN)
    print(f'TPR: {np.mean(TPR)}')

    # false positive rate - FPR
    FPR = FP/(FP+TN)
    print(f'FPR: {np.mean(FPR)}')

    # F1 Score
    f1 = f1_score(y_true, classes, average='weighted')
    print(f'F1 score: {f1}')

    return (accuracy, np.mean(TPR), np.mean(FPR), f1)
    
def load_config(config_file_path):
    with open(config_file_path, 'r') as config_file:
        config = json.load(config_file)
    return config