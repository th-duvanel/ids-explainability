import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import json

def evaluation_metrics(y_true, classes, predicted_test):
    '''Calculate evaluation metrics and return them as a dictionary.'''

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

    # True Positive Rate - TPR (Recall or Sensitivity)
    # Handle division by zero for cases where a class has no positive instances
    with np.errstate(divide='ignore', invalid='ignore'):
        TPR = np.true_divide(TP, (TP + FN))
        TPR[np.isnan(TPR)] = 0 # Replace NaN with 0
    mean_tpr = np.mean(TPR)
    print(f'TPR: {mean_tpr}')

    # False Positive Rate - FPR
    with np.errstate(divide='ignore', invalid='ignore'):
        FPR = np.true_divide(FP, (FP + TN))
        FPR[np.isnan(FPR)] = 0 # Replace NaN with 0
    mean_fpr = np.mean(FPR)
    print(f'FPR: {mean_fpr}')

    # F1 Score
    f1 = f1_score(y_true, classes, average='weighted')
    print(f'F1 score: {f1}')

    # Return metrics as a dictionary
    return {
        'accuracy': accuracy,
        'TPR': mean_tpr,
        'FPR': mean_fpr,
        'f1_score': f1
    }

def load_config(config_file_path):
    with open(config_file_path, 'r') as config_file:
        config = json.load(config_file)
    return config