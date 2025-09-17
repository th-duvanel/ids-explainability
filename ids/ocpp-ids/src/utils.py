import json
from sklearn.metrics import classification_report

def load_config(path):
    with open(path, "r") as json_file:
        config = json.load(json_file)
    return config

def evaluation_metrics(y_true, y_pred, class_names):
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    flat_report = {}
    for key, value in report.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat_report[f'{key}_{sub_key}'] = sub_value
        else:
            flat_report[key] = value
    
    final_report = {k.replace('-', '_'): v for k, v in flat_report.items()}
    
    return final_report