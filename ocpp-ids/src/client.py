import pandas as pd
import numpy as np
import flwr as fl
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
import argparse
import os

# --- Imports for Explainable AI (XAI) ---
import shap
import lime
import lime.lime_tabular

from utils import evaluation_metrics, load_config
from dnn_model import create_model

# --- ADDED: Imports for saving metrics ---
import csv
from datetime import datetime
import json


class ExplainableClient(fl.client.NumPyClient):
    '''
    ## MODIFIED: Renamed to ExplainableClient
    Flower client class with integrated SHAP and LIME explanations,
    controlled by command-line arguments.
    '''
    def __init__(self, path_to_config, xai_method):
        self.conf = load_config(path_to_config)
        self.xai_method = xai_method

        self.experiment_type = self.conf.get('type')
        
        self.client_number, self.dataset_name = self._parse_config_identifiers()

        self.load_dataset()
        self.X_train, self.Y_train, self.feature_names = preprocess(self.data_train, self.conf['features_to_drop'])
        self.X_test, self.Y_test, _ = preprocess(self.data_test, self.conf['features_to_drop'])
        
        # Encode labels and normalize data before creating the model
        self.encode_labels()
        self.normalize_data()
        
        self.eval_metrics_lst = []
        
        num_classes = len(self.label_encoder.classes_)
        if num_classes < 2:
            raise ValueError(f"Error: Only one class found after preprocessing: {self.class_names}. "
                             "Please check the 'normal_label_name' in the preprocess function "
                             "and ensure it matches the label in your CSV file (e.g., 'Benign' vs 'BENIGN').")

        self.model = create_model(self.X_train.shape[1], num_classes)
        
        opt = tf.keras.optimizers.Adam(learning_rate=self.conf['learning_rate'], beta_1=0.99, beta_2=0.999, epsilon=1e-08)
        
        if num_classes == 2:
            self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        else:
            self.model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        
        self.current_round = 0

    ## ADDED: Helper method to parse the combined client ID
    def _parse_config_identifiers(self):
        '''Decodes the combined client_id from config into a client number and dataset name.'''
        combined_id = self.conf['client_id']
        client_number = combined_id // 10
        dataset_code = combined_id % 10
        
        if dataset_code == 1:
            dataset_name = 'cicflowmeter'
        elif dataset_code == 2:
            dataset_name = 'ocppflowmeter'
        else:
            dataset_name = 'unknown_dataset'
            
        return client_number, dataset_name

    ## MODIFIED: Now uses the parsed client_number and dataset_name
    def save_metrics_to_csv(self, metrics_dict):
        '''Appends evaluation metrics for the current round to a client-specific CSV file.'''
        output_dir = self.conf.get('results_output_path')
        os.makedirs(output_dir, exist_ok=True)

        filename = os.path.join(output_dir, f'{self.experiment_type}_metrics_{self.client_number}.csv')

        row_data = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'client_id': self.client_number,
            'dataset_name': self.dataset_name,
            'round': self.current_round,
            'xai_method_used': self.xai_method,
            **metrics_dict
        }
        
        header = list(row_data.keys())
        
        file_exists = os.path.isfile(filename)
        with open(filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_data)

        print(f"Metrics for round {self.current_round} saved to {filename}")
    
    def load_dataset(self):
        '''Loads the dataset from the path specified in the config.'''
        self.data_train = pd.read_csv(self.conf['dataset_path'] + '/Train.csv')
        self.data_test = pd.read_csv(self.conf['dataset_path'] + '/Test.csv')
    
    def normalize_data(self):
        '''Normalizes data with StandardScaler.'''
        scaler = StandardScaler()
        self.scaler = scaler
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test) 

    def encode_labels(self):
        '''Encodes the labels using LabelEncoder.'''
        enc_Y = LabelEncoder()
        self.Y_train = enc_Y.fit_transform(self.Y_train.to_numpy().reshape(-1))
        self.Y_test = enc_Y.transform(self.Y_test.to_numpy().reshape(-1))
        self.label_encoder = enc_Y
        self.class_names = self.label_encoder.classes_.astype(str).tolist()
    
    def get_parameters(self, config):
        '''Returns the current local model weights.'''
        return self.model.get_weights()

    def fit(self, parameters, config):
        '''Performs local model training.'''
        self.model.set_weights(parameters)
        self.epochs = config['local_epochs']
        self.model.fit(self.X_train, self.Y_train, epochs=self.epochs, batch_size=config['batch_size'], verbose=0)
        return self.model.get_weights(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        '''Evaluates the model and optionally runs XAI methods.'''
        self.current_round += 1
        
        self.model.set_weights(parameters)
        print(f"\nEvaluation - Client {self.client_number} ({self.dataset_name}), Round {self.current_round}")
        loss, accuracy = self.model.evaluate(self.X_test, self.Y_test, verbose=0)
        
        predicted_test = self.model.predict(self.X_test, verbose=0)
        
        if len(self.label_encoder.classes_) == 2:
            predicted_classes = (predicted_test > 0.5).astype("int32")
        else:
            predicted_classes = np.argmax(predicted_test, axis=1)
        
        eval_metrics = evaluation_metrics(self.Y_test, predicted_classes, self.class_names)
        self.eval_metrics_lst.append(eval_metrics)

        self.save_metrics_to_csv(eval_metrics)

        if self.xai_method in ['shap', 'both']:
            print("\nCalculating SHAP values...")
            self.calculate_and_save_shap(self.conf.get('shap_output_path', './results/shap'))
        
        if self.xai_method in ['lime', 'both']:
            print("\nCalculating LIME explanations...")
            self.calculate_and_save_lime(self.conf.get('lime_output_path', './results/lime'))
        
        return loss, len(self.X_test), {"accuracy": accuracy}

    def calculate_and_save_shap(self, output_path):
        '''Calculates and saves SHAP values, data samples, and feature names.'''
        os.makedirs(output_path, exist_ok=True)

        background_data = self.X_train[np.random.choice(self.X_train.shape[0], 100, replace=False)]
        explainer = shap.KernelExplainer(self.model.predict, background_data)
        
        sample_indices = np.random.choice(self.X_test.shape[0], min(10, self.X_test.shape[0]), replace=False)
        data_to_explain = self.X_test[sample_indices]
        shap_values = explainer.shap_values(data_to_explain)

        filename_shap = f'shap_values_client_{self.client_number}.npy'
        np.save(os.path.join(output_path, filename_shap), shap_values)
        print(f"SHAP values saved to {os.path.join(output_path, filename_shap)}")

        filename_data = f'shap_data_client_{self.client_number}.npy'
        np.save(os.path.join(output_path, filename_data), data_to_explain)
        print(f"SHAP data samples saved to {os.path.join(output_path, filename_data)}")

        filename_features = f'shap_features_client_{self.client_number}.json'
        with open(os.path.join(output_path, filename_features), 'w') as f:
            json.dump(self.feature_names, f)
        print(f"SHAP feature names saved to {os.path.join(output_path, filename_features)}")

        filename_classes = f'shap_class_names_client_{self.client_number}.json'
        with open(os.path.join(output_path, filename_classes), 'w') as f:
            json.dump(self.class_names, f)
        print(f"SHAP class names saved to {os.path.join(output_path, filename_classes)}")

    def calculate_and_save_lime(self, output_path):
        '''Calculates and saves LIME explanations for a subset of the test data.'''
        os.makedirs(output_path, exist_ok=True)

        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.X_train,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode='classification'
        )

        def lime_predict_wrapper(data):
            prob_attack = self.model.predict(data)
            prob_normal = 1 - prob_attack
            return np.hstack((prob_normal, prob_attack))

        num_samples_to_explain = min(5, self.X_test.shape[0])
        sample_indices = np.random.choice(self.X_test.shape[0], num_samples_to_explain, replace=False)

        print(f"Generating LIME explanations for {num_samples_to_explain} samples...")
        for i, instance_index in enumerate(sample_indices):
            instance_to_explain = self.X_test[instance_index]
            
            explanation = explainer.explain_instance(
                instance_to_explain,
                lime_predict_wrapper,
                num_features=len(self.feature_names),
                labels=(0, 1)
            )

            filename = f'lime_explanation_client_{self.client_number}_instance_{i}.html'
            explanation.save_to_file(os.path.join(output_path, filename))
            print(f"LIME explanation for instance {i} saved to {os.path.join(output_path, filename)}")

def gen_client(args):
    '''Generates and returns the client instance.'''
    client = ExplainableClient(args.config_path, args.xai)
    return client

## MODIFIED: Corrected the normal_label_name to match the data
def preprocess(data, feats_to_drop):
    '''
    ## MODIFIED: Cleans data, converts labels to binary, and returns feature names.
    '''
    data = data.dropna(axis=0, how='any')
    label_name = 'Label' if 'Label' in data.columns else 'label'
    
    print(f"DEBUG: Unique labels found in raw data: {data[label_name].unique()}")
    
    Y = data[[label_name]].copy()

    normal_label_name = 'normal' 
    
    Y[label_name] = np.where(Y[label_name].str.strip() == normal_label_name, 'Normal', 'Attack')
    # ---------------------------------------------
    
    X_df = data.drop(feats_to_drop + [label_name], axis=1)

    feature_names = X_df.columns.tolist()
    X = X_df.to_numpy()
    
    return X, Y, feature_names

def main():
    parser = argparse.ArgumentParser(description='Federated Learning Client with XAI')
    parser.add_argument('config_path', type=str, help='Path to the configuration file of the client.')
    
    parser.add_argument(
        '--xai',
        type=str,
        default='none',
        choices=['shap', 'lime', 'both', 'none'],
        help='Specify the explainability method to run (shap, lime, both, or none). Default: none.'
    )
    
    args = parser.parse_args()
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    client = gen_client(args)
    
    fl.client.start_numpy_client(
        server_address=client.conf.get("server_address", "127.0.0.1:8080"),
        client=client,
    )
    
    path = client.conf.get('saved_model_path')
    if path:
        os.makedirs(path, exist_ok=True)

        experiment_type = client.conf.get('type', 'unknown')
        client_number = client.client_number
        dataset_name = client.dataset_name
        model_filename = f"model_{dataset_name}_{experiment_type}_client_{client_number}.h5"

        full_path = os.path.join(path, model_filename)
        client.model.save(full_path)
        print(f"\nFinal model saved to {full_path}")
    
if __name__ == "__main__":
    main()
