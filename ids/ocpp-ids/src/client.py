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

from utils import  def evaluate(self, parameters, config):
        '''
        Evaluates the model and runs XAI methods ONLY on the final round.
        '''
        self.current_round += 1
        
        self.model.set_weights(parameters)
        print(f"\nEvaluation - {self.dataset_name}, Round {self.current_round}")
        loss, accuracy = self.model.evaluate(self.X_test, self.Y_test, verbose=0)
        
        predicted_test = self.model.predict(self.X_test, verbose=0)
        
        if len(self.label_encoder.classes_) == 2:
            predicted_classes = (predicted_test > 0.5).astype("int32")
        else:
            predicted_classes = np.argmax(predicted_test, axis=1)
        
        eval_metrics = evaluation_metrics(self.Y_test, predicted_classes, self.class_names)
        self.eval_metrics_lst.append(eval_metrics)

        self.save_metrics_to_csv(eval_metrics)

        is_last_round = self.current_round == self.conf.get("num_rounds")

        if is_last_round:
            if self.xai_method in ['shap', 'both']:
                print(f"\n--- FINAL ROUND: Calculating SHAP values... ---")
                self.calculate_and_save_shap(os.path.join('results', self.dataset_name, self.experiment_type, 'shap'))
            
            if self.xai_method in ['lime', 'both']:
                print(f"\n--- FINAL ROUND: Calculating LIME explanations... ---")
                self.calculate_and_save_lime(os.path.join('results', self.dataset_name, self.experiment_type, 'lime'))
        # ----------------------------------------
        
        return loss, len(self.X_test), {"accuracy": accuracy}
trics, load_config
from dnn_model import create_model

# --- ADDED: Imports for saving metrics ---
import csv
from datetime import datetime
import json

import re
from collections import defaultdict


class ExplainableClient(fl.client.NumPyClient):
    '''
    ## MODIFIED: Renamed to ExplainableClient
    Flower client class with integrated SHAP and LIME explanations,
    controlled by command-line arguments.
    '''
    def __init__(self, path_to_config, xai_method):
        self.conf = load_config(path_to_config)
        self.xai_method = xai_method

        self.experiment_type = self.conf.get('experiment_type')
        self.dataset_name = self.conf.get('dataset_name')

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

    def save_metrics_to_csv(self, metrics_dict):
        '''Appends evaluation metrics for the current round to a client-specific CSV file.'''
        output_dir = os.path.join('results', self.dataset_name, self.experiment_type)
        os.makedirs(output_dir, exist_ok=True)

        filename = os.path.join(output_dir, f'{self.experiment_type}.csv')

        row_data = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'dataset_name': self.dataset_name,
            'round': self.current_round,
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
        self.data_train = pd.read_csv('data/' + self.dataset_name + '/Train.csv')
        self.data_test = pd.read_csv('data/' + self.dataset_name + '/Test.csv')

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
        '''
        Evaluates the model and runs XAI methods ONLY on the final round.
        '''
        self.current_round += 1
        
        self.model.set_weights(parameters)
        print(f"\nEvaluation - {self.dataset_name}, Round {self.current_round}")
        loss, accuracy = self.model.evaluate(self.X_test, self.Y_test, verbose=0)
        
        predicted_test = self.model.predict(self.X_test, verbose=0)
        
        if len(self.label_encoder.classes_) == 2:
            predicted_classes = (predicted_test > 0.5).astype("int32")
        else:
            predicted_classes = np.argmax(predicted_test, axis=1)
        
        eval_metrics = evaluation_metrics(self.Y_test, predicted_classes, self.class_names)
        self.eval_metrics_lst.append(eval_metrics)

        self.save_metrics_to_csv(eval_metrics)

        is_last_round = self.current_round == self.conf.get("num_rounds")

        if is_last_round:
            if self.xai_method in ['shap', 'both']:
                print(f"\n--- FINAL ROUND: Calculating SHAP values... ---")
                self.calculate_and_save_shap(os.path.join('results', self.dataset_name, self.experiment_type, 'shap'))
            
            if self.xai_method in ['lime', 'both']:
                print(f"\n--- FINAL ROUND: Calculating LIME explanations... ---")
                self.calculate_and_save_lime(os.path.join('results', self.dataset_name, self.experiment_type, 'lime'))
        # ----------------------------------------
        
        return loss, len(self.X_test), {"accuracy": accuracy}

    def calculate_and_save_shap(self, output_path):
        '''Calculates and saves SHAP values, data samples, and feature names.'''
        os.makedirs(output_path, exist_ok=True)

        background_data = self.X_train[np.random.choice(self.X_train.shape[0], 100, replace=False)]
        explainer = shap.KernelExplainer(self.model.predict, background_data)
        
        sample_indices = np.random.choice(self.X_test.shape[0], min(10, self.X_test.shape[0]), replace=False)
        data_to_explain = self.X_test[sample_indices]
        shap_values = explainer.shap_values(data_to_explain)

        filename_shap = f'shap_values.npy'
        np.save(os.path.join(output_path, filename_shap), shap_values)
        print(f"SHAP values saved to {os.path.join(output_path, filename_shap)}")

        filename_data = f'shap_data.npy'
        np.save(os.path.join(output_path, filename_data), data_to_explain)
        print(f"SHAP data samples saved to {os.path.join(output_path, filename_data)}")

        filename_features = f'shap_features.json'
        with open(os.path.join(output_path, filename_features), 'w') as f:
            json.dump(self.feature_names, f)
        print(f"SHAP feature names saved to {os.path.join(output_path, filename_features)}")

        filename_classes = f'shap_class_names.json'
        with open(os.path.join(output_path, filename_classes), 'w') as f:
            json.dump(self.class_names, f)
        print(f"SHAP class names saved to {os.path.join(output_path, filename_classes)}")

    def calculate_and_save_lime(self, output_path):
        '''
        Calculates LIME explanations for many samples, aggregates feature importances,
        and saves the aggregated list to a JSON file with robust feature name parsing.
        '''
        os.makedirs(output_path, exist_ok=True)

        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.X_train,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode='classification'
        )

        def lime_predict_wrapper(data):
            prob_attack = self.model.predict(data, verbose=0)
            prob_normal = 1 - prob_attack
            return np.hstack((prob_normal, prob_attack))

        num_samples_to_explain = min(100, self.X_test.shape[0])
        sample_indices = np.random.choice(self.X_test.shape[0], num_samples_to_explain, replace=False)

        print(f"Generating and aggregating LIME explanations for {num_samples_to_explain} samples...")
        
        feature_importance_sum = defaultdict(float)

        for i, instance_index in enumerate(sample_indices):
            instance_to_explain = self.X_test[instance_index]
            
            explanation = explainer.explain_instance(
                instance_to_explain,
                lime_predict_wrapper,
                num_features=len(self.feature_names),
                labels=(0, 1)
            )

            lime_list = explanation.as_list()
            
            for feature_string, weight in lime_list:
                # --- LÓGICA DE EXTRAÇÃO CORRIGIDA E MAIS ROBUSTA ---
                base_feature_name = None
                # Procura qual dos nossos nomes de feature conhecidos está na string do LIME
                for fname in self.feature_names:
                    if fname in feature_string:
                        base_feature_name = fname
                        break
                
                # Se um nome de feature válido for encontrado, agrega sua importância
                if base_feature_name:
                    feature_importance_sum[base_feature_name] += abs(weight)
                # --------------------------------------------------------

        sorted_importances = dict(sorted(feature_importance_sum.items(), key=lambda item: item[1]))

        filename_json = f'lime_feature_importance.json'
        output_file_path = os.path.join(output_path, filename_json)
        with open(output_file_path, 'w') as f:
            json.dump(sorted_importances, f, indent=4)
        
        print(f"\nAggregated LIME feature importance ranking saved to {output_file_path}")

def gen_client(args):
    '''Generates and returns the client instance.'''
    client = ExplainableClient(args.config_path, args.xai)
    return client

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

    path = os.path.join('results', client.dataset_name, client.experiment_type)
    if path:
        os.makedirs(path, exist_ok=True)

        full_path = os.path.join(path, 'model.h5')
        client.model.save(full_path)
        print(f"\nFinal model saved to {full_path}")
    
if __name__ == "__main__":
    main()
