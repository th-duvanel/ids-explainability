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

# --- (Assuming utils and dnn_model are in the same directory) ---
from utils import evaluation_metrics, load_config
from dnn_model import create_model


class ExplainableClient(fl.client.NumPyClient):
    '''
    ## MODIFIED: Renamed to ExplainableClient
    Flower client class with integrated SHAP and LIME explanations,
    controlled by command-line arguments.
    '''
    def __init__(self, path_to_config, xai_method):
        self.conf = load_config(path_to_config)
        ## ADDED: Store the XAI method from command-line arguments.
        self.xai_method = xai_method

        self.load_dataset()
        ## MODIFIED: preprocess now returns feature_names for LIME/SHAP.
        self.X_train, self.Y_train, self.feature_names = preprocess(self.data_train, self.conf['features_to_drop'])
        self.X_test, self.Y_test, _ = preprocess(self.data_test, self.conf['features_to_drop'])
        
        self.eval_metrics_lst = []
        self.model = create_model(self.X_train.shape[1], len(self.Y_train.value_counts()))
        opt = tf.keras.optimizers.Adam(learning_rate=self.conf['learning_rate'], beta_1=0.99, beta_2=0.999, epsilon=1e-08)
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
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
        ## ADDED: Store class names for human-readable LIME/SHAP outputs.
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
        self.model.set_weights(parameters)
        print("\nEvaluation:")
        loss, accuracy = self.model.evaluate(self.X_test, self.Y_test, verbose=0)
        
        predicted_test = self.model.predict(self.X_test, verbose=0)
        predicted_classes = np.argmax(predicted_test, axis=1)
        eval_metrics = evaluation_metrics(self.Y_test, predicted_classes, predicted_test)
        self.eval_metrics_lst.append(eval_metrics)

        ## --- ADDED: Centralized XAI method control ---
        # Checks the command-line argument to decide which XAI method to run.
        if self.xai_method in ['shap', 'both']:
            print("\nCalculating SHAP values...")
            self.calculate_and_save_shap(self.conf.get('shap_output_path', './results/shap'))
        
        if self.xai_method in ['lime', 'both']:
            print("\nCalculating LIME explanations...")
            self.calculate_and_save_lime(self.conf.get('lime_output_path', './results/lime'))
        
        return loss, len(self.X_test), {"accuracy": accuracy}

    ## --- ADDED: SHAP calculation method ---
    def calculate_and_save_shap(self, output_path):
        '''Calculates and saves SHAP values for a subset of the test data.'''
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        client_id = self.conf['client_id']
        # Use a small subset of the training data as the background dataset for performance.
        background_data = self.X_train[np.random.choice(self.X_train.shape[0], 100, replace=False)]
        
        # KernelExplainer is a good model-agnostic choice for any model.
        explainer = shap.KernelExplainer(self.model.predict, background_data)
        
        # Explain a few random instances from the test set.
        sample_indices = np.random.choice(self.X_test.shape[0], min(10, self.X_test.shape[0]), replace=False)
        data_to_explain = self.X_test[sample_indices]
        shap_values = explainer.shap_values(data_to_explain)

        # Save SHAP values to a .npy file for later analysis.
        filename = f'shap_values_client_{client_id}.npy'
        np.save(os.path.join(output_path, filename), shap_values)
        print(f"SHAP values saved to {os.path.join(output_path, filename)}")

    ## --- ADDED: LIME calculation method ---
    def calculate_and_save_lime(self, output_path):
        '''Calculates and saves LIME explanations for a subset of the test data.'''
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        client_id = self.conf['client_id']
        # LIME's explainer requires training data statistics, feature and class names.
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.X_train,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode='classification'
        )

        # Explain a few random instances from the test set.
        num_samples_to_explain = min(5, self.X_test.shape[0])
        sample_indices = np.random.choice(self.X_test.shape[0], num_samples_to_explain, replace=False)

        print(f"Generating LIME explanations for {num_samples_to_explain} samples...")
        for i, instance_index in enumerate(sample_indices):
            instance_to_explain = self.X_test[instance_index]
            
            # Generate the explanation for the instance.
            explanation = explainer.explain_instance(
                instance_to_explain,
                self.model.predict,
                num_features=len(self.feature_names)
            )

            # Save the explanation as a user-friendly HTML file.
            filename = f'lime_explanation_client_{client_id}_instance_{i}.html'
            explanation.save_to_file(os.path.join(output_path, filename))
            print(f"LIME explanation for instance {i} saved to {os.path.join(output_path, filename)}")

def gen_client(args):
    '''Generates and returns the client instance.'''
    ## MODIFIED: Pass the xai_method argument to the client's constructor.
    client = ExplainableClient(args.config_path, args.xai)
    client.encode_labels()
    client.normalize_data()
    return client

def preprocess(data, feats_to_drop):
    '''
    ## MODIFIED: Cleans data and also returns the list of feature names.
    '''
    data = data.dropna(axis=0, how='any')
    label_name = 'Label' if 'Label' in data.columns else 'label'
    
    Y = data[[label_name]]
    # Drop features and the label to get the final feature set.
    X_df = data.drop(feats_to_drop + [label_name], axis=1)

    feature_names = X_df.columns.tolist()
    X = X_df.to_numpy()
    
    return X, Y, feature_names

def main():
    parser = argparse.ArgumentParser(description='Federated Learning Client with XAI')
    parser.add_argument('config_path', type=str, help='Path to the configuration file of the client.')
    
    ## ADDED: Command-line argument to select the XAI method.
    parser.add_argument(
        '--xai',
        type=str,
        default='none',
        choices=['shap', 'lime', 'both', 'none'],
        help='Specify the explainability method to run (shap, lime, both, or none). Default: none.'
    )
    
    args = parser.parse_args()
    
    tf.config.set_visible_devices([], 'GPU')
    
    client = gen_client(args)
    
    fl.client.start_numpy_client(
        server_address=client.conf['server_address'],
        client=client,
    )
    
    # Save the final model.
    path = client.conf['saved_model_path']
    if not os.path.exists(path):
        os.makedirs(path)
    full_path = os.path.join(path, 'model.h5')
    client.model.save(full_path)
    print(f"\nFinal model saved to {full_path}")
    
if __name__ == "__main__":
    main()