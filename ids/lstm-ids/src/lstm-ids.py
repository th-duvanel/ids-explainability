import LSTM_model as mine
import coloredPrinting as pr
import seqMaker
import aggregate
import maxDepth as depth
import observableToSink as obs
import LSTM_FED
import pandas as pd
import numpy as np
import os
import time
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import csv
import argparse
from datetime import datetime

parser = argparse.ArgumentParser(description="Treina e avalia o modelo LSTM IDS.")
parser.add_argument(
    '--exp', 
    type=str, 
    required=True, 
    help="Nome do experimento, usado para salvar os resultados."
)
args = parser.parse_args()
experiment_name = args.exp

results_dir = os.path.join("../results", experiment_name)
model_save_path = os.path.join(results_dir, "model.h5")
csv_save_path = os.path.join(results_dir, f"{experiment_name}.csv")

os.makedirs(results_dir, exist_ok=True)


BASE_PATH = "../data"
DIS_FLOODING_TYPE = "base"

trainDataList = []
testDataList = []

def data_from_csv(path, node_quantity, interaction, dis_flooding_type):
    filepath = f"{path}/{dis_flooding_type}-{node_quantity}-{interaction}.csv"
    data = pd.read_csv(filepath, sep = ',')
    if 'Unnamed: 0' in data.columns:
        data = data.drop(['Unnamed: 0'],axis = 1)
    return data

print("Carregando dados de TREINO (5, 10, 15 nós)...")
for i in {5, 10, 15}:
    for j in range(1, 21):
        try:
            trainDataList.append(data_from_csv(BASE_PATH, i, j, DIS_FLOODING_TYPE))
        except FileNotFoundError:
            pass

print("Carregando dados de TESTE (20 nós)...")
for j in range(1, 21):
    try:
        testDataList.append(data_from_csv(BASE_PATH, 20, j, DIS_FLOODING_TYPE))
    except FileNotFoundError:
        pass


print(f"Carregamento concluído: {len(trainDataList)} arquivos de treino, {len(testDataList)} arquivos de teste.")


min_max_list = [(df.min().to_frame(name='Min').T, df.max().to_frame(name='Max').T) for df in trainDataList]

min_dfs = [min_df for min_df, max_df in min_max_list]
max_dfs = [max_df for min_df, max_df in min_max_list]

base_all_mins_df = pd.concat(min_dfs, ignore_index=True)
base_all_maxs_df = pd.concat(max_dfs, ignore_index=True)

global_max = base_all_maxs_df.max(axis = 0)
global_min = base_all_mins_df.min(axis = 0)
    
normalized_train = [df.apply(lambda x: (x - global_min[x.name]) / (global_max[x.name] - global_min[x.name])) for df in trainDataList]



normalized_test = [df.apply(lambda x: (x - global_min[x.name]) / (global_max[x.name] - global_min[x.name])) for df in testDataList]



seq_Train = [seqMaker.seq_maker(df,10) for df in normalized_train]
seq_Train = pd.concat(seq_Train, ignore_index=True)

seq_Test = [seqMaker.seq_maker(df,10) for df in normalized_test]
seq_Test = pd.concat(seq_Test, ignore_index=True)




X_Train = seq_Train.iloc[:, :-1].values
y_Train = seq_Train.iloc[:, -1].values
X_Test = seq_Test.iloc[:, :-1].values
y_Test = seq_Test.iloc[:, -1].values

X_Train = np.array(X_Train)
y_Train = np.array(y_Train)
X_Test = np.array(X_Test)
y_Test = np.array(y_Test)

X_Train = torch.tensor(X_Train, dtype=torch.float32)
y_Train = torch.tensor(y_Train, dtype = torch.long)
X_Test = torch.tensor(X_Test, dtype=torch.float32)
y_Test = torch.tensor(y_Test, dtype = torch.long)

X_Train = torch.nan_to_num(X_Train, nan=0.0)
X_Test = torch.nan_to_num(X_Test, nan=0.0)

X_Train = X_Train.view(-1, 1, 140)
train_dataset = TensorDataset(X_Train, y_Train)
X_Test = X_Test.view(-1, 1, 140)
test_dataset = TensorDataset(X_Test, y_Test)


batch_size = 2048
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

input_dim = 140
hidden_dim = 10
fc_hidden_dim = 10
num_layers = 1
output_dim = 2
lr = 0.001

ids_model = mine.LSTMClassifier(input_dim = input_dim, hidden_dim = hidden_dim, num_layers = num_layers, output_dim = output_dim,fc_hidden_dim = fc_hidden_dim, learning_rate = lr)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")
ids_model.to(device)

epochs = 30
pr.prGreen(f"Iniciando treinamento por {epochs} épocas...")
training_time = ids_model.model_train(epochs = epochs, train_loader = train_loader)
pr.prGreen("Treinamento concluído.")

pr.prGreen("Avaliando o modelo...")
metrics = ids_model.evaluate_model_ROCAUC(test_loader)

model_size_mb = ids_model.save_and_get_model_size(model_save_path)

print("\n--- Resultados da Avaliação ---")
print(f"Tempo de Treinamento: {training_time:.2f} segundos")
print(f"Tamanho do Modelo:      {model_size_mb:.4f} MB (salvo como '{model_save_path}')")
print("--- Métricas de Desempenho ---")
print(f"Acurácia:   {metrics['accuracy']:.4f}")
print(f"Precisão (Attack):   {metrics['Attack']['precision']:.4f}")
print(f"Recall (Attack):     {metrics['Attack']['recall']:.4f}")
print(f"F1-Score (Attack):   {metrics['Attack']['f1-score']:.4f}")
print(f"ROC-AUC:    {metrics['ROC-AUC']:.4f}")
print("---------------------------------")


pr.prGreen("Calculando e salvando valores SHAP...")

#shap_dir = os.path.join(results_dir, "shap")
#os.makedirs(shap_dir, exist_ok=True)
#
#try:
#    shap_values, expected_value, data_batch = ids_model.explain_with_shap(
#        train_loader, 
#        test_loader
#    )
#    
#    np.save(os.path.join(shap_dir, "shap_values.npy"), shap_values)
#    np.save(os.path.join(shap_dir, "expected_value.npy"), expected_value)
#    np.save(os.path.join(shap_dir, "data_batch.npy"), data_batch.cpu().numpy())
#    
#    print(f"Valores SHAP, valores esperados e lote de dados salvos em: {shap_dir}")
#    
#except Exception as e:
#    pr.prRed(f"Erro ao calcular ou salvar SHAP: {e}")

pr.prGreen("Calculando e salvando explicação LIME...")
lime_dir = os.path.join(results_dir, "lime")
os.makedirs(lime_dir, exist_ok=True)

try:
    # Explicar a primeira amostra (index=0) do primeiro lote de teste
    # Pedir as 20 features mais importantes
    lime_explanation = ids_model.explain_with_lime(
        train_loader,
        test_loader,
        instance_index=0,
        num_features=20
    )
    
    # Salvar a explicação como um arquivo HTML
    lime_save_path = os.path.join(lime_dir, "lime_explanation_instance_0.html")
    lime_explanation.save_to_file(lime_save_path)
    
    print(f"Explicação LIME salva em: {lime_save_path}")

except Exception as e:
    pr.prRed(f"Erro ao calcular ou salvar LIME: {e}")


results_data = {
    'timestamp': datetime.now().isoformat(),
    'dataset_name': experiment_name,
    'round': 1,
    
    'Attack_precision': metrics['Attack']['precision'],
    'Attack_recall': metrics['Attack']['recall'],
    'Attack_f1_score': metrics['Attack']['f1-score'],
    'Attack_support': metrics['Attack']['support'],
    
    'Normal_precision': metrics['Normal']['precision'],
    'Normal_recall': metrics['Normal']['recall'],
    'Normal_f1_score': metrics['Normal']['f1-score'],
    'Normal_support': metrics['Normal']['support'],
    
    'accuracy': metrics['accuracy'],
    
    'macro avg_precision': metrics['macro avg']['precision'],
    'macro avg_recall': metrics['macro avg']['recall'],
    'macro avg_f1_score': metrics['macro avg']['f1-score'],
    'macro avg_support': metrics['macro avg']['support'],
    
    'weighted avg_precision': metrics['weighted avg']['precision'],
    'weighted avg_recall': metrics['weighted avg']['recall'],
    'weighted avg_f1_score': metrics['weighted avg']['f1-score'],
    'weighted avg_support': metrics['weighted avg']['support']
}

results_df = pd.DataFrame([results_data])
results_df.to_csv(csv_save_path, index=False)

print(f"\nResultados salvos em: {csv_save_path}")