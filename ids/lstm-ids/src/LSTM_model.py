#!/usr/bin/env python3

import time
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.profiler import profile, record_function, ProfilerActivity
import shap
import lime
import lime.lime_tabular

from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score, \
    recall_score, precision_score, accuracy_score, classification_report



class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, fc_hidden_dim, learning_rate):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # --- CORREÇÃO ADICIONADA AQUI ---
        # Salva o input_dim para ser usado nos métodos de explicação
        self.input_dim = input_dim 
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, output_dim)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate)
        self.loss_train = []
        # self.loss_test = [] # Removido das versões anteriores


    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc1(out[:, -1, :])
        out = torch.relu(out)
        out = self.fc2(out)
        # Softmax é aplicado nos métodos de avaliação, não aqui
        return out


    def model_train(self, epochs, train_loader):
        start_time = time.time()
        
        for epoch in range(epochs):
            self.train()
            total_loss = 0
            for i, (x_batch, y_batch) in enumerate(train_loader):
                
                device = next(self.parameters()).device
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                
                self.optimizer.zero_grad()
                y_prediction = self(x_batch) # Logits
                loss = self.criterion(y_prediction,y_batch)
                self.loss_train.append(loss.item())
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
        
        end_time = time.time()
        training_time = end_time - start_time
        return training_time
        

    def eval_model(self, x):
        self.eval()
        device = next(self.parameters()).device
        x = x.to(device)
        with torch.no_grad():
            outputs = self(x)
            outputs = torch.softmax(outputs, dim=1) # Softmax aqui
            
        return outputs
    

    def check_model_nans(self, loader):
        device = next(self.parameters()).device
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = self(inputs)
            if torch.isnan(outputs).any():
                print("ERRO: NaNs detectados na saída do modelo")

    
    def evaluate_model_ROCAUC(self, test_loader):
        self.eval()
        with torch.no_grad():
            all_logits = []
            all_targets = []
            device = next(self.parameters()).device
            
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                logits = self(inputs)
                all_logits.append(logits)
                all_targets.append(targets)
            
            all_logits = torch.cat(all_logits, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            all_probs = torch.softmax(all_logits, dim=1)
            probabilities_np = all_probs[:,1].cpu().detach().numpy()
            targets_np = all_targets.cpu().detach().numpy()
            
            model_fpr, model_tpr, threshold = roc_curve(targets_np, probabilities_np)
            roc_auc = auc(model_fpr, model_tpr)
            
            predicted_labels_np = torch.argmax(all_probs, dim=1).cpu().detach().numpy()
            
            report = classification_report(targets_np, 
                                           predicted_labels_np, 
                                           target_names=['Normal', 'Attack'], 
                                           output_dict=True, 
                                           zero_division=0)
            
            metrics = {
                "ROC-AUC": roc_auc,
                "Normal": report['Normal'],
                "Attack": report['Attack'],
                "accuracy": report['accuracy'],
                "macro avg": report['macro avg'],
                "weighted avg": report['weighted avg']
            }
        
        return metrics

    def save_and_get_model_size(self, save_path="ids_model.pth"):
        torch.save(self.state_dict(), save_path)
        
        try:
            model_size_bytes = os.path.getsize(save_path)
            model_size_mb = model_size_bytes / (1024 * 1024)
            return model_size_mb
        except FileNotFoundError:
            print(f"Erro: Não foi possível encontrar o modelo salvo em {save_path} para medir o tamanho.")
            return 0


    def explain_with_shap(self, background_loader, data_loader, n_summary=100, n_explain=50):
        print(f"Calculando SHAP com KernelExplainer...")
        print(f"Atenção: Explicando apenas {n_explain} amostras usando {n_summary} amostras de background.")
        print("Isso pode ser MUITO lento.")

        self.eval() 
        device = next(self.parameters()).device

        def model_fn_wrapper(numpy_data_2d):
            with torch.no_grad():
                tensor_data_2d = torch.from_numpy(numpy_data_2d).float().to(device)
                # Usa self.input_dim
                tensor_data_3d = tensor_data_2d.view(-1, 1, self.input_dim) 
                logits = self(tensor_data_3d)
                probs = torch.softmax(logits, dim=1)
                return probs.cpu().numpy()

        background_data_list = []
        for x_batch, _ in background_loader:
            # Usa self.input_dim
            background_data_list.append(x_batch.view(-1, self.input_dim))
            if sum(len(b) for b in background_data_list) >= n_summary:
                break
        
        background_tensor_2d = torch.cat(background_data_list, dim=0)
        
        indices = np.random.choice(background_tensor_2d.shape[0], n_summary, replace=False)
        background_summary_np = background_tensor_2d[indices].cpu().numpy()
        print(f"Tamanho do background summary: {background_summary_np.shape}")

        data_to_explain_3d = next(iter(data_loader))[0] 
        
        if data_to_explain_3d.shape[0] > n_explain:
             data_to_explain_3d = data_to_explain_3d[:n_explain]
        
        # Usa self.input_dim
        data_to_explain_2d_np = data_to_explain_3d.view(-1, self.input_dim).cpu().numpy()
        print(f"Tamanho dos dados a explicar: {data_to_explain_2d_np.shape}")

        explainer = shap.KernelExplainer(model_fn_wrapper, background_summary_np)
        
        shap_values = explainer.shap_values(data_to_explain_2d_np, l1_reg="auto")
        
        print("Cálculo SHAP concluído.")
        
        data_batch_tensor = data_to_explain_3d.cpu()

        # Retorna o dicionário
        return {
            "shap_values": shap_values[1],
            "expected_value": explainer.expected_value[1],
            "data_batch": data_batch_tensor
        }


    # --- MÉTODO LIME ---
    
    def explain_with_lime(self, background_loader, data_loader, instance_index=0, num_features=20, n_summary=100):
        print(f"Calculando LIME para a amostra {instance_index} do primeiro lote de teste...")
        print(f"Usando {n_summary} amostras de background...")

        self.eval()
        device = next(self.parameters()).device

        def lime_predictor_fn(numpy_data_2d):
            with torch.no_grad():
                tensor_data_2d = torch.from_numpy(numpy_data_2d).float().to(device)
                # Usa self.input_dim
                tensor_data_3d = tensor_data_2d.view(-1, 1, self.input_dim)
                logits = self(tensor_data_3d)
                probs = torch.softmax(logits, dim=1)
                return probs.cpu().numpy()

        background_data_list = []
        for x_batch, _ in background_loader:
            # Usa self.input_dim
            background_data_list.append(x_batch.view(-1, self.input_dim))
            if sum(len(b) for b in background_data_list) >= n_summary:
                break
        background_tensor_2d = torch.cat(background_data_list, dim=0)
        indices = np.random.choice(background_tensor_2d.shape[0], n_summary, replace=False)
        background_summary_np = background_tensor_2d[indices].cpu().numpy()

        # Usa self.input_dim
        feature_names = [f'feature_{i}' for i in range(self.input_dim)]
        class_names = ['Normal', 'Attack']

        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=background_summary_np,
            feature_names=feature_names,
            class_names=class_names,
            mode='classification',
            discretize_continuous=False
        )

        data_batch_3d, _ = next(iter(data_loader))
        
        if instance_index >= data_batch_3d.shape[0]:
            print(f"Erro: instance_index {instance_index} fora do alcance do lote (tamanho {data_batch_3d.shape[0]}). Usando índice 0.")
            instance_index = 0
            
        instance_to_explain_3d = data_batch_3d[instance_index]
        
        # Usa self.input_dim
        instance_to_explain_1d_np = instance_to_explain_3d.view(self.input_dim).cpu().numpy()

        print(f"Gerando explicação para a amostra {instance_index}...")
        explanation = explainer.explain_instance(
            data_row=instance_to_explain_1d_np,
            predict_fn=lime_predictor_fn,
            num_features=num_features,
            top_labels=2
        )
        
        print("Cálculo LIME concluído.")
        return explanation