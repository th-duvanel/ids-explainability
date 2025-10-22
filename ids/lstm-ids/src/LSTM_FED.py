#!/usr/bin/env python3


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.profiler import profile, record_function, ProfilerActivity
from sklearn.metrics import roc_auc_score, roc_curve, auc
import coloredPrinting as pr
from collections import OrderedDict



def get_device():
   return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking = True)




class LSTMModel(nn.Module):
   def __init__(self, input_dim, hidden_dim, output_dim, num_layers, fc_hidden_dim, learning_rate, device):
      super(LSTMModel,self).__init__()
      self.device = device 
      self.hidden_dim = hidden_dim
      self.num_layers = num_layers
      self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True).to(self.device)
      self.fc1 = nn.Linear(hidden_dim, fc_hidden_dim).to(self.device)
      self.fc2 = nn.Linear(fc_hidden_dim, output_dim).to(self.device)
      self.criterion = nn.CrossEntropyLoss().to(self.device)
      self.optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate)
      self.loss_train = []
      self.loss_test = []
      # track layers 
      self.track_layers = {'lstm1': self.lstm, 'fc1': self.fc1, 'fc2': self.fc2}
      
      
   def forward(self, x):
      # Ensure the input data is moved to the correct device (GPU/CPU)
      h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
      c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
      
      out, _ = self.lstm(x, (h0, c0))
      out = self.fc1(out[:, -1, :])
      out = torch.relu(out)  # Activation function for the hidden layer
      out = self.fc2(out)
      out = torch.softmax(out,dim = 1)

      return out

   

   def get_parameters(self):
      return {name: param.data for name, param in self.named_parameters()}

   def set_parameters(self, parameters):
      for name, param in self.named_parameters():
         param.data = parameters[name].clone()

   def evaluate_model_ROCAUC(self, test_loader):
      # Set the model to evaluation mode
      self.eval()
      with torch.no_grad():
         all_outputs = []
         all_targets = []
            
         for inputs, targets in test_loader:
            outputs = self(inputs)
            all_outputs.append(outputs)
            all_targets.append(targets)
            
         # Concatenate all batches to get complete set
         all_outputs = torch.cat(all_outputs, dim=0)
         all_targets = torch.cat(all_targets, dim=0)
            
         # Convert outputs to probabilities using softmax and get probabilities for class 1
         #probabilities = torch.softmax(all_outputs, dim=1)[:, 1]
         probabilities = all_outputs[:,1]
         #all_targets = all_targets[:,1]
            
         # Calculate ROC AUC score
         #roc_auc = roc_auc_score(all_targets.cpu().numpy(), probabilities.cpu().numpy())
         probabilities = probabilities.detach().numpy()
         all_targets = all_targets.detach().numpy()
         model_fpr, model_tpr, threshold = roc_curve(all_targets, probabilities)
         roc_auc = auc(model_fpr, model_tpr)
            
        
      return roc_auc

   def train_client(self, epochs, training_data_fcn):
      """ Client Side Training """
      history = []
      for epoch in range(epochs):
         for x_batch, y_batch in training_data_fcn:
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            y_prediction = self(x_batch)
            loss = self.criterion(y_prediction, y_batch)
            
            loss.backward()
            self.optimizer.step()
            self.loss_train.append(loss)

         avg_loss = torch.stack(self.loss_train).mean().item()
         history.append(avg_loss)
      
      return history



   def federated_averaging(global_state, client_states):
      """ Average Model Updates from clients using Federated Averaging """
      new_global_state = OrderedDict()

      for key in global_state.keys():
         new_global_state[key] = sum(client_state[key] for client_state in cllient_states) / len(client_states)

      return new_global_state






def average_parameters(client_parameters, client_data_sizes):
   total_data_points = sum(client_data_sizes)
   avg_params = {}
   for key in client_parameters[0].keys():
      # Weighted sum of client parameters
      avg_params[key] = sum([client_parameters[i][key] * (client_data_sizes[i] / total_data_points) for i in range(len(client_parameters))])

   return avg_params



def federated_training(server_model, clients, global_epochs, client_epochs, training_loaders):
   # Get the size of each client's dataset
   client_data_sizes = [len(loader.dataset) for loader in training_loaders]

   for global_epoch in range(global_epochs):
      clients_params = []

      # each client trains locally and sends the model parameters
      for i, client in enumerate(clients):
         pr.prGreen(f"Training client {i+1}...")
         client.train_client(client_epochs,training_loaders[i])
         clients_params.append(client.get_parameters())

      # server aggregates the parameters
      new_params = average_parameters(clients_params, client_data_sizes)

      # update global model (each client updates its parameters by receiving from the global model)
      for client in clients:
         client.set_parameters(new_params)

      pr.prCyan(f"Global Epoch {global_epoch + 1} complete.")

   return server_model



