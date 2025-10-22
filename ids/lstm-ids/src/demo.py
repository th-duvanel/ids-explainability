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

# Suppose you have run two scenarios in the Cooja simulator successfully. Therefore in the two resulting folders the following files exist:

# - events.log:            it includes information about the time that the network entered the steady state, the attacker 
#                          node’s id, attacking time and the time that simulation finishes. All times are in microseconds. 
# - mote-output.log:       it is the most important file. It includes all information that we need to extract features, the UDP packets 
#                          that has been sent to the sink nodes by IoT nodes and the udp packets that has beenreceived by the sink node. 
# - script.log:            it includes the same information as events.log, but in more detail.
# - cooja.log 
# - radio-log.pcap
# - radio-medium.log



# OBSERVABLE PACKETS IN THE SINK NODE
# In the first step, extract the UDP messages that are observable to the sink node. 
# The following command, 
# 1 - extracts all the observable messages in the sink node
# 2 - calculates the mean and standard deviation of all observed messages in each 60 seconds (binSize) 
# 3 - save the resulting 14 features and corresponding labels in the features_timeseries_60_sec.csv file. The file is located in the same folder.
obs.MyDataSet(dataAdd = "/path/to/scenario/folder1/", binSize = 60)
obs.MyDataSet(dataAdd = "/path/to/scenario/folder2/", binSize = 60)
obs.MyDataSet(dataAdd = "/path/to/scenario/folder3/", binSize = 60)



# AGGREGATE
# To use the extracted data from both scenarios to train and test and IDS model, we need to aggregate data.
# NOTE: to aggregate data for the scenario in the paper, you can use available functions in the aggregate.py file.
# Here, as an example we use data from the 1st and the 2nd scenarios to train a model and data from the 3rd scenario to test that model:


trainDataList = []
testDataList = []

# training data
data1 = pd.read_csv("/path/to/scenario/folder1/features_timeseries_60_sec.csv", sep = ',')
data1 = data1.drop(['Unnamed: 0'],axis = 1)       # if there is a column for row numbers
trainDataList.append(data1)
data2 = pd.read_csv("/path/to/scenario/folder2/features_timeseries_60_sec.csv", sep = ',')
data2 = data2.drop(['Unnamed: 0'],axis = 1)       # if there is a column for row numbers
trainDataList.append(data2)
# testing data
data3 = pd.read_csv("/path/to/scenario/folder3/features_timeseries_60_sec.csv", sep = ',')
data3 = data3.drop(['Unnamed: 0'],axis = 1)       # if there is a column for row numbers
testDataList.append(data3)





# MIN MAX NORMALIZATION
# The train data constitiutes of two scenarios. Therefore in each scnario there are a min and a max for each feature. 
# To normalize the train data we find the global min and max and then normalize the data with reapect to them.

#### normalize train data
# find the list of features MINs and MAXs for all scenarios
min_max_list = [(df.min().to_frame(name='Min').T, df.max().to_frame(name='Max').T) for df in trainDataList]

# Separate lists to store min and max DataFrames
min_dfs = [min_df for min_df, max_df in min_max_list]
max_dfs = [max_df for min_df, max_df in min_max_list]

# Concatenate all min and max DataFrames into one DataFrame each
base_all_mins_df = pd.concat(min_dfs, ignore_index=True)
base_all_maxs_df = pd.concat(max_dfs, ignore_index=True)

global_max = base_all_maxs_df.max(axis = 0)
global_min = base_all_mins_df.min(axis = 0)
   
normalized_train = [df.apply(lambda x: (x - global_min[x.name]) / (global_max[x.name] - global_min[x.name])) for df in trainDataList]



##### normmalize test data based on the global_min and global_max found in the train data
normalized_test = [df.apply(lambda x: (x - global_min[x.name]) / (global_max[x.name] - global_min[x.name])) for df in testDataList]



# SEQUENCES AS INPUT TO THE LSTM MODEL
# To use a LSTM model we have to decide about the sequence length of input data. As we have two groups of timeseries 
# (each Cooja's scenario generates a 14 timeseries data), we can not easily use the sequence length hyperparameter in 
# the LSTM model. Therefore, in this step we generate the sequences and save them in 
# each scenario's folder and then we set the sequence length hyperparameter in the LSTM model as 1. To generate these 
# sequences use seqMaker.py. In our paper we used sequence_length = 10 (10 minutes) as the input.
seq_Train = [seqMaker.seq_maker(df,10) for df in normalized_train]
seq_Train = pd.concat(seq_Train, ignore_index=True)        

seq_Test = [seqMaker.seq_maker(df,10) for df in normalized_test]
seq_Test = pd.concat(seq_Test, ignore_index=True)




# MODELING
# extract X and y 
X_Train = seq_Train.iloc[:, :-1].values  # All columns except the last one
y_Train = seq_Train.iloc[:, -1].values  # The last column
X_Test = seq_Test.iloc[:, :-1].values  # All columns except the last one
y_Test = seq_Test.iloc[:, -1].values  # The last column

# tensorizing the data for base
X_Train = np.array(X_Train)
y_Train = np.array(y_Train)
X_Test = np.array(X_Test)
y_Test = np.array(y_Test)

# add one dimention for the LSTM model
X_Train = torch.tensor(X_Train, dtype=torch.float32)  # Shape will be (num_samples, sequence_length = 1, feature_size)
y_Train = torch.tensor(y_Train, dtype = torch.long)   # Shape will be (num_samples, 1)
X_Test = torch.tensor(X_Test, dtype=torch.float32)    # Shape will be (num_samples, sequence_length = 1, feature_size)
y_Test = torch.tensor(y_Test, dtype = torch.long)     # Shape will be (num_samples, 1)

# Check for NaNs due to min-max normalization
X_Train = torch.nan_to_num(X_Train, nan=0.0)
X_Test = torch.nan_to_num(X_Test, nan=0.0)

# Reshape X to add sequence length dimension 
X_Train = X_Train.view(-1, 1, 140)             # Reshapes to (num_samples, sequence_length = 1, feature_size)
train_dataset = TensorDataset(X_Train, y_Train)
X_Test = X_Test.view(-1, 1, 140)               # Reshapes to (num_samples, sequence_length = 1, feature_size)
test_dataset = TensorDataset(X_Test, y_Test)


# Create DataLoaders for train and test sets 
batch_size = 2048
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the IDS models
input_dim = 140             # we defined the sequence length as 10 before, and we knew that the number of features is 14. Therefore, input_dime = 14 * 10
hidden_dim = 10             # LSTM hyperparameter
fc_hidden_dim = 10          # LSTM hyperparameter
num_layers = 1              # LSTM hyperparameter
output_dim = 2              # output: attack, non-attack
lr = 0.001                  # learning rate

ids_model = mine.LSTMClassifier(input_dim = input_dim, hidden_dim = hidden_dim, num_layers = num_layers, output_dim = output_dim,fc_hidden_dim = fc_hidden_dim, learning_rate = lr)

# Train the model
epochs = 10
pr.prGreen("Training ...")
ids_model.model_train(epochs = epochs, train_loader = train_loader)

# Test the trained model
pr.prRed(round(ids_model.evaluate_model_ROCAUC(test_loader),3))


pr.prCyan("Congratulations!  You have finished the demo successfuly.")