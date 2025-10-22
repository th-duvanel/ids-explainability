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



def seq_maker(df, sequence_length = 10):
   #sequences = pd.DataFrame()
   sequences = []
   start_attack = df[df['label'] == 1].index[0]
   start_attack = start_attack - sequence_length
   sequence_zeros = [0] * start_attack
   df2 = df.iloc[:,:-1]
   #df2 = df2.drop(df.columns[0], axis=1)
   for i in range(len(df2) - sequence_length):
      seq = df2.iloc[i:i+sequence_length]
      flattened_seq = seq.values.flatten()
      sequences.append(flattened_seq)
      
   df2 = pd.DataFrame(sequences)
   sequence_ones = [1] * (df2.shape[0] - start_attack)
   label = sequence_zeros + sequence_ones
   df2['label'] = label
   
   return df2


