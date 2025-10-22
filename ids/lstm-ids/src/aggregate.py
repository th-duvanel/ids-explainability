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




"""
Functions in this file aggregate data from all Cooja scenarios upon the needs.
For example to test generalizability of IDS models on different attack types (BH and DIS), 
we aggregate all BH data to train IDS model 1 and aggregate all DIS data to train IDS model 2.
"""



def aggregateBH():
   rootName = "/root/add/of/all/BH/scenarios/"
   dataAll = pd.DataFrame()

   size_folders = os.listdir(rootName)


   for l1 in size_folders:
      var_folders = os.listdir(rootName + l1)
      for l2 in var_folders:
         scenario_folders = os.listdir(rootName + l1 + "/" + l2)
         # Filter out only folders (directories)
         scenario_folders = [item for item in scenario_folders if os.path.isdir(os.path.join(rootName + l1 + "/" + l2 + "/" , item))]
         # Select scenarios for Train
         scenario_folders = random.sample(scenario_folders, min(15, len(scenario_folders)))
         # Select scenarios for Test
         test_scenario_folders = [folder for folder in folders if folder not in scenario_folders]


         for l3 in scenario_folders:
            file_name = rootName + l1 + "/" + l2 + "/" + l3 + "/sequences_10.csv"
            myData = pd.read_csv(file_name, sep = ',')
            myData = myData.drop(['Unnamed: 0'],axis = 1)
            dataAll = pd.concat([dataAll,myData])
         
   

   
   print(dataAll.shape)

   return dataAll


def aggregateBH_list():
   rootName = "/root/add/of/all/BH/scenarios/"
   trainDataList = []
   testDataList = []
   dataAll = pd.DataFrame()
   size_folders = os.listdir(rootName)
   
   X = []
   y = []   
   
   for l1 in size_folders:
      var_folders = os.listdir(rootName + l1)
      for l2 in var_folders:
         all_folders = os.listdir(rootName + l1 + "/" + l2)
         # Filter out only folders (directories)
         all_folders = [item for item in all_folders if os.path.isdir(os.path.join(rootName + l1 + "/" + l2 + "/" , item))]
         scenario_folders = random.sample(all_folders, min(15, len(all_folders)))
         # Select scenarios for Test
         test_scenario_folders = [folder for folder in all_folders if folder not in scenario_folders]

         for l3 in scenario_folders:
            file_name = rootName + l1 + "/" + l2 + "/" + l3 + "/features_timeseries_60_sec.csv"
            myData = pd.read_csv(file_name, sep = ',')
            myData = myData.drop(['Unnamed: 0'],axis = 1)
            trainDataList.append(myData)
            #input_features = myData.iloc[:, :-1].values  # All columns except the last one
            #targets = myData.iloc[:, -1].values  # The last column
         
         for l3 in test_scenario_folders:
            file_name = rootName + l1 + "/" + l2 + "/" + l3 + "/features_timeseries_60_sec.csv"
            myData = pd.read_csv(file_name, sep = ',')
            myData = myData.drop(['Unnamed: 0'],axis = 1)
            testDataList.append(myData)
            #input_features = myData.iloc[:, :-1].values  # All columns except the last one
            #targets = myData.iloc[:, -1].values  # The last column

            # for i in range(len(myData) - sequence_length):
            #    X.append(input_features[i:i + sequence_length])
            #    y.append(targets[i + sequence_length])   

   

   return trainDataList, testDataList




def aggregateDIS_list():
   rootName = "/root/add/of/all/DIS/scenarios/"
   trainDataList = []
   testDataList = []
   dataAll = pd.DataFrame()
   size_folders = os.listdir(rootName)
   
   X = []
   y = []   
   
   for l1 in size_folders:
      var_folders = os.listdir(rootName + l1)
      for l2 in var_folders:
         all_folders = os.listdir(rootName + l1 + "/" + l2)
         # Filter out only folders (directories)
         all_folders = [item for item in all_folders if os.path.isdir(os.path.join(rootName + l1 + "/" + l2 + "/" , item))]
         scenario_folders = random.sample(all_folders, min(15, len(all_folders)))
         # Select scenarios for Test
         test_scenario_folders = [folder for folder in all_folders if folder not in scenario_folders]

         for l3 in scenario_folders:
            file_name = rootName + l1 + "/" + l2 + "/" + l3 + "/features_timeseries_60_sec.csv"
            myData = pd.read_csv(file_name, sep = ',')
            myData = myData.drop(['Unnamed: 0'],axis = 1)
            trainDataList.append(myData)
            #input_features = myData.iloc[:, :-1].values  # All columns except the last one
            #targets = myData.iloc[:, -1].values  # The last column
         
         for l3 in test_scenario_folders:
            file_name = rootName + l1 + "/" + l2 + "/" + l3 + "/features_timeseries_60_sec.csv"
            myData = pd.read_csv(file_name, sep = ',')
            myData = myData.drop(['Unnamed: 0'],axis = 1)
            testDataList.append(myData)
            #input_features = myData.iloc[:, :-1].values  # All columns except the last one
            #targets = myData.iloc[:, -1].values  # The last column

            # for i in range(len(myData) - sequence_length):
            #    X.append(input_features[i:i + sequence_length])
            #    y.append(targets[i + sequence_length])   

   

   return trainDataList, testDataList




def aggregate_list_all():
   bhTrain, bhTest = aggregateBH_list()
   disTrain, disTest = aggregateDIS_list()

   trainDataList = bhTrain + disTrain
   testDataList = bhTest + disTest

   return trainDataList, testDataList



   






def aggregateDIS_list_5():
   rootName = "/root/add/of/all/DIS/scenarios/var5/"
   trainDataList = []
   testDataList = []
   
   
   var_folders = os.listdir(rootName)
   for l2 in var_folders:
      all_folders = os.listdir(rootName + l2)
      # Filter out only folders (directories)
      all_folders = [item for item in all_folders if os.path.isdir(os.path.join(rootName + l2 + "/" , item))]
      scenario_folders = random.sample(all_folders, min(15, len(all_folders)))
      # Select scenarios for Test
      test_scenario_folders = [folder for folder in all_folders if folder not in scenario_folders]
      
      for l3 in scenario_folders:
         file_name = rootName + l2 + "/" + l3 + "/features_timeseries_60_sec.csv"
         myData = pd.read_csv(file_name, sep = ',')
         myData = myData.drop(['Unnamed: 0'],axis = 1)
         trainDataList.append(myData)
         
      for l3 in test_scenario_folders:
         file_name = rootName +  l2 + "/" + l3 + "/features_timeseries_60_sec.csv"
         myData = pd.read_csv(file_name, sep = ',')
         myData = myData.drop(['Unnamed: 0'],axis = 1)
         testDataList.append(myData)

   return trainDataList, testDataList


def aggregateDIS_list_10():
   rootName = "/root/add/of/all/DIS/scenarios/var10/"
   trainDataList = []
   testDataList = []
   
   
   var_folders = os.listdir(rootName)
   for l2 in var_folders:
      all_folders = os.listdir(rootName + l2)
      # Filter out only folders (directories)
      all_folders = [item for item in all_folders if os.path.isdir(os.path.join(rootName + l2 + "/" , item))]
      scenario_folders = random.sample(all_folders, min(15, len(all_folders)))
      # Select scenarios for Test
      test_scenario_folders = [folder for folder in all_folders if folder not in scenario_folders]
      
      for l3 in scenario_folders:
         file_name = rootName + l2 + "/" + l3 + "/features_timeseries_60_sec.csv"
         myData = pd.read_csv(file_name, sep = ',')
         myData = myData.drop(['Unnamed: 0'],axis = 1)
         trainDataList.append(myData)
         
      for l3 in test_scenario_folders:
         file_name = rootName +  l2 + "/" + l3 + "/features_timeseries_60_sec.csv"
         myData = pd.read_csv(file_name, sep = ',')
         myData = myData.drop(['Unnamed: 0'],axis = 1)
         testDataList.append(myData)

   return trainDataList, testDataList






def aggregateDIS_list_15():
   rootName = "/root/add/of/all/DIS/scenarios/var15/"
   trainDataList = []
   testDataList = []
   
   
   var_folders = os.listdir(rootName)
   for l2 in var_folders:
      all_folders = os.listdir(rootName + l2)
      # Filter out only folders (directories)
      all_folders = [item for item in all_folders if os.path.isdir(os.path.join(rootName + l2 + "/" , item))]
      scenario_folders = random.sample(all_folders, min(15, len(all_folders)))
      # Select scenarios for Test
      test_scenario_folders = [folder for folder in all_folders if folder not in scenario_folders]
      
      for l3 in scenario_folders:
         file_name = rootName + l2 + "/" + l3 + "/features_timeseries_60_sec.csv"
         myData = pd.read_csv(file_name, sep = ',')
         myData = myData.drop(['Unnamed: 0'],axis = 1)
         trainDataList.append(myData)
         
      for l3 in test_scenario_folders:
         file_name = rootName + l2 + "/" + l3 + "/features_timeseries_60_sec.csv"
         myData = pd.read_csv(file_name, sep = ',')
         myData = myData.drop(['Unnamed: 0'],axis = 1)
         testDataList.append(myData)

   return trainDataList, testDataList







def aggregateDIS_list_20():
   rootName = "/root/add/of/all/DIS/scenarios/var20/"
   trainDataList = []
   testDataList = []
   
   
   var_folders = os.listdir(rootName)
   for l2 in var_folders:
      all_folders = os.listdir(rootName + l2)
      # Filter out only folders (directories)
      all_folders = [item for item in all_folders if os.path.isdir(os.path.join(rootName + l2 + "/" , item))]
      scenario_folders = random.sample(all_folders, min(15, len(all_folders)))
      # Select scenarios for Test
      test_scenario_folders = [folder for folder in all_folders if folder not in scenario_folders]
      
      for l3 in scenario_folders:
         file_name = rootName  + l2 + "/" + l3 + "/features_timeseries_60_sec.csv"
         myData = pd.read_csv(file_name, sep = ',')
         myData = myData.drop(['Unnamed: 0'],axis = 1)
         trainDataList.append(myData)
         
      for l3 in test_scenario_folders:
         file_name = rootName  + l2 + "/" + l3 + "/features_timeseries_60_sec.csv"
         myData = pd.read_csv(file_name, sep = ',')
         myData = myData.drop(['Unnamed: 0'],axis = 1)
         testDataList.append(myData)

   return trainDataList, testDataList






def aggregateBH_list_5():
   rootName = "/root/add/of/all/BH/scenarios/var5/"
   trainDataList = []
   testDataList = []
   
   
   var_folders = os.listdir(rootName)
   for l2 in var_folders:
      all_folders = os.listdir(rootName + l2)
      # Filter out only folders (directories)
      all_folders = [item for item in all_folders if os.path.isdir(os.path.join(rootName + l2 + "/" , item))]
      scenario_folders = random.sample(all_folders, min(15, len(all_folders)))
      # Select scenarios for Test
      test_scenario_folders = [folder for folder in all_folders if folder not in scenario_folders]
      
      for l3 in scenario_folders:
         file_name = rootName + l2 + "/" + l3 + "/features_timeseries_60_sec.csv"
         myData = pd.read_csv(file_name, sep = ',')
         myData = myData.drop(['Unnamed: 0'],axis = 1)
         trainDataList.append(myData)
         
      for l3 in test_scenario_folders:
         file_name = rootName +  l2 + "/" + l3 + "/features_timeseries_60_sec.csv"
         myData = pd.read_csv(file_name, sep = ',')
         myData = myData.drop(['Unnamed: 0'],axis = 1)
         testDataList.append(myData)

   return trainDataList, testDataList


def aggregateBH_list_10():
   rootName = "/root/add/of/all/BH/scenarios/var10/"
   trainDataList = []
   testDataList = []
   
   
   var_folders = os.listdir(rootName)
   for l2 in var_folders:
      all_folders = os.listdir(rootName + l2)
      # Filter out only folders (directories)
      all_folders = [item for item in all_folders if os.path.isdir(os.path.join(rootName + l2 + "/" , item))]
      scenario_folders = random.sample(all_folders, min(15, len(all_folders)))
      # Select scenarios for Test
      test_scenario_folders = [folder for folder in all_folders if folder not in scenario_folders]
      
      for l3 in scenario_folders:
         file_name = rootName + l2 + "/" + l3 + "/features_timeseries_60_sec.csv"
         myData = pd.read_csv(file_name, sep = ',')
         myData = myData.drop(['Unnamed: 0'],axis = 1)
         trainDataList.append(myData)
         
      for l3 in test_scenario_folders:
         file_name = rootName +  l2 + "/" + l3 + "/features_timeseries_60_sec.csv"
         myData = pd.read_csv(file_name, sep = ',')
         myData = myData.drop(['Unnamed: 0'],axis = 1)
         testDataList.append(myData)

   return trainDataList, testDataList




def aggregateBH_list_15():
   rootName = "/root/add/of/all/BH/scenarios/var15/"
   trainDataList = []
   testDataList = []
   
   
   var_folders = os.listdir(rootName)
   for l2 in var_folders:
      all_folders = os.listdir(rootName + l2)
      # Filter out only folders (directories)
      all_folders = [item for item in all_folders if os.path.isdir(os.path.join(rootName + l2 + "/" , item))]
      scenario_folders = random.sample(all_folders, min(15, len(all_folders)))
      # Select scenarios for Test
      test_scenario_folders = [folder for folder in all_folders if folder not in scenario_folders]
      
      for l3 in scenario_folders:
         file_name = rootName + l2 + "/" + l3 + "/features_timeseries_60_sec.csv"
         myData = pd.read_csv(file_name, sep = ',')
         myData = myData.drop(['Unnamed: 0'],axis = 1)
         trainDataList.append(myData)
         
      for l3 in test_scenario_folders:
         file_name = rootName +  l2 + "/" + l3 + "/features_timeseries_60_sec.csv"
         myData = pd.read_csv(file_name, sep = ',')
         myData = myData.drop(['Unnamed: 0'],axis = 1)
         testDataList.append(myData)

   return trainDataList, testDataList





def aggregateBH_list_20():
   rootName = "/root/add/of/all/BH/scenarios/var20/"
   trainDataList = []
   testDataList = []
   
   
   var_folders = os.listdir(rootName)
   for l2 in var_folders:
      all_folders = os.listdir(rootName + l2)
      # Filter out only folders (directories)
      all_folders = [item for item in all_folders if os.path.isdir(os.path.join(rootName + l2 + "/" , item))]
      scenario_folders = random.sample(all_folders, min(15, len(all_folders)))
      # Select scenarios for Test
      test_scenario_folders = [folder for folder in all_folders if folder not in scenario_folders]
      
      for l3 in scenario_folders:
         file_name = rootName + l2 + "/" + l3 + "/features_timeseries_60_sec.csv"
         myData = pd.read_csv(file_name, sep = ',')
         myData = myData.drop(['Unnamed: 0'],axis = 1)
         trainDataList.append(myData)
         
      for l3 in test_scenario_folders:
         file_name = rootName +  l2 + "/" + l3 + "/features_timeseries_60_sec.csv"
         myData = pd.read_csv(file_name, sep = ',')
         myData = myData.drop(['Unnamed: 0'],axis = 1)
         testDataList.append(myData)

   return trainDataList, testDataList


def aggregate_list_5():
   trainDataList_BH, testDataList_BH = aggregateBH_list_5()
   trainDataList_DIS, testDataList_DIS = aggregateDIS_list_5()
   
   trainDataList = trainDataList_BH + trainDataList_DIS
   testDataList = testDataList_BH + testDataList_DIS

   return trainDataList, testDataList



def aggregate_list_10():
   trainDataList_BH, testDataList_BH = aggregateBH_list_10()
   trainDataList_DIS, testDataList_DIS = aggregateDIS_list_10()
   
   trainDataList = trainDataList_BH + trainDataList_DIS
   testDataList = testDataList_BH + testDataList_DIS

   return trainDataList, testDataList


def aggregate_list_15():
   trainDataList_BH, testDataList_BH = aggregateBH_list_15()
   trainDataList_DIS, testDataList_DIS = aggregateDIS_list_15()
   
   trainDataList = trainDataList_BH + trainDataList_DIS
   testDataList = testDataList_BH + testDataList_DIS

   return trainDataList, testDataList


def aggregate_list_20():
   trainDataList_BH, testDataList_BH = aggregateBH_list_20()
   trainDataList_DIS, testDataList_DIS = aggregateDIS_list_20()
   
   trainDataList = trainDataList_BH + trainDataList_DIS
   testDataList = testDataList_BH + testDataList_DIS

   return trainDataList, testDataList







def aggregateDIS_list_base():
   rootName5 = "/root/add/of/all/DIS/scenarios/var5/base/"
   rootName10 = "/root/add/of/all/DIS/scenarios/var10/base/"
   rootName15 = "/root/add/of/all/DIS/scenarios/var15/base/"
   rootName20 = "/root/add/of/all/DIS/scenarios/var20/base/"

   trainDataList = []
   testDataList = []
   
   trainDataList5 = []
   trainDataList10 = []
   trainDataList15 = []
   trainDataList20 = []
   testDataList5 = []
   testDataList10 = []
   testDataList15 = []
   testDataList20 = []

   
   all_folders = os.listdir(rootName5)
   # Filter out only folders (directories)
   all_folders = [item for item in all_folders if os.path.isdir(os.path.join(rootName5 , item))]
   scenario_folders = random.sample(all_folders, min(15, len(all_folders)))
   # Select scenarios for Test
   test_scenario_folders = [folder for folder in all_folders if folder not in scenario_folders]
      
   for l3 in scenario_folders:
      file_name = rootName5 + l3 + "/features_timeseries_60_sec.csv"
      myData = pd.read_csv(file_name, sep = ',')
      myData = myData.drop(['Unnamed: 0'],axis = 1)
      trainDataList5.append(myData)
         
   for l3 in test_scenario_folders:
      file_name = rootName5 + l3 + "/features_timeseries_60_sec.csv"
      myData = pd.read_csv(file_name, sep = ',')
      myData = myData.drop(['Unnamed: 0'],axis = 1)
      testDataList5.append(myData)

   all_folders = os.listdir(rootName10)
   # Filter out only folders (directories)
   all_folders = [item for item in all_folders if os.path.isdir(os.path.join(rootName10 , item))]
   scenario_folders = random.sample(all_folders, min(15, len(all_folders)))
   # Select scenarios for Test
   test_scenario_folders = [folder for folder in all_folders if folder not in scenario_folders]
      
   for l3 in scenario_folders:
      file_name = rootName10 + l3 + "/features_timeseries_60_sec.csv"
      myData = pd.read_csv(file_name, sep = ',')
      myData = myData.drop(['Unnamed: 0'],axis = 1)
      trainDataList10.append(myData)
         
   for l3 in test_scenario_folders:
      file_name = rootName10 + l3 + "/features_timeseries_60_sec.csv"
      myData = pd.read_csv(file_name, sep = ',')
      myData = myData.drop(['Unnamed: 0'],axis = 1)
      testDataList10.append(myData)
   
   all_folders = os.listdir(rootName15)
   # Filter out only folders (directories)
   all_folders = [item for item in all_folders if os.path.isdir(os.path.join(rootName15 , item))]
   scenario_folders = random.sample(all_folders, min(15, len(all_folders)))
   # Select scenarios for Test
   test_scenario_folders = [folder for folder in all_folders if folder not in scenario_folders]
      
   for l3 in scenario_folders:
      file_name = rootName15 + l3 + "/features_timeseries_60_sec.csv"
      myData = pd.read_csv(file_name, sep = ',')
      myData = myData.drop(['Unnamed: 0'],axis = 1)
      trainDataList15.append(myData)
         
   for l3 in test_scenario_folders:
      file_name = rootName15 + l3 + "/features_timeseries_60_sec.csv"
      myData = pd.read_csv(file_name, sep = ',')
      myData = myData.drop(['Unnamed: 0'],axis = 1)
      testDataList15.append(myData)

   all_folders = os.listdir(rootName20)
   # Filter out only folders (directories)
   all_folders = [item for item in all_folders if os.path.isdir(os.path.join(rootName20 , item))]
   scenario_folders = random.sample(all_folders, min(15, len(all_folders)))
   # Select scenarios for Test
   test_scenario_folders = [folder for folder in all_folders if folder not in scenario_folders]
      
   for l3 in scenario_folders:
      file_name = rootName20 + l3 + "/features_timeseries_60_sec.csv"
      myData = pd.read_csv(file_name, sep = ',')
      myData = myData.drop(['Unnamed: 0'],axis = 1)
      trainDataList20.append(myData)
         
   for l3 in test_scenario_folders:
      file_name = rootName20 + l3 + "/features_timeseries_60_sec.csv"
      myData = pd.read_csv(file_name, sep = ',')
      myData = myData.drop(['Unnamed: 0'],axis = 1)
      testDataList20.append(myData)

   testDataList = testDataList5 + testDataList10 + testDataList15 + testDataList20
   trainDataList = trainDataList5 + trainDataList10 + trainDataList15 + trainDataList20


   return trainDataList, testDataList





def aggregateDIS_list_oo():
   rootName5 = "/root/add/of/all/DIS/scenarios/var5/oo/"
   rootName10 = "/root/add/of/all/DIS/scenarios/var10/oo/"
   rootName15 = "/root/add/of/all/DIS/scenarios/var15/oo/"
   rootName20 = "/root/add/of/all/DIS/scenarios/var20/oo/"

   trainDataList = []
   testDataList = []
   
   trainDataList5 = []
   trainDataList10 = []
   trainDataList15 = []
   trainDataList20 = []
   testDataList5 = []
   testDataList10 = []
   testDataList15 = []
   testDataList20 = []

   
   all_folders = os.listdir(rootName5)
   # Filter out only folders (directories)
   all_folders = [item for item in all_folders if os.path.isdir(os.path.join(rootName5 , item))]
   scenario_folders = random.sample(all_folders, min(15, len(all_folders)))
   # Select scenarios for Test
   test_scenario_folders = [folder for folder in all_folders if folder not in scenario_folders]
      
   for l3 in scenario_folders:
      file_name = rootName5 + l3 + "/features_timeseries_60_sec.csv"
      myData = pd.read_csv(file_name, sep = ',')
      myData = myData.drop(['Unnamed: 0'],axis = 1)
      trainDataList5.append(myData)
         
   for l3 in test_scenario_folders:
      file_name = rootName5 + l3 + "/features_timeseries_60_sec.csv"
      myData = pd.read_csv(file_name, sep = ',')
      myData = myData.drop(['Unnamed: 0'],axis = 1)
      testDataList5.append(myData)

   all_folders = os.listdir(rootName10)
   # Filter out only folders (directories)
   all_folders = [item for item in all_folders if os.path.isdir(os.path.join(rootName10 , item))]
   scenario_folders = random.sample(all_folders, min(15, len(all_folders)))
   # Select scenarios for Test
   test_scenario_folders = [folder for folder in all_folders if folder not in scenario_folders]
      
   for l3 in scenario_folders:
      file_name = rootName10 + l3 + "/features_timeseries_60_sec.csv"
      myData = pd.read_csv(file_name, sep = ',')
      myData = myData.drop(['Unnamed: 0'],axis = 1)
      trainDataList10.append(myData)
         
   for l3 in test_scenario_folders:
      file_name = rootName10 + l3 + "/features_timeseries_60_sec.csv"
      myData = pd.read_csv(file_name, sep = ',')
      myData = myData.drop(['Unnamed: 0'],axis = 1)
      testDataList10.append(myData)
   
   all_folders = os.listdir(rootName15)
   # Filter out only folders (directories)
   all_folders = [item for item in all_folders if os.path.isdir(os.path.join(rootName15 , item))]
   scenario_folders = random.sample(all_folders, min(15, len(all_folders)))
   # Select scenarios for Test
   test_scenario_folders = [folder for folder in all_folders if folder not in scenario_folders]
      
   for l3 in scenario_folders:
      file_name = rootName15 + l3 + "/features_timeseries_60_sec.csv"
      myData = pd.read_csv(file_name, sep = ',')
      myData = myData.drop(['Unnamed: 0'],axis = 1)
      trainDataList15.append(myData)
         
   for l3 in test_scenario_folders:
      file_name = rootName15 + l3 + "/features_timeseries_60_sec.csv"
      myData = pd.read_csv(file_name, sep = ',')
      myData = myData.drop(['Unnamed: 0'],axis = 1)
      testDataList15.append(myData)

   all_folders = os.listdir(rootName20)
   # Filter out only folders (directories)
   all_folders = [item for item in all_folders if os.path.isdir(os.path.join(rootName20 , item))]
   scenario_folders = random.sample(all_folders, min(15, len(all_folders)))
   # Select scenarios for Test
   test_scenario_folders = [folder for folder in all_folders if folder not in scenario_folders]
      
   for l3 in scenario_folders:
      file_name = rootName20 + l3 + "/features_timeseries_60_sec.csv"
      myData = pd.read_csv(file_name, sep = ',')
      myData = myData.drop(['Unnamed: 0'],axis = 1)
      trainDataList20.append(myData)
         
   for l3 in test_scenario_folders:
      file_name = rootName20 + l3 + "/features_timeseries_60_sec.csv"
      myData = pd.read_csv(file_name, sep = ',')
      myData = myData.drop(['Unnamed: 0'],axis = 1)
      testDataList20.append(myData)

   testDataList = testDataList5 + testDataList10 + testDataList15 + testDataList20
   trainDataList = trainDataList5 + trainDataList10 + trainDataList15 + trainDataList20


   return trainDataList, testDataList




def aggregateDIS_list_dec():
   rootName5 = "/root/add/of/all/DIS/scenarios/var5/dec/"
   rootName10 = "/root/add/of/all/DIS/scenarios/var10/dec/"
   rootName15 = "/root/add/of/all/DIS/scenarios/var15/dec/"
   rootName20 = "/root/add/of/all/DIS/scenarios/var20/dec/"

   trainDataList = []
   testDataList = []
   
   trainDataList5 = []
   trainDataList10 = []
   trainDataList15 = []
   trainDataList20 = []
   testDataList5 = []
   testDataList10 = []
   testDataList15 = []
   testDataList20 = []

   
   all_folders = os.listdir(rootName5)
   # Filter out only folders (directories)
   all_folders = [item for item in all_folders if os.path.isdir(os.path.join(rootName5 , item))]
   scenario_folders = random.sample(all_folders, min(15, len(all_folders)))
   # Select scenarios for Test
   test_scenario_folders = [folder for folder in all_folders if folder not in scenario_folders]
      
   for l3 in scenario_folders:
      file_name = rootName5 + l3 + "/features_timeseries_60_sec.csv"
      myData = pd.read_csv(file_name, sep = ',')
      myData = myData.drop(['Unnamed: 0'],axis = 1)
      trainDataList5.append(myData)
         
   for l3 in test_scenario_folders:
      file_name = rootName5 + l3 + "/features_timeseries_60_sec.csv"
      myData = pd.read_csv(file_name, sep = ',')
      myData = myData.drop(['Unnamed: 0'],axis = 1)
      testDataList5.append(myData)

   all_folders = os.listdir(rootName10)
   # Filter out only folders (directories)
   all_folders = [item for item in all_folders if os.path.isdir(os.path.join(rootName10 , item))]
   scenario_folders = random.sample(all_folders, min(15, len(all_folders)))
   # Select scenarios for Test
   test_scenario_folders = [folder for folder in all_folders if folder not in scenario_folders]
      
   for l3 in scenario_folders:
      file_name = rootName10 + l3 + "/features_timeseries_60_sec.csv"
      myData = pd.read_csv(file_name, sep = ',')
      myData = myData.drop(['Unnamed: 0'],axis = 1)
      trainDataList10.append(myData)
         
   for l3 in test_scenario_folders:
      file_name = rootName10 + l3 + "/features_timeseries_60_sec.csv"
      myData = pd.read_csv(file_name, sep = ',')
      myData = myData.drop(['Unnamed: 0'],axis = 1)
      testDataList10.append(myData)
   
   all_folders = os.listdir(rootName15)
   # Filter out only folders (directories)
   all_folders = [item for item in all_folders if os.path.isdir(os.path.join(rootName15 , item))]
   scenario_folders = random.sample(all_folders, min(15, len(all_folders)))
   # Select scenarios for Test
   test_scenario_folders = [folder for folder in all_folders if folder not in scenario_folders]
      
   for l3 in scenario_folders:
      file_name = rootName15 + l3 + "/features_timeseries_60_sec.csv"
      myData = pd.read_csv(file_name, sep = ',')
      myData = myData.drop(['Unnamed: 0'],axis = 1)
      trainDataList15.append(myData)
         
   for l3 in test_scenario_folders:
      file_name = rootName15 + l3 + "/features_timeseries_60_sec.csv"
      myData = pd.read_csv(file_name, sep = ',')
      myData = myData.drop(['Unnamed: 0'],axis = 1)
      testDataList15.append(myData)

   all_folders = os.listdir(rootName20)
   # Filter out only folders (directories)
   all_folders = [item for item in all_folders if os.path.isdir(os.path.join(rootName20 , item))]
   scenario_folders = random.sample(all_folders, min(15, len(all_folders)))
   # Select scenarios for Test
   test_scenario_folders = [folder for folder in all_folders if folder not in scenario_folders]
      
   for l3 in scenario_folders:
      file_name = rootName20 + l3 + "/features_timeseries_60_sec.csv"
      myData = pd.read_csv(file_name, sep = ',')
      myData = myData.drop(['Unnamed: 0'],axis = 1)
      trainDataList20.append(myData)
         
   for l3 in test_scenario_folders:
      file_name = rootName20 + l3 + "/features_timeseries_60_sec.csv"
      myData = pd.read_csv(file_name, sep = ',')
      myData = myData.drop(['Unnamed: 0'],axis = 1)
      testDataList20.append(myData)

   testDataList = testDataList5 + testDataList10 + testDataList15 + testDataList20
   trainDataList = trainDataList5 + trainDataList10 + trainDataList15 + trainDataList20


   return trainDataList, testDataList














def aggregateBH_list_base():
   rootName5 = "/root/add/of/all/BH/scenarios/var5/base/"
   rootName10 = "/root/add/of/all/BH/scenarios/var10/base/"
   rootName15 = "/root/add/of/all/BH/scenarios/var15/base/"
   rootName20 = "/root/add/of/all/BH/scenarios/var20/base/"

   trainDataList = []
   testDataList = []
   
   trainDataList5 = []
   trainDataList10 = []
   trainDataList15 = []
   trainDataList20 = []
   testDataList5 = []
   testDataList10 = []
   testDataList15 = []
   testDataList20 = []

   
   all_folders = os.listdir(rootName5)
   # Filter out only folders (directories)
   all_folders = [item for item in all_folders if os.path.isdir(os.path.join(rootName5 , item))]
   scenario_folders = random.sample(all_folders, min(15, len(all_folders)))
   # Select scenarios for Test
   test_scenario_folders = [folder for folder in all_folders if folder not in scenario_folders]
      
   for l3 in scenario_folders:
      file_name = rootName5 + l3 + "/features_timeseries_60_sec.csv"
      myData = pd.read_csv(file_name, sep = ',')
      myData = myData.drop(['Unnamed: 0'],axis = 1)
      trainDataList5.append(myData)
         
   for l3 in test_scenario_folders:
      file_name = rootName5 + l3 + "/features_timeseries_60_sec.csv"
      myData = pd.read_csv(file_name, sep = ',')
      myData = myData.drop(['Unnamed: 0'],axis = 1)
      testDataList5.append(myData)

   all_folders = os.listdir(rootName10)
   # Filter out only folders (directories)
   all_folders = [item for item in all_folders if os.path.isdir(os.path.join(rootName10 , item))]
   scenario_folders = random.sample(all_folders, min(15, len(all_folders)))
   # Select scenarios for Test
   test_scenario_folders = [folder for folder in all_folders if folder not in scenario_folders]
      
   for l3 in scenario_folders:
      file_name = rootName10 + l3 + "/features_timeseries_60_sec.csv"
      myData = pd.read_csv(file_name, sep = ',')
      myData = myData.drop(['Unnamed: 0'],axis = 1)
      trainDataList10.append(myData)
         
   for l3 in test_scenario_folders:
      file_name = rootName10 + l3 + "/features_timeseries_60_sec.csv"
      myData = pd.read_csv(file_name, sep = ',')
      myData = myData.drop(['Unnamed: 0'],axis = 1)
      testDataList10.append(myData)
   
   all_folders = os.listdir(rootName15)
   # Filter out only folders (directories)
   all_folders = [item for item in all_folders if os.path.isdir(os.path.join(rootName15 , item))]
   scenario_folders = random.sample(all_folders, min(15, len(all_folders)))
   # Select scenarios for Test
   test_scenario_folders = [folder for folder in all_folders if folder not in scenario_folders]
      
   for l3 in scenario_folders:
      file_name = rootName15 + l3 + "/features_timeseries_60_sec.csv"
      myData = pd.read_csv(file_name, sep = ',')
      myData = myData.drop(['Unnamed: 0'],axis = 1)
      trainDataList15.append(myData)
         
   for l3 in test_scenario_folders:
      file_name = rootName15 + l3 + "/features_timeseries_60_sec.csv"
      myData = pd.read_csv(file_name, sep = ',')
      myData = myData.drop(['Unnamed: 0'],axis = 1)
      testDataList15.append(myData)

   all_folders = os.listdir(rootName20)
   # Filter out only folders (directories)
   all_folders = [item for item in all_folders if os.path.isdir(os.path.join(rootName20 , item))]
   scenario_folders = random.sample(all_folders, min(15, len(all_folders)))
   # Select scenarios for Test
   test_scenario_folders = [folder for folder in all_folders if folder not in scenario_folders]
      
   for l3 in scenario_folders:
      file_name = rootName20 + l3 + "/features_timeseries_60_sec.csv"
      myData = pd.read_csv(file_name, sep = ',')
      myData = myData.drop(['Unnamed: 0'],axis = 1)
      trainDataList20.append(myData)
         
   for l3 in test_scenario_folders:
      file_name = rootName20 + l3 + "/features_timeseries_60_sec.csv"
      myData = pd.read_csv(file_name, sep = ',')
      myData = myData.drop(['Unnamed: 0'],axis = 1)
      testDataList20.append(myData)

   testDataList = testDataList5 + testDataList10 + testDataList15 + testDataList20
   trainDataList = trainDataList5 + trainDataList10 + trainDataList15 + trainDataList20


   return trainDataList, testDataList





def aggregateBH_list_oo():
   rootName5 = "/root/add/of/all/BH/scenarios/var5/oo/"
   rootName10 = "/root/add/of/all/BH/scenarios/var10/oo/"
   rootName15 = "/root/add/of/all/BH/scenarios/var15/oo/"
   rootName20 = "/root/add/of/all/BH/scenarios/var20/oo/"

   trainDataList = []
   testDataList = []
   
   trainDataList5 = []
   trainDataList10 = []
   trainDataList15 = []
   trainDataList20 = []
   testDataList5 = []
   testDataList10 = []
   testDataList15 = []
   testDataList20 = []

   
   all_folders = os.listdir(rootName5)
   # Filter out only folders (directories)
   all_folders = [item for item in all_folders if os.path.isdir(os.path.join(rootName5 , item))]
   scenario_folders = random.sample(all_folders, min(15, len(all_folders)))
   # Select scenarios for Test
   test_scenario_folders = [folder for folder in all_folders if folder not in scenario_folders]
      
   for l3 in scenario_folders:
      file_name = rootName5 + l3 + "/features_timeseries_60_sec.csv"
      myData = pd.read_csv(file_name, sep = ',')
      myData = myData.drop(['Unnamed: 0'],axis = 1)
      trainDataList5.append(myData)
         
   for l3 in test_scenario_folders:
      file_name = rootName5 + l3 + "/features_timeseries_60_sec.csv"
      myData = pd.read_csv(file_name, sep = ',')
      myData = myData.drop(['Unnamed: 0'],axis = 1)
      testDataList5.append(myData)

   all_folders = os.listdir(rootName10)
   # Filter out only folders (directories)
   all_folders = [item for item in all_folders if os.path.isdir(os.path.join(rootName10 , item))]
   scenario_folders = random.sample(all_folders, min(15, len(all_folders)))
   # Select scenarios for Test
   test_scenario_folders = [folder for folder in all_folders if folder not in scenario_folders]
      
   for l3 in scenario_folders:
      file_name = rootName10 + l3 + "/features_timeseries_60_sec.csv"
      myData = pd.read_csv(file_name, sep = ',')
      myData = myData.drop(['Unnamed: 0'],axis = 1)
      trainDataList10.append(myData)
         
   for l3 in test_scenario_folders:
      file_name = rootName10 + l3 + "/features_timeseries_60_sec.csv"
      myData = pd.read_csv(file_name, sep = ',')
      myData = myData.drop(['Unnamed: 0'],axis = 1)
      testDataList10.append(myData)
   
   all_folders = os.listdir(rootName15)
   # Filter out only folders (directories)
   all_folders = [item for item in all_folders if os.path.isdir(os.path.join(rootName15 , item))]
   scenario_folders = random.sample(all_folders, min(15, len(all_folders)))
   # Select scenarios for Test
   test_scenario_folders = [folder for folder in all_folders if folder not in scenario_folders]
      
   for l3 in scenario_folders:
      file_name = rootName15 + l3 + "/features_timeseries_60_sec.csv"
      myData = pd.read_csv(file_name, sep = ',')
      myData = myData.drop(['Unnamed: 0'],axis = 1)
      trainDataList15.append(myData)
         
   for l3 in test_scenario_folders:
      file_name = rootName15 + l3 + "/features_timeseries_60_sec.csv"
      myData = pd.read_csv(file_name, sep = ',')
      myData = myData.drop(['Unnamed: 0'],axis = 1)
      testDataList15.append(myData)

   all_folders = os.listdir(rootName20)
   # Filter out only folders (directories)
   all_folders = [item for item in all_folders if os.path.isdir(os.path.join(rootName20 , item))]
   scenario_folders = random.sample(all_folders, min(15, len(all_folders)))
   # Select scenarios for Test
   test_scenario_folders = [folder for folder in all_folders if folder not in scenario_folders]
      
   for l3 in scenario_folders:
      file_name = rootName20 + l3 + "/features_timeseries_60_sec.csv"
      myData = pd.read_csv(file_name, sep = ',')
      myData = myData.drop(['Unnamed: 0'],axis = 1)
      trainDataList20.append(myData)
         
   for l3 in test_scenario_folders:
      file_name = rootName20 + l3 + "/features_timeseries_60_sec.csv"
      myData = pd.read_csv(file_name, sep = ',')
      myData = myData.drop(['Unnamed: 0'],axis = 1)
      testDataList20.append(myData)

   testDataList = testDataList5 + testDataList10 + testDataList15 + testDataList20
   trainDataList = trainDataList5 + trainDataList10 + trainDataList15 + trainDataList20


   return trainDataList, testDataList




def aggregateBH_list_dec():
   rootName5 = "/root/add/of/all/BH/scenarios/var5/dec/"
   rootName10 = "/root/add/of/all/BH/scenarios/var10/dec/"
   rootName15 = "/root/add/of/all/BH/scenarios/var15/dec/"
   rootName20 = "/root/add/of/all/BH/scenarios/var20/dec/"

   trainDataList = []
   testDataList = []
   
   trainDataList5 = []
   trainDataList10 = []
   trainDataList15 = []
   trainDataList20 = []
   testDataList5 = []
   testDataList10 = []
   testDataList15 = []
   testDataList20 = []

   
   all_folders = os.listdir(rootName5)
   # Filter out only folders (directories)
   all_folders = [item for item in all_folders if os.path.isdir(os.path.join(rootName5 , item))]
   scenario_folders = random.sample(all_folders, min(15, len(all_folders)))
   # Select scenarios for Test
   test_scenario_folders = [folder for folder in all_folders if folder not in scenario_folders]
      
   for l3 in scenario_folders:
      file_name = rootName5 + l3 + "/features_timeseries_60_sec.csv"
      myData = pd.read_csv(file_name, sep = ',')
      myData = myData.drop(['Unnamed: 0'],axis = 1)
      trainDataList5.append(myData)
         
   for l3 in test_scenario_folders:
      file_name = rootName5 + l3 + "/features_timeseries_60_sec.csv"
      myData = pd.read_csv(file_name, sep = ',')
      myData = myData.drop(['Unnamed: 0'],axis = 1)
      testDataList5.append(myData)

   all_folders = os.listdir(rootName10)
   # Filter out only folders (directories)
   all_folders = [item for item in all_folders if os.path.isdir(os.path.join(rootName10 , item))]
   scenario_folders = random.sample(all_folders, min(15, len(all_folders)))
   # Select scenarios for Test
   test_scenario_folders = [folder for folder in all_folders if folder not in scenario_folders]
      
   for l3 in scenario_folders:
      file_name = rootName10 + l3 + "/features_timeseries_60_sec.csv"
      myData = pd.read_csv(file_name, sep = ',')
      myData = myData.drop(['Unnamed: 0'],axis = 1)
      trainDataList10.append(myData)
         
   for l3 in test_scenario_folders:
      file_name = rootName10 + l3 + "/features_timeseries_60_sec.csv"
      myData = pd.read_csv(file_name, sep = ',')
      myData = myData.drop(['Unnamed: 0'],axis = 1)
      testDataList10.append(myData)
   
   all_folders = os.listdir(rootName15)
   # Filter out only folders (directories)
   all_folders = [item for item in all_folders if os.path.isdir(os.path.join(rootName15 , item))]
   scenario_folders = random.sample(all_folders, min(15, len(all_folders)))
   # Select scenarios for Test
   test_scenario_folders = [folder for folder in all_folders if folder not in scenario_folders]
      
   for l3 in scenario_folders:
      file_name = rootName15 + l3 + "/features_timeseries_60_sec.csv"
      myData = pd.read_csv(file_name, sep = ',')
      myData = myData.drop(['Unnamed: 0'],axis = 1)
      trainDataList15.append(myData)
         
   for l3 in test_scenario_folders:
      file_name = rootName15 + l3 + "/features_timeseries_60_sec.csv"
      myData = pd.read_csv(file_name, sep = ',')
      myData = myData.drop(['Unnamed: 0'],axis = 1)
      testDataList15.append(myData)

   all_folders = os.listdir(rootName20)
   # Filter out only folders (directories)
   all_folders = [item for item in all_folders if os.path.isdir(os.path.join(rootName20 , item))]
   scenario_folders = random.sample(all_folders, min(15, len(all_folders)))
   # Select scenarios for Test
   test_scenario_folders = [folder for folder in all_folders if folder not in scenario_folders]
      
   for l3 in scenario_folders:
      file_name = rootName20 + l3 + "/features_timeseries_60_sec.csv"
      myData = pd.read_csv(file_name, sep = ',')
      myData = myData.drop(['Unnamed: 0'],axis = 1)
      trainDataList20.append(myData)
         
   for l3 in test_scenario_folders:
      file_name = rootName20 + l3 + "/features_timeseries_60_sec.csv"
      myData = pd.read_csv(file_name, sep = ',')
      myData = myData.drop(['Unnamed: 0'],axis = 1)
      testDataList20.append(myData)

   testDataList = testDataList5 + testDataList10 + testDataList15 + testDataList20
   trainDataList = trainDataList5 + trainDataList10 + trainDataList15 + trainDataList20


   return trainDataList, testDataList





def aggregate_list_base():
   trainDataList_BH, testDataList_BH = aggregateBH_list_base()
   trainDataList_DIS, testDataList_DIS = aggregateDIS_list_base()

   trainDataList = trainDataList_BH + trainDataList_DIS
   testDataList = testDataList_BH + testDataList_DIS

   return trainDataList, testDataList


def aggregate_list_oo():
   trainDataList_BH, testDataList_BH = aggregateBH_list_oo()
   trainDataList_DIS, testDataList_DIS = aggregateDIS_list_oo()

   trainDataList = trainDataList_BH + trainDataList_DIS
   testDataList = testDataList_BH + testDataList_DIS

   return trainDataList, testDataList


def aggregate_list_dec():
   trainDataList_BH, testDataList_BH = aggregateBH_list_dec()
   trainDataList_DIS, testDataList_DIS = aggregateDIS_list_dec()

   trainDataList = trainDataList_BH + trainDataList_DIS
   testDataList = testDataList_BH + testDataList_DIS

   return trainDataList, testDataList











def aggregateBH_list_shallow():
   rootName = "/root/add/of/all/BH/scenarios/"
   
   trainDataList = []
   testDataList = []
   
   size_folders = os.listdir(rootName)

   for l1 in size_folders:
      var_folders = os.listdir(rootName + l1)
      for l2 in var_folders:
         all_folders = os.listdir(rootName + l1 + "/" + l2)
         # Filter out only folders (directories)
         all_folders = [item for item in all_folders if os.path.isdir(os.path.join(rootName + l1 + "/" + l2 , item))]
         # Select scenarios for Train
         scenario_folders = random.sample(all_folders, min(15, len(all_folders)))
         # Select scenarios for Test
         test_scenario_folders = [folder for folder in all_folders if folder not in scenario_folders]

         

         for l3 in scenario_folders:
            file = open(rootName + "/" + l1 + "/" + l2 + "/" + l3 + "/depth.txt", "r")
            depth = file.readline()
            file.close()
            depth = int(depth)
            if (depth == 1 or depth == 2):
               file_name = rootName + l1 + "/" + l2 + "/" + l3 + "/features_timeseries_60_sec.csv"
               myData = pd.read_csv(file_name, sep = ',')
               myData = myData.drop(['Unnamed: 0'],axis = 1)
               trainDataList.append(myData)
         
         for l3 in test_scenario_folders:
            file = open(rootName + "/" + l1 + "/" + l2 + "/" + l3 +"/depth.txt", "r")
            depth = file.readline()
            file.close()
            depth = int(depth)
            if (depth == 1 or depth == 2):
               file_name = rootName + l1 + "/" + l2 + "/" + l3 + "/features_timeseries_60_sec.csv"
               myData = pd.read_csv(file_name, sep = ',')
               myData = myData.drop(['Unnamed: 0'],axis = 1)
               testDataList.append(myData)
         
    
   return trainDataList, testDataList



def aggregateDIS_list_shallow():
   rootName = "/root/add/of/all/DIS/scenarios/"
   
   trainDataList = []
   testDataList = []
   
   size_folders = os.listdir(rootName)

   for l1 in size_folders:
      var_folders = os.listdir(rootName + l1)
      for l2 in var_folders:
         all_folders = os.listdir(rootName + l1 + "/" + l2)
         # Filter out only folders (directories)
         all_folders = [item for item in all_folders if os.path.isdir(os.path.join(rootName + l1 + "/" + l2 , item))]
         # Select scenarios for Train
         scenario_folders = random.sample(all_folders, min(15, len(all_folders)))
         # Select scenarios for Test
         test_scenario_folders = [folder for folder in all_folders if folder not in scenario_folders]


         for l3 in scenario_folders:
            file = open(rootName + "/" + l1 + "/" + l2 + "/" + l3 +"/depth.txt", "r")
            depth = file.readline()
            file.close()
            depth = int(depth)
            if (depth == 1 or depth == 2):
               file_name = rootName + l1 + "/" + l2 + "/" + l3 + "/features_timeseries_60_sec.csv"
               myData = pd.read_csv(file_name, sep = ',')
               myData = myData.drop(['Unnamed: 0'],axis = 1)
               trainDataList.append(myData)
         
         for l3 in test_scenario_folders:
            file = open(rootName + "/" + l1 + "/" + l2 + "/" + l3 +"/depth.txt", "r")
            depth = file.readline()
            file.close()
            depth = int(depth)
            if (depth == 1 or depth == 2):
               file_name = rootName + l1 + "/" + l2 + "/" + l3 + "/features_timeseries_60_sec.csv"
               myData = pd.read_csv(file_name, sep = ',')
               myData = myData.drop(['Unnamed: 0'],axis = 1)
               testDataList.append(myData)
         
 

   return trainDataList, testDataList





def aggregateBH_list_medium():
   rootName = "/root/add/of/all/BH/scenarios/"
   
   trainDataList = []
   testDataList = []
   
   size_folders = os.listdir(rootName)

   for l1 in size_folders:
      var_folders = os.listdir(rootName + l1)
      for l2 in var_folders:
         all_folders = os.listdir(rootName + l1 + "/" + l2)
         # Filter out only folders (directories)
         all_folders = [item for item in all_folders if os.path.isdir(os.path.join(rootName + l1 + "/" + l2 , item))]
         # Select scenarios for Train
         scenario_folders = random.sample(all_folders, min(15, len(all_folders)))
         # Select scenarios for Test
         test_scenario_folders = [folder for folder in all_folders if folder not in scenario_folders]


         for l3 in scenario_folders:
            file = open(rootName + "/" + l1 + "/" + l2 + "/" + l3 +"/depth.txt", "r")
            depth = file.readline()
            file.close()
            depth = int(depth)
            if (depth == 3 or depth == 4 or depth == 5):
               file_name = rootName + l1 + "/" + l2 + "/" + l3 + "/features_timeseries_60_sec.csv"
               myData = pd.read_csv(file_name, sep = ',')
               myData = myData.drop(['Unnamed: 0'],axis = 1)
               trainDataList.append(myData)
         
         for l3 in test_scenario_folders:
            file = open(rootName + "/" + l1 + "/" + l2 + "/" + l3 +"/depth.txt", "r")
            depth = file.readline()
            file.close()
            depth = int(depth)
            if (depth == 3 or depth == 4 or depth == 5):
               file_name = rootName + l1 + "/" + l2 + "/" + l3 + "/features_timeseries_60_sec.csv"
               myData = pd.read_csv(file_name, sep = ',')
               myData = myData.drop(['Unnamed: 0'],axis = 1)
               testDataList.append(myData)

 

   return trainDataList, testDataList







def aggregateDIS_list_medium():
   rootName = "/root/add/of/all/DIS/scenarios/"
   trainDataList = []
   testDataList = []
   
   size_folders = os.listdir(rootName)

   for l1 in size_folders:
      var_folders = os.listdir(rootName + l1)
      for l2 in var_folders:
         all_folders = os.listdir(rootName + l1 + "/" + l2)
         # Filter out only folders (directories)
         all_folders = [item for item in all_folders if os.path.isdir(os.path.join(rootName + l1 + "/" + l2 , item))]
         # Select scenarios for Train
         scenario_folders = random.sample(all_folders, min(15, len(all_folders)))
         # Select scenarios for Test
         test_scenario_folders = [folder for folder in all_folders if folder not in scenario_folders]


         for l3 in scenario_folders:
            file = open(rootName + "/" + l1 + "/" + l2 + "/" + l3 +"/depth.txt", "r")
            depth = file.readline()
            file.close()
            depth = int(depth)
            if (depth == 3 or depth == 4 or depth == 5):
               file_name = rootName + l1 + "/" + l2 + "/" + l3 + "/features_timeseries_60_sec.csv"
               myData = pd.read_csv(file_name, sep = ',')
               myData = myData.drop(['Unnamed: 0'],axis = 1)
               trainDataList.append(myData)
         
         for l3 in test_scenario_folders:
            file = open(rootName + "/" + l1 + "/" + l2 + "/" + l3 +"/depth.txt", "r")
            depth = file.readline()
            file.close()
            depth = int(depth)
            if (depth == 3 or depth == 4 or depth == 5):
               file_name = rootName + l1 + "/" + l2 + "/" + l3 + "/features_timeseries_60_sec.csv"
               myData = pd.read_csv(file_name, sep = ',')
               myData = myData.drop(['Unnamed: 0'],axis = 1)
               testDataList.append(myData)
         
         

   return trainDataList, testDataList





def aggregateBH_list_deep():
   rootName = "/root/add/of/all/BH/scenarios/"
   trainDataList = []
   testDataList = []
   
   size_folders = os.listdir(rootName)

   for l1 in size_folders:
      var_folders = os.listdir(rootName + l1)
      for l2 in var_folders:
         all_folders = os.listdir(rootName + l1 + "/" + l2)
         # Filter out only folders (directories)
         all_folders = [item for item in all_folders if os.path.isdir(os.path.join(rootName + l1 + "/" + l2 , item))]
         # Select scenarios for Train
         scenario_folders = random.sample(all_folders, min(15, len(all_folders)))
         # Select scenarios for Test
         test_scenario_folders = [folder for folder in all_folders if folder not in scenario_folders]


         for l3 in scenario_folders:
            file = open(rootName + "/" + l1 + "/" + l2 + "/" + l3 +"/depth.txt", "r")
            depth = file.readline()
            file.close()
            depth = int(depth)
            if (depth >= 6):
               file_name = rootName + l1 + "/" + l2 + "/" + l3 + "/features_timeseries_60_sec.csv"
               myData = pd.read_csv(file_name, sep = ',')
               myData = myData.drop(['Unnamed: 0'],axis = 1)
               trainDataList.append(myData)
         
         for l3 in test_scenario_folders:
            file = open(rootName + "/" + l1 + "/" + l2 + "/" + l3 +"/depth.txt", "r")
            depth = file.readline()
            file.close()
            depth = int(depth)
            if (depth >= 6):
               file_name = rootName + l1 + "/" + l2 + "/" + l3 + "/features_timeseries_60_sec.csv"
               myData = pd.read_csv(file_name, sep = ',')
               myData = myData.drop(['Unnamed: 0'],axis = 1)
               testDataList.append(myData)
         
         

   return trainDataList, testDataList





def aggregateDIS_list_deep():
   rootName = "/root/add/of/all/DIS/scenarios/"
   trainDataList = []
   testDataList = []
   
   size_folders = os.listdir(rootName)

   for l1 in size_folders:
      var_folders = os.listdir(rootName + l1)
      for l2 in var_folders:
         all_folders = os.listdir(rootName + l1 + "/" + l2)
         # Filter out only folders (directories)
         all_folders = [item for item in all_folders if os.path.isdir(os.path.join(rootName + l1 + "/" + l2 , item))]
         # Select scenarios for Train
         scenario_folders = random.sample(all_folders, min(15, len(all_folders)))
         # Select scenarios for Test
         test_scenario_folders = [folder for folder in all_folders if folder not in scenario_folders]


         for l3 in scenario_folders:
            file = open(rootName + "/" + l1 + "/" + l2 + "/" + l3 +"/depth.txt", "r")
            depth = file.readline()
            file.close()
            depth = int(depth)
            if (depth >= 6):
               file_name = rootName + l1 + "/" + l2 + "/" + l3 + "/features_timeseries_60_sec.csv"
               myData = pd.read_csv(file_name, sep = ',')
               myData = myData.drop(['Unnamed: 0'],axis = 1)
               trainDataList.append(myData)
         
         for l3 in test_scenario_folders:
            file = open(rootName + "/" + l1 + "/" + l2 + "/" + l3 +"/depth.txt", "r")
            depth = file.readline()
            file.close()
            depth = int(depth)
            if (depth >=  6):
               file_name = rootName + l1 + "/" + l2 + "/" + l3 + "/features_timeseries_60_sec.csv"
               myData = pd.read_csv(file_name, sep = ',')
               myData = myData.drop(['Unnamed: 0'],axis = 1)
               testDataList.append(myData)

         

   return trainDataList, testDataList







def aggregate_list_shallow():
   trainDataList_BH, testDataList_BH = aggregateBH_list_shallow()
   trainDataList_DIS, testDataList_DIS = aggregateDIS_list_shallow()

   trainDataList = trainDataList_BH + trainDataList_DIS
   testDataList = testDataList_BH + testDataList_DIS

   return trainDataList, testDataList



def aggregate_list_medium():
   trainDataList_BH, testDataList_BH = aggregateBH_list_medium()
   trainDataList_DIS, testDataList_DIS = aggregateDIS_list_medium()

   trainDataList = trainDataList_BH + trainDataList_DIS
   testDataList = testDataList_BH + testDataList_DIS

   return trainDataList, testDataList


def aggregate_list_deep():
   trainDataList_BH, testDataList_BH = aggregateBH_list_deep()
   trainDataList_DIS, testDataList_DIS = aggregateDIS_list_deep()

   trainDataList = trainDataList_BH + trainDataList_DIS
   testDataList = testDataList_BH + testDataList_DIS

   return trainDataList, testDataList