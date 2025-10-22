#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import re
import time
import coloredPrinting as pr



"""
The depthOfNodes() function, finds the maximum depth of the RPL topoly in the given network. 
The input file is the mote_output.log file of a Cooja scenario.
Other functions are auxiliary funtions for the depthOfNodes() function.
"""



def allPackets(fileName):
    logs = pd.read_csv(fileName, sep = '\t')
    return(logs)

def incomingUDPtoSink(fileName):
    logs = pd.read_csv(fileName, sep = '\t')
    x = logs[logs['message'].str.contains('Incoming UDP', na=False)]
    x = x[x['mote'] == 1]
    x.reset_index(drop=True,inplace = True)
    return(x)

def incomingICMPtoSink(fileName):
    logs = pd.read_csv(fileName, sep = '\t')
    x = logs[logs['message'].str.contains('Incoming ICMP6', na=False)]
    x = x[x['mote'] == 1]
    x.reset_index(drop=True,inplace = True)
    return(x)

def incomingToSink(fileName):
    logs = pd.read_csv(fileName, sep = '\t')
    x = logs[logs['message'].str.contains('Incoming', na=False)]
    x = x[x['mote'] == 1]
    x.reset_index(drop=True,inplace = True)
    return(x)

"""

def readLog(fileName):
    # read the log file
    logFile = open(fileName)
    read = logFile.read()

    # return the cursor to the beginning of file
    logFile.seek(0)

    # creat an empty array 
    logs = []

    # count the number of lines in the file

    line = 0
    for word in read:
        if (word == '\n'):
    	    line += 1
    	

    # read the log file line by line
    for i in range(line):
        logs.append(logFile.readline())
    
    print(len(logs))

"""

def findLinesContainWord(word, linesList):
    containingLines = []
    for i in range(len(linesList)):
        print(i)








def depthOfNodes(moteFile):
    allLogs = allPackets(fileName = moteFile)
    
    depth = []
    for i in range(allLogs.shape[0]):
        # ,,,,,,,,,,,,,,,,,,,,,,,,,,,, if the sink node receives a packet
        if (allLogs.loc[i,'mote'] == 1 and 'Received  message' in allLogs.loc[i,'message']):
            # ,,,,,,,,,,,,,,,,,,,,,,,,,,,, find the packet's sender
            keyword = 'from'
            before_keyword, keyword, after_keyword = allLogs.loc[i,'message'].partition(keyword)
            sender = int(after_keyword.split(':')[-1],16)
            keyword2 = 'message'
            before_keyword2, keyword2, after_keyword2 = before_keyword.partition(keyword2)
            rcvSQ = int(after_keyword2.split(' ')[1])
            # ----------------------------
            # ,,,,,,,,,,,,,,,,,,,,,,,,,,,, find the original packet when it was contstructed in the IoT node: find based on the (sender, having 'Sending request' and the same sq number) 
            #
            j = i 
            
            while (('Sending request' not in allLogs.loc[j,'message']) or (int(allLogs.loc[j,'mote']) != sender)):
                j -= 1
            j += 1
            if ((allLogs.loc[j,'mote'] == sender)  and (int(allLogs.loc[j,'message'].split(',')[0].split(':')[3]) == rcvSQ)):
                if (int(allLogs.loc[j,'message'].split(',')[0].split(':')[3]) == 31): 
                    k = j
                    hop = 1
                    while (k < i):
                        if ('Forwarding' in allLogs.loc[k,'message']):
                            keyword3 = 'from'
                            before_keyword3, keyword3, after_keyword3 = allLogs.loc[k,'message'].partition(keyword3)
                            sender3 = int(after_keyword3.split(':')[-1],16)
                            if (sender3 == sender):
                                hop += 1
                        k += 1
                    
                    depth.append(hop)    
                    
    return(max(depth))





