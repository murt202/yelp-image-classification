# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 18:29:05 2019

@author: arvin
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 16:37:31 2019

@author: arvin
"""
# helper file to read data
import pandas as pd
import os
import pickle
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.naive_bayes import GaussianNB
TOTAL_IMAGES = 1830
CLASS_LABELS = ['C0','C1','C2','C3','C4','C5','C6','C7','C8']

objects = []
with (open("C:/Projects/SML project/flatten_features/feat_label_map", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break

names = []
for filename in os.listdir("C:/Projects/SML project/flatten_features/flatten_features"):
    names.append(filename)
    
names.sort(key = lambda x: int(x.split('_')[2]))

data = []
i = TOTAL_IMAGES
length_iterated = 0
label = []
allLabels = objects[0]


def test_train_split(data, percent):
    data_array = data.pop(0)
    total = len(data_array)
    print(data_array.shape)
    trainsize = round(total * percent)
#    testsize = round(total-trainsize)
    traindata = data_array[:trainsize,:]
    testdata = data_array[trainsize:,:]
    return traindata,testdata


#1.Open a file
#2.Get all the rows
#3.Get label for each row
#4.Give the label and data to partial fit

dftrain = pd.DataFrame()
dftrain = dftrain.fillna(0) # with 0s rather than NaNs
labelTrain = pd.DataFrame()
labelTrain = dftrain.fillna(0)

clf0 = GaussianNB()
clf1 = GaussianNB()
clf2 = GaussianNB()
clf3 = GaussianNB()
clf4 = GaussianNB()
clf5 = GaussianNB()
clf6 = GaussianNB()
clf7 = GaussianNB()
clf8 = GaussianNB()


testedCount = 0
correctCount = [0,0,0,0,0,0,0,0,0]

clfs = [clf0,clf1,clf2,clf3,clf4,clf5,clf6,clf7,clf8]

no_of_test_files = 500

for name in names:
    file = open("C:/Projects/SML project/flatten_features/flatten_features/"+name, "rb")
    print(name) 
    data = []
    label = []
    data.append(pickle.load(file))
    traindata, testdata = test_train_split(data,1)

    dftrain = dftrain.iloc[0:0] # Empty a dataframe
    dftrain = pd.DataFrame(traindata)
    for labelIndex in range(length_iterated, length_iterated+len(traindata)):
        label.append(allLabels[labelIndex])
    labelTrain = labelTrain.iloc[0:0]
    labelTrain = pd.DataFrame(label,columns = CLASS_LABELS )
    length_iterated = length_iterated+len(traindata)
    print(dftrain.shape)
    if i <= no_of_test_files:     
        for testIndex in range(len(traindata)):
            testVal = pd.DataFrame(dftrain.iloc[testIndex]).T
            testedCount +=1
            for classifierId in range(9):
                if clfs[classifierId].predict(testVal) == label[testIndex][1]:
                    correctCount[classifierId] += 1
    else :
        clf0.partial_fit(dftrain,labelTrain['C0'], classes =[0,1])    
        clf1.partial_fit(dftrain,labelTrain['C1'], classes =[0,1])
        clf2.partial_fit(dftrain,labelTrain['C2'], classes =[0,1])    
        clf3.partial_fit(dftrain,labelTrain['C3'], classes =[0,1])    
        clf4.partial_fit(dftrain,labelTrain['C4'], classes =[0,1])    
        clf5.partial_fit(dftrain,labelTrain['C5'], classes =[0,1])    
        clf6.partial_fit(dftrain,labelTrain['C6'], classes =[0,1])    
        clf7.partial_fit(dftrain,labelTrain['C7'], classes =[0,1])    
        clf8.partial_fit(dftrain,labelTrain['C8'], classes =[0,1])    
    i -= 1
    
for itr in range(9):
    print(correctCount[itr]/testedCount)
