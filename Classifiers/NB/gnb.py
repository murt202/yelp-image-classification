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
for name in names:
    file = open("C:/Projects/SML project/flatten_features/flatten_features/"+name, "rb")
    print(name)
    data.append(pickle.load(file))
    break;
names = []

def test_train_split(data, percent):
    data_array = data.pop(0)
    total = len(data_array)
    print(data_array.shape)
    trainsize = round(total * percent)
#    testsize = round(total-trainsize)
    traindata = data_array[:trainsize,:]
    testdata = data_array[trainsize:,:]
    return traindata,testdata
    
traindata,testdata = test_train_split(data, 0.8)

df = pd.DataFrame(traindata)
dftest = pd.DataFrame(testdata)
#traindata = []
#testdata = []
#print(df.describe)
it = objects[0]
#print(it[0])
#testX = dftest.iloc[:,:].values.tolist()

lab=[]
testLab = []
for i in range(102):
    lab.append(it[i])

for i in range(26):
    testLab.append(it[i+102])
    
it = []
#objects = []
    
label = pd.DataFrame(lab, columns = ['C0','C1','C2','C3','C4','C5','C6','C7','C8'])


#c0_mean = df.loc[label['C0'] == 0].mean()
#c01_mean = df.loc[label['C0'] == 1].mean()
#c0_mean = df.loc[label['C0'] == 0].mean()
#c0_cov = df.loc[label['C0'] == 0].cov().values
#df = []
#dftest = []
#p0 = multivariate_normal.pdf(testX[1],mean = c0_mean, cov = c0_cov)
#print(p0)
#c1_mean = df.loc[label['C0'] == 1].mean()
#c1_cov = df.loc[label['C0'] == 1].cov().values
#
#j = -1
#count = 0
#total = 26
#k = 0
#for i in testX:
#    print(i)
#    if j==-1:
#        j+=1
#        continue
#    p0 = multivariate_normal.pdf(i, mean = c0_mean, cov = c0_cov)
#    print(k+''+p0)
#    k = k+1
#    p1 = multivariate_normal.pdf(i, mean = c1_mean, cov = c1_cov)
#    if(p0 > p1):
#        if testLab[j] == 0 : count+=1
#    else:
#        if testLab[j] == 1 : count+=1
#    j+=1
#print(count/total)

#print(np.shape(c0_cov))
for i in range(25):
#    print(label['C0'])
    testVal = pd.DataFrame(dftest.iloc[i]).T
    clf = GaussianNB()
    clf.fit(df,label['C2'])
    print(clf.predict(testVal))
    print('')
    print(testLab[i])
    print('')
