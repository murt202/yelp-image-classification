"""
Created on Sat Apr 20 13:25:51 2019

@author: ajay
"""
import pandas as pd
import pickle
from sklearn.model_selection import KFold
import numpy as np

def read_labels( index ):
    labels = []
    with (open("feat_label_map", "rb")) as openfile:
        while True:
            try:
                labels.append(pd.DataFrame(pickle.load(openfile)))
            except EOFError:
                break
    
    class_label = labels[0]
    st = index*128
    if(index == 1835):
        return class_label.iloc[:, st : st+90]
    return class_label.iloc[:, st : st+128]
         
def read_data( index ):
    file = open("flatten_features/feat_vgg16_" + str(index) , "rb")
    data = pd.DataFrame(pickle.load(file))
    return data

def training(train_indices):
    
    weights = []
    for i in range(9):
        weights.append(np.zeros((1,25089))[0]) 
        
    for index in train_indices:
        index += 1
        file = read_data(index)
        label = read_labels(index)
        label.reset_index(drop = True)
        print("read_data")
        for x in range(len(label)):
            file["label"] = label.iloc[x,:]
            t_data = file.iloc[:,:-1]
            t_label = file.iloc[:,-1]
            
            dim = t_data.shape[1] + 1 
            max_itr = 20
            n = 0.1
                
            for itr in  range(max_itr):
                w_x = pd.Series(np.dot(t_data.values,weights[x][1:]))
                sigmoid_value = 1/(1+np.exp(-(w_x + weights[x][0])))
                error = t_label - sigmoid_value
                weights[x][0] += n*sum(error)
                
                for i in range(dim - 1):
                    grad = t_data.iloc[:,i] * error 
                    weights[x][i+1] += n*sum(grad)
         
    return weights

def testing( test_indices , weights):
    confusion = np.zeros((2,2))
    for index in test_indices:
        index += 1
        label = read_labels(index)
        df = read_data(index)
        for x in range(len(label)):
            df["label"] = label.iloc[x,:].reset_index(drop = True)
            xTest = df.iloc[:,:-1]
            yTest = df.iloc[:,-1]
            
            test_w_x = pd.Series(np.dot(xTest.values,weights[x][1:]))
            sigmoid = 1/(1+np.exp(-(test_w_x + weights[x][0])))
            
            pred = ((sigmoid > 0.5) * (1))
                
            for i in range(len(pred)):
                confusion[int(yTest[i])][pred[i]] += 1
    
    tp = confusion[0][0]
    fn = confusion[1][1]
    accuracy = (tp + fn) / np.sum(confusion)
    return accuracy

if __name__ == "__main__":
       
    split = np.zeros(1835)
    kfold = KFold(3, True, 1)
    for train, test in kfold.split(split):
        print(test , train)
        weights = training(train)
        print(testing(test,weights))
    
