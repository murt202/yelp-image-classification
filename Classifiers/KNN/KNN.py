import numpy as np
# import pandas as pd
import math
from sklearn.metrics.pairwise import euclidean_distances
# from matplotlib import pyplot as plt
import os
import pickle


def log(strd):
    print()
    print(strd)
    print()

getFileNumber = lambda x: int(x.split('_')[2])

getFeatLabelIndex = lambda x: (x - 1)*128

def test(x_train, y_train, x_test, y_test, k):
    distance = euclidean_distances(x_test, x_train)
    sorted_distance = np.asarray([ sorted(each)[:k] for each in distance])
    min_dist_index = [i.argsort()[:k] for i in distance]
    min_dist_labels = [[y_train[j] for j in i[:k]] for i in min_dist_index]
    min_dist_data = [[x_train[j] for j in i[:k]] for i in min_dist_index]
    return min_dist_data, min_dist_labels, sorted_distance

feat_label = []
filePath = './../../Data/Pickle/'
with (open(filePath + "feat_label_map", "rb")) as openfile:
    while True:
        try:
            feat_label.append(pickle.load(openfile))
        except EOFError:
            break

feat_label = feat_label[0]

names = []
for filename in os.listdir(filePath + "flatten_features"):
    names.append(filename)

names.sort(key = lambda x: int(x.split('_')[2]))

testDataFileSample = names[1300:]
trainDataFileSample = names[:1300]

data = []
for name in testDataFileSample:
    file = open(filePath + "flatten_features/"+name, "rb")
    fileData = pickle.load(file)
    data.extend(fileData)

test_data_length = len(data)
log('Test data length ' + str(test_data_length))
x_test = np.asarray(data)
y_test = []


for i in range(len(x_test)):
    y_test.append(feat_label[getFeatLabelIndex(getFileNumber(testDataFileSample[0])) +  i])

log('labels for test data are loaded')

total_sample = len(trainDataFileSample) * 128
log('Training sample ' + str(total_sample))
k = int((math.sqrt(total_sample))/2)
if(not k % 2):
    k += 1


log('The value of k '+ str(k))

def getData(index, k, newData, currentData):
    if(index < k):
        # print(index, k, len(currentData))
        return currentData[index]
    else:
        return newData[index - k]

final_k_label = []
final_sorted_dist = []
for fileIndex in range(0, len(trainDataFileSample), 3):
    x_temp = []
    y_train = []
    for file in trainDataFileSample[fileIndex: fileIndex+3]:
        print('processing ', file)
        f = pickle.load(open(filePath + "flatten_features/"+file, "rb"))
        x_temp.extend(f)
        for i in range(len(f)):
            y_train.append(feat_label[getFeatLabelIndex(getFileNumber(file)) +  i])
    x_train = np.asarray(x_temp)
    k_data, k_label, dist= test(x_train, y_train, x_test, y_test, k)
    temp_dist = []
    if (len(final_k_label)):
        for c in range(len(dist)):
            current = list(final_sorted_dist[c])
            current.extend(list(dist[c]))
            temp_dist.append(np.asarray(current))
        newKDist = [j.argsort()[:k] for j in temp_dist]
        for each in range(len(newKDist)):
            final_k_label[each] = [getData(j, k, k_label[each], final_k_label[each]) for j in newKDist[each]]
            # final_k_data[each] = [getData(j, k, k_data[each], final_k_data[each]) for j in newKDist[each]]
        final_sorted_dist = np.asarray([ sorted(each)[:k] for each in temp_dist])
    else:
        # final_k_data.extend(k_data)
        final_k_label.extend(k_label)
        final_sorted_dist.extend(dist)

log('K-closest value for each test data is calculated')
# print(final_k_label[0])
correctness_count = []
for j in range(9):
    print('Predicting '+str(j)+' feature')
    count = 0
    for i in range(len(final_k_label)):
        currentLabel = list(map(lambda x: x[j], final_k_label[i]))
        label = max(set(currentLabel), key = currentLabel.count)
        if (label == y_test[i][j]):
            count +=1
    correctness_count.append(count)

accuracy_count =  list(map(lambda x: (x / test_data_length) * 100, correctness_count))
log(accuracy_count)

    # label = [max(set(map(lambda x: x[j], i)), key = i.count) for i in final_k_label]
    # label = np.asarray(label)
    # print(label)
    # error = 1 - np.sum(label == y_test)/y_test.shape[0]

# label = np.asarray(label)
# error = 1 - np.sum(label == y_test)/y_test.shape[0]





    # test_error.append(test_error_avg)

# plt.plot(k_values, train_error, 'g-', label='Train Error')
# plt.plot(k_values, test_error, 'r-', label='Test Error')
# plt.title('Train and Test error vs K for KNN')
# plt.xlabel('K')
# plt.ylabel('Error')
# plt.legend()
# plt.grid()
# plt.show()
