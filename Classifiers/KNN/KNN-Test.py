
import os
import pickle
import numpy as np

filePath = './../../Data/Pickle/'
names = []

feat_label = []
filePath = './../../Data/Pickle/'
with (open(filePath + "feat_label_map", "rb")) as openfile:
    while True:
        try:
            feat_label.append(pickle.load(openfile))
        except EOFError:
            break

feat_label = feat_label[0]


getFileNumber = lambda x: int(x.split('_')[2])

getFeatLabelIndex = lambda x: (x - 1)*128

for filename in os.listdir(filePath + "flatten_features"):
    names.append(filename)

names.sort(key = lambda x: int(x.split('_')[2]))


trainDataFileSample = names[:10]

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
