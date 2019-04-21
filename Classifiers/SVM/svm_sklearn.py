import numpy as np
from sklearn import svm
import pickle as pkl

data = pkl.load(open('../../Data/Pickle/flatten_features/feat_vgg16_1', 'rb'))
labels = pkl.load(open('../../Data/Pickle/feat_label_map', 'rb'))
labels = np.array([labels[key] for key in labels.keys() if key in range(128)])
labels = labels[:, 0]

classifier = svm.SVC(kernel='linear', C=0.5)
classifier.fit(data, labels)

labels_predict = classifier.predict(data)
print(labels_predict)
print(labels)
print("Accuracy: ", 1 - (np.sum(abs(labels_predict - labels))/labels.shape[0]))
