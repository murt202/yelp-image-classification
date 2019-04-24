from sklearn.decomposition import IncrementalPCA
import numpy as np
import pickle as pkl
import glob

features_list = glob.glob('../Data/Pickle/flatten_features/*')[:100]
ipca = IncrementalPCA()
for i in range(0,len(features_list),2):
    if ((i*100)/len(features_list)) % 2 == 0:
        print((i*100)/len(features_list),"% completed")
    feat = np.vstack((pkl.load(open(features_list[i], 'rb')), pkl.load(open(features_list[i+1], 'rb'))))
    # print(feat.shape)
    ipca.partial_fit(feat)

print("Variance")
print(ipca.explained_variance_ratio_)
# print(ipca.explained_variance_ratio_.shape)
ind = []
s = 0
for i in range(feat.shape[0]):
    if s > 0.90:
        break
    s += ipca.explained_variance_ratio_[i]
    ind.append(i)
print(ind)
print(s)

# for f in features_list:
#     feat = pkl.load(open(f, 'rb'))
#     transformed_feat = ipca.transform(feat)
#     print(transformed_feat.shape)