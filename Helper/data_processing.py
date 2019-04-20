import numpy as np
import glob, os
import pickle as pkl

def flatten_features():
    feature_list = glob.glob('../Data/Pickle/features/*')
    for f in feature_list:
        flat_features = []
        features = pkl.load(open(f, 'rb'))
        for i in range(features.shape[0]):
             flat_features.append(features[i].flatten())
        flat_features = np.array(flat_features)
        with open('../Data/Pickle/flatten_features/'+os.path.basename(f), 'wb') as file:
            pkl.dump(flat_features, file)


if __name__ == "__main__":
    flatten_features()