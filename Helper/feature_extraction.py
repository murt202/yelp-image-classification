from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
import numpy as np
import glob
import pickle as pkl
import os
import pandas as pd

def features_extract():
	# model = VGG16(weights='imagenet', include_top=False)
	count = 0
	count_2 = 0
	total = len(glob.glob('../Data/train/train_photos/*.jpg'))
	feat = []
	images = []
	print(total)

	for filename in path_list:
		if count % 128 == 0 and count != 0:
			count_2 += 1
			images = np.array(images)
			feat.append(model.predict(images, batch_size=32, verbose=1))
			images = []
			feat = np.array(feat).reshape((1,-1, 7, 7, 512))
			feat = np.squeeze(feat, axis=0)
			if count_2 % 50 == 0:
				print("Feature batch ",count_2)
			file = open('../Data/Pickle/feat_vgg16_'+str(count_2), 'wb')
			pkl.dump(feat, file)
			feat = []

		img = image.load_img(filename, target_size=(224, 224))
		x = image.img_to_array(img)
		x = preprocess_input(x)
		images.append(x)
		count += 1
		if (count % 1000 == 0):
			print(count*100 / total, "% completed")

	# print("count ",count)
	# print(np.array(feat).shape)
	count_2 += 1
	images = []
	feat = []
	for filename in path_list[total - (total % 128)-1:-1]:
		img = image.load_img(filename, target_size=(224, 224))
		x = image.img_to_array(img)
		x = preprocess_input(x)
		images.append(x)
		count += 1
		if (count % 1000 == 0):
			print(count*100 / total, "% completed")
	images = np.array(images)
	# print(images.shape)
	a = model.predict(images, batch_size=32, verbose=1)
	print("Last batch: ",np.array(a).shape)
	# feat = np.vstack((feat, a))
	# print("Features: ",feat.shape)
	# feat = np.squeeze(np.array(feat), axis=0)
	# print(feat.shape)
	file = open('../Data/Pickle/feat_vgg16_'+str(count_2), 'wb')
	pkl.dump(a, file)

def labels_extract():
	bus_label_map = {}
	bus_label_data = pd.read_csv('../Data/train/train.csv')
	bus_label_data['labels'] = bus_label_data['labels'].apply(lambda x: str(x))
	for index, row in bus_label_data.iterrows():
		ind = []
		value = np.zeros(9)
		if row['labels'] != 'nan':
			ind = [int(e) for e in row['labels'].split()]
		value[ind] = 1
		bus_label_map[row['business_id']] = value
	with open('../Data/Pickle/bus_label_map', 'wb') as file:
		pkl.dump(bus_label_map, file)
	return bus_label_map

def map_images_to_business():
	imgs_bus_map = {}
	imgs_bus_data = pd.read_csv('../Data/train/train_photo_to_biz_ids.csv')
	for index, row in imgs_bus_data.iterrows():
		imgs_bus_map[row['photo_id']] = row['business_id']
	with open('../Data/Pickle/imgs_bus_map', 'wb') as file:
		pkl.dump(imgs_bus_map, file)
	return imgs_bus_map

def map_images_to_feature():
	imgs_feat_map = {}
	path_list = glob.glob('../Data/train/train_photos/*.jpg')
	imgs_list = [os.path.basename(p).split('.')[0] for p in path_list]
	for i in range(len(imgs_list)):
		imgs_feat_map[int(imgs_list[i])] = i 
	with open('../Data/Pickle/imgs_feat_map','wb') as file:
		pkl.dump(imgs_feat_map, file)
	return imgs_feat_map

def map_features_to_labels():
	feat_label_map = {}
	imgs_feat_map = map_images_to_feature()
	imgs_bus_map = map_images_to_business()
	bus_label_map = labels_extract()

	for imgs, feat in imgs_feat_map.items():
		value = bus_label_map[imgs_bus_map[imgs]]
		feat_label_map[feat] = value
	with open('../Data/Pickle/feat_label_map','wb') as file:
		pkl.dump(feat_label_map, file)



if __name__ == "__main__":
	# features_extract()
	map_features_to_labels()