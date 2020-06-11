"""
Code to get the feature vectors for the images from pre-trained CNN
"""
import numpy as np
from cv2 import resize
import os
import cv2
import time
import pickle
from keras.models import Sequential, Model
from classification_models.keras import Classifiers
from keras.preprocessing import image
from keras.models import load_model

ResNet34, preprocess_input = Classifiers.get('resnet34')
k_model = ResNet34(input_shape=(224,224,3), weights='imagenet')
model = Model(inputs = k_model.input, outputs = k_model.get_layer('pool1').output)

path_to_train = '*/data/train'
path_to_test = '*/data/test'

with open(path_to_train, 'rb') as fo:
    train_batch = pickle.load(fo, encoding='latin1')

with open(path_to_test, 'rb') as fo:
    test_batch = pickle.load(fo, encoding='latin1')

train_images = train_batch['data'].reshape((len(train_batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
train_labels = train_batch['fine_labels']
test_images = test_batch['data'].reshape((len(test_batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
test_labels = test_batch['fine_labels']

categories = dict()
total_classes=100
total_num=[0]*total_classes
train_features = []
test_features = []

for i in range(0,len(train_images)):
    print (i)
    total_num[train_labels[i]]+=1
    img = resize(train_images[i],(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    features_np=np.array(features)
    features_f = features_np.flatten()
    train_features.append(features_f)

for i in range(0,len(test_images)):
    print ('test',i)
    img = resize(test_images[i],(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    features_np=np.array(features)
    features_f = features_np.flatten()
    test_features.append(features_f)

train_features = np.array(train_features)
test_features = np.array(test_features)

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

with open('CIFAR_Resnet18_train_features.data', 'wb') as filehandle:
    pickle.dump(train_features, filehandle)
with open('CIFAR_Resnet18_test_features.data', 'wb') as filehandle:
    pickle.dump(test_features, filehandle)
with open('CIFAR_Resnet18_train_labels.data', 'wb') as filehandle:
    pickle.dump(train_labels, filehandle)
with open('CIFAR_Resnet18_test_labels.data', 'wb') as filehandle:
    pickle.dump(test_labels, filehandle)
