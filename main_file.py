"""
Complete CBCL code
"""

import numpy as np
from copy import deepcopy
import pickle
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
from Functions import get_centroids
from Functions import check_reduce_centroids
from Functions import get_test_accuracy
from get_incremental_data import getIncrementalData
import random

seed = random.randint(0,1000)
random.seed(seed)
np.random.seed(seed)

with open('CIFAR_Resnet34_train_features.data', 'rb') as filehandle:
    train_features = pickle.load(filehandle)
with open('CIFAR_Resnet34_test_features.data', 'rb') as filehandle:
    test_features = pickle.load(filehandle)
with open('CIFAR_Resnet34_train_labels.data', 'rb') as filehandle:
    train_labels = pickle.load(filehandle)
with open('CIFAR_Resnet34_test_labels.data', 'rb') as filehandle:
    test_labels = pickle.load(filehandle)

distance_metric = 'euclidean'
clustering_type = 'Agg_Var'
full_classes = 100
total_classes = 2
k_shot = 10
total_centroids_limit = 7500
current_total_centroids = 0
distance_threshold = 17
k = 1

for iterations in range(0,1):
    # get incremental data
    incremental_data_creator = getIncrementalData(train_features,train_labels,test_features,test_labels,full_classes=full_classes,seed=seed)
    incremental_data_creator.incremental_data(total_classes=total_classes,limiter=full_classes)
    train_features_increment = incremental_data_creator.train_features_increment
    train_labels_increment = incremental_data_creator.train_labels_increment
    test_features_increment = incremental_data_creator.test_features_increment
    test_labels_increment = incremental_data_creator.test_labels_increment

    complete_x_test = []
    complete_y_test = []
    complete_centroids = []
    complete_centroid_labels = []
    total_num = []
    full_total_num = []

    # for the complete number of increments cluster and test
    for increment in range(0,int(100/total_classes)):
        total_num.extend([0 for y in range(total_classes)])
        x_test = test_features_increment[increment]
        y_test = test_labels_increment[increment]

        # get some random k_shot images for each class
        x_train_2,x_test_2,y_train_2,y_test_2 = train_test_split(train_features_increment[increment],train_labels_increment[increment],test_size=1)
        total_num_temp = [0 for x in range(0,total_classes)]
        x_train_increment = []
        y_train_increment = []
        for i in range(0,len(y_train_2)):
            if total_num_temp[y_train_2[i]-(increment*total_classes)]<k_shot:
                total_num_temp[y_train_2[i]-(increment*total_classes)]+=1
                x_train_increment.append(x_train_2[i])
                y_train_increment.append(y_train_2[i])
        print ('number of training images',len(y_train_increment))

        ### CLUSTERING PHASE ###
        x_train_increment = train_features_increment[increment]
        y_train_increment = train_labels_increment[increment]
        train_data = [[] for y in range(total_classes)]
        for i in range(0,len(y_train_increment)):
            train_data[y_train_increment[i]-(increment*total_classes)].append(x_train_increment[i])
            total_num[y_train_increment[i]]+=1
        weighting = np.divide([1 for x in range(0,(len(total_num)))],total_num)
        weighting = np.divide(weighting,np.sum(weighting))

        centroids = [[[0 for x in range(len(x_train_increment[0]))]] for y in range(total_classes)]
        train_data = [[] for y in range(total_classes)]
        for i in range(0,len(y_train_increment)):
            train_data[y_train_increment[i]-(increment*total_classes)].append(x_train_increment[i])

        # for multiprocessing
        train_pack = []
        for i in range(0,total_classes):
            train_pack.append([train_data[i],distance_threshold,clustering_type])
        my_pool = Pool(total_classes)
        centroids = my_pool.map(get_centroids,train_pack)
        my_pool.close()
        exp_centroids = 0
        for i in range(0,len(centroids)):
            exp_centroids+=len(centroids[i])

        # remove centroids to keep total centroids in limit
        complete_centroids = check_reduce_centroids(complete_centroids,current_total_centroids,exp_centroids,total_centroids_limit,increment,total_classes)
        complete_centroids.extend(centroids)
        total_centroids = 0
        for i in range(0,len(complete_centroids)):
            total_centroids+=len(complete_centroids[i])
        current_total_centroids = total_centroids

        ### TESTING PHASE ###
        complete_x_test.extend(x_test)
        complete_y_test.extend(y_test)
        test_pack = []

        test_pack=[complete_x_test,complete_y_test,complete_centroids,k,total_classes+(increment*total_classes),weighting]
        test_accuracy = get_test_accuracy(test_pack)
        accuracies = []
        predicted_label = []
        print ("test_accuracy", test_accuracy)
