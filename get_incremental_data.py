"""
To divide data into increments
"""
import numpy as np
from copy import deepcopy
import pickle
import math
import random

class getIncrementalData:
    def __init__(self,train_features,train_labels,test_features,test_labels,full_classes,seed):
        self.train_features = train_features
        self.train_labels = train_labels
        self.test_features = test_features
        self.test_labels = test_labels
        self.full_classes = full_classes
        self.train_features_increment = None
        self.train_labels_increment = None
        self.test_features_increment = None
        self.test_labels_increment = None
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def initialize(self,train_features,train_labels,test_features,test_labels,full_classes,seed):
        self.train_features = train_features
        self.train_labels = train_labels
        self.test_features = test_features
        self.test_labels = test_labels
        self.full_classes = full_classes
        self.train_features_increment = None
        self.train_labels_increment = None
        self.test_features_increment = None
        self.test_labels_increment = None
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def incremental_data(self,total_classes=2,limiter=None):
        orig_lab = [i for i in range(0,self.full_classes)]
        random.shuffle(orig_lab)
        shuf_train_labels = deepcopy(self.train_labels)
        for i in range(0,len(shuf_train_labels)):
            shuf_train_labels[i] = orig_lab[self.train_labels[i]]
        indices_sort_train = np.argsort(shuf_train_labels)

        shuf_test_labels = deepcopy(self.test_labels)
        for i in range(0,len(shuf_test_labels)):
            shuf_test_labels[i] = orig_lab[self.test_labels[i]]
        indices_sort_test = np.argsort(shuf_test_labels)

        shuf_train_features = self.train_features[indices_sort_train[::1]]
        shuf_train_labels = shuf_train_labels[indices_sort_train[::1]]

        shuf_test_features = self.test_features[indices_sort_test[::1]]
        shuf_test_labels = shuf_test_labels[indices_sort_test[::1]]

        train_features = shuf_train_features
        test_features = shuf_test_features
        train_labels = shuf_train_labels
        test_labels = shuf_test_labels

        # Divide data into increments of 2, 5, 10, 20 classes
        if limiter!=None:
            self.full_classes = limiter
        self.train_features_increment = [[] for x in range(0,int(self.full_classes/total_classes))]
        self.train_labels_increment = [[] for x in range(0,int(self.full_classes/total_classes))]
        self.test_features_increment = [[] for x in range(0,int(self.full_classes/total_classes))]
        self.test_labels_increment = [[] for x in range(0,int(self.full_classes/total_classes))]
        for increment in range(0,int(self.full_classes/total_classes)):
            for i in range(0+(increment*total_classes),total_classes+(increment*total_classes)):
                indices = np.where(train_labels==i)
                for j in indices[0]:
                    self.train_features_increment[increment].append(train_features[j])
                    self.train_labels_increment[increment].append(train_labels[j])
                indices = np.where(test_labels==i)
                for j in indices[0]:
                    self.test_features_increment[increment].append(test_features[j])
                    self.test_labels_increment[increment].append(test_labels[j])
