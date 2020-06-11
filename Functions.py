"""
Helper functions
"""
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance
from copy import deepcopy
import math
from multiprocessing import Pool
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fclusterdata
from scipy.cluster.hierarchy import fcluster, ward, average, weighted, complete, single
from scipy.spatial.distance import pdist

distance_metric = 'euclidean'
def find_distance(data_vec,centroid,distance_metric):
    if distance_metric=='euclidean':
        return np.linalg.norm(data_vec-centroid)
    elif distance_metric == 'euclidean_squared':
        return np.square(np.linalg.norm(data_vec-centroid))
    elif distance_metric == 'cosine':
        return distance.cosine(data_vec,centroid)

# different clustering types
def get_centroids(train_pack):
    # unpack x_train
    x_train = train_pack[0]
    distance_threshold = train_pack[1]
    clustering_type = train_pack[2]

    if clustering_type == 'Agglomerative':
        dist_mat=pdist(x_train,metric='euclidean')
        Z = weighted(dist_mat)
        dn = hierarchy.dendrogram(Z)
        labels=fcluster(Z, t=distance_threshold, criterion='distance')

        total_number = [0 for x in range(0,max(labels))]
        centroids = [[0 for x in range(len(x_train[0]))] for y in range(0,max(labels))]
        for j in range(0,len(x_train)):
            centroids[labels[j]-1]+=x_train[j]
            total_number[labels[j]-1]+=1
        for j in range(0,len(centroids)):
            centroids[j] = np.divide(centroids[j],total_number[j])

    elif clustering_type == 'Agg_Var':
        if len(x_train)>0:
            centroids = [[0 for x in range(len(x_train[0]))]]
            # initalize centroids
            centroids[0] = x_train[0]
            total_num = [1]
            for i in range(1,len(x_train)):
                distances=[]
                indices = []
                for j in range(0,len(centroids)):
                    d = find_distance(x_train[i],centroids[j],distance_metric)
                    if d<distance_threshold:
                        distances.append(d)
                        indices.append(j)
                if len(distances)==0:
                    centroids.append(x_train[i])
                    total_num.append(1)
                else:
                    min_d = np.argmin(distances)
                    centroids[indices[min_d]] = np.add(np.multiply(total_num[indices[min_d]],centroids[indices[min_d]]),x_train[i])
                    total_num[indices[min_d]]+=1
                    centroids[indices[min_d]] = np.divide(centroids[indices[min_d]],(total_num[indices[min_d]]))
                    #min_d = np.argmin(distances)
                    #centroids[indices[min_d]] = np.add(centroids[indices[min_d]],x_train[i])
                    #total_num[indices[min_d]]+=1
            #for j in range(0,len(total_num)):
            #    centroids[j]=np.divide(centroids[j],total_num[j])
        else:
            centroids = []

    elif clustering_type == 'k_means':
        kmeans = KMeans(n_clusters=distance_threshold, random_state = 0).fit(x_train)
        centroids = kmeans.cluster_centers_
    elif clustering_type == 'NCM':
        centroids = [[0 for x in range(len(x_train[0]))]]
        centroids[0] = np.average(x_train,0)
    return centroids

# reduce given centroids using k-means
def reduce_centroids(centroid_pack):
    centroids = centroid_pack[0]
    reduction_per_class = centroid_pack[1]
    n_clusters = len(centroids) - reduction_per_class
    out_centroids = get_centroids([centroids,n_clusters,'k_means'])
    return out_centroids


# check if the centroids should be reduced and reduce them
def check_reduce_centroids(temp_complete_centroids,current_total_centroids,temp_exp_centroids,total_centroids_limit,increment,total_classes):
    if current_total_centroids + temp_exp_centroids > total_centroids_limit:
        reduction_centroids = current_total_centroids + temp_exp_centroids - total_centroids_limit
        classes_so_far = increment*total_classes
        centroid_pack = []
        for i in range(0,len(temp_complete_centroids)):
            reduction_per_class = round((len(temp_complete_centroids[i])/current_total_centroids)*reduction_centroids)
            centroid_pack.append([temp_complete_centroids[i],reduction_per_class])
        my_pool = Pool(len(temp_complete_centroids))
        temp_complete_centroids = my_pool.map(reduce_centroids,centroid_pack)
        my_pool.close()
    return temp_complete_centroids

def predict_multiple_class(data_vec,centroids,class_centroid,distance_metric):
    dist = [[0,class_centroid] for x in range(len(centroids))]
    for i in range(0,len(centroids)):
        dist[i][0] = find_distance(data_vec,centroids[i],distance_metric)
    return dist

# weighted voting scheme functions
def predict_multiple(data_vec,centroids,distance_metric,tops,weighting):
    dist = []
    for i in range(0,len(centroids)):
        temp = predict_multiple_class(data_vec,centroids[i],i,distance_metric)
        dist.extend(temp)
    sorted_dist = sorted(dist)
    common_classes = [0]*len(centroids)
    if tops>len(sorted_dist):
        tops = len(sorted_dist)
    for i in range(0,tops):
        if sorted_dist[i][0]==0.0:
            common_classes[sorted_dist[i][1]] += 1
        else:
            common_classes[sorted_dist[i][1]] += ((1/(i+1))*
                                                ((sorted_dist[len(sorted_dist)-1][0]-sorted_dist[i][0])/(sorted_dist[len(sorted_dist)-1][0]-sorted_dist[0][0])))
    common_classes = np.multiply(common_classes,weighting)
    return np.argmax(common_classes)

# get test accuracy
def get_test_accuracy(test_pack):
    x_test = test_pack[0]
    y_test = test_pack[1]
    centroids = test_pack[2]
    k = test_pack[3]
    total_classes = test_pack[4]
    weighting = test_pack[5]
    t_acc = []
    predicted_label = -1
    accus = [0]*total_classes
    total_labels = [0]*total_classes
    acc=0
    for i in range(0,len(y_test)):
        total_labels[y_test[i]]+=1
        predicted_label=predict_multiple(x_test[i],centroids,'euclidean',k,weighting)
        if predicted_label == y_test[i]:
            accus[y_test[i]]+=1
    for i in range(0,total_classes):
        if total_labels[i]>0:
            accus[i] = accus[i]/total_labels[i]
        else:
            accus[i] = 1.0
    acc = np.mean(accus)
    return acc
