from __future__ import print_function 
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
np.random.seed(11)

x=0.1
X0 = np.array([[6 + x    ,5 + x    ,9 + x],[9 + x,    9 + x    ,8 + x],[8 + x,    3 + x,    7 + x],[6 + x,    4 + x    ,3 + x],[8 + x,    5 + x,    4 + x],[6 + x,    5 + x,    9 + x],[7 + x,    8 + x,    8 + x],
[5 + x,    4 + x,    6 + x],[7 + x    ,3 + x,    4 + x],[9 + x,    9 + x,    7 + x],[2+ x,    5 + x,    3 + x],[8 + x,    8 + x    ,6 + x],[3 + x,    9 + x    ,4 + x],[3 + x,    9 + x,    9 + x],
[7 + x,    6 + x,    9 + x],[9 + x,    7 + x    ,6 + x],[7 + x,    4 + x,    4 + x],[7 + x    ,9 + x,    5 + x],[4 + x,    9 + x,    7 + x],[1 + x,    3 + x,    4 + x]])


X = np.concatenate((X0 ,), axis = 0)
K = 3

original_label = np.asarray([0]*20  ).T
print(original_label)
def kmeans_display(X, label):
    K = np.amax(label) + 1
    X0 = X[label == 0, :]
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]
    
    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize = 4, alpha = .8)
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize = 4, alpha = .8)
    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize = 4, alpha = .8)

    plt.axis('equal')
    plt.plot()
    plt.show()
    

def kmeans_init_centers(X, k):
    # randomly pick k rows of X as initial centers
    return X[np.random.choice(X.shape[0], k, replace=False)]

def kmeans_assign_labels(X, centers):
    # calculate pairwise distances btw data and centers
    D = cdist(X, centers)
    # return index of the closest center
    return np.argmin(D, axis = 1)

def kmeans_update_centers(X, labels, K):
    centers = np.zeros((K, X.shape[1]))
    for k in range(K):
        # collect all points assigned to the k-th cluster 
        Xk = X[labels == k, :]
        # take average
        centers[k,:] = np.mean(Xk, axis = 0)
    return centers

def has_converged(centers, new_centers):
    # return True if two sets of centers are the same
    return (set([tuple(a) for a in centers]) == 
        set([tuple(a) for a in new_centers]))
def kmeans(X, K):
    centers = [kmeans_init_centers(X, K)]
    labels = []
    it = 0 
    while True:
        labels.append(kmeans_assign_labels(X, centers[-1]))
        new_centers = kmeans_update_centers(X, labels[-1], K)
        if has_converged(centers[-1], new_centers):
            break
        centers.append(new_centers)
        it += 1
    return (centers, labels, it)
(centers, labels, it) = kmeans(X, K)
print('Centers found by our algorithm:')
print(centers[-1])

kmeans_display(X, labels[-1])
