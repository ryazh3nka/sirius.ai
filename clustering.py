import pandas as pd
import numpy as np
import gensim
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from gensim.models import Doc2Vec

cluster_number=74
d2v_model = gensim.models.Doc2Vec.load("d2v.model")

kmeans_model = KMeans(n_clusters=cluster_number, init="k-means++", max_iter=100) 
X = kmeans_model.fit(d2v_model.dv.vectors)
labels=kmeans_model.labels_.tolist()

l = kmeans_model.fit_predict(d2v_model.dv.vectors)
pca = PCA(n_components=2).fit(d2v_model.dv.vectors)
datapoint = pca.transform(d2v_model.dv.vectors)

#%matplotlib inline
plt.figure
chars = '0123456789ABCDEF'
label1 = ['#'+''.join(random.sample(chars,6)) for i in range(cluster_number)]
color = [label1[i] for i in labels]
plt.scatter(datapoint[:, 0], datapoint[:, 1], c=color)

centroids = kmeans_model.cluster_centers_
centroidpoint = pca.transform(centroids)
plt.scatter(centroidpoint[:, 0], centroidpoint[:, 1], marker="^", s=150, c="#000000")
plt.show()