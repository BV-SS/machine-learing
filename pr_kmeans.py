import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as so
from sklearn import datasets
from sklearn.cluster  import KMeans

x=[1,22,3,4,59,78,80]
y=[2,20,9,16,106,80,90]
x=np.array([[1,2],[22,20],[3,9],[4,16],[59,106],[3,5],[80,56]])
model=KMeans(n_clusters=6)
model.fit(x)
centroids=model.cluster_centers_
print('centroids',centroids)
labels=model.labels_
print(labels)
for i in range(len(x)):
    print('points',x[i],"is having cluster no",labels[i])
