import pandas as pd
import matplotlib.pyplot as plt
veriler = pd.read_csv('musteriler.csv')

X=veriler.iloc[:,3:].values
from sklearn.cluster import AgglomerativeClustering

ac=AgglomerativeClustering(n_clusters=4, metric='euclidean',linkage='ward')

y_pred = ac.fit_predict(X)
plt.scatter(X[y_pred==0,0],X[y_pred==0,1],s=100,c='red')
plt.scatter(X[y_pred==1,0],X[y_pred==1,1],s=100,c='blue')
plt.scatter(X[y_pred==2,0],X[y_pred==2,1],s=100,c='green')
plt.scatter(X[y_pred==3,0],X[y_pred==3,1],s=100,c='yellow')
plt.show()

import scipy.cluster.hierarchy as sch
dendogram=sch.dendrogram(sch.linkage(X,method='ward'))
plt.show()