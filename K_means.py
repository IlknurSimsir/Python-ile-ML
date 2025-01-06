import pandas as pd
import matplotlib.pyplot as plt
veriler = pd.read_csv('musteriler.csv')

X=veriler.iloc[:,3:]
from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=3,init='k-means++')
kmeans.fit(X)
sonuc=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=123)
    kmeans.fit(X)
    sonuc.append(kmeans.inertia_)
plt.plot(range(1,11),sonuc)

