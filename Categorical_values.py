import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

veriler=pd.read_csv('veriler.csv')

ulke=veriler.iloc[:,0:1].values
print(ulke)

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

le=LabelEncoder()

ulke[:,0]=le.fit_transform(veriler.iloc[:,0])
print(ulke)

ohe=OneHotEncoder()
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)