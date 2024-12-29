import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

veriler=pd.read_csv('veriler.csv')
print(veriler)

from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')

Yas=veriler.iloc[:,1:4].values

imputer=imputer.fit(Yas)
Yas=imputer.transform(Yas)

veriler.iloc[:, 1:4] = Yas