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

from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')

Yas=veriler.iloc[:,1:4].values

imputer=imputer.fit(Yas[:,1:4])
Yas[:,1:4]=imputer.transform(Yas[:,1:4])

sonuc=pd.DataFrame(data=ulke,index=range(22),columns=['fr','tr','us'])

sonuc2=pd.DataFrame(data=Yas,index=range(22),columns=['boy','kilo','yas'])

cinsiyet=veriler.iloc[:,-1].values

sonuc3=pd.DataFrame(data=cinsiyet,index=range(22),columns=['cinsiyet'])

s=pd.concat([sonuc,sonuc2,sonuc3],axis=1)

print(s)
