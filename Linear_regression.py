import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
veriler =pd.read_csv('satislar.csv')


aylar=veriler[['Aylar']]
satislar=veriler[['Satislar']]

x_train,x_test,y_train,y_test=train_test_split(aylar, satislar,test_size=0.33,random_state=0)
'''
sc=StandardScaler()

X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)

y_test=sc.fit_transform(y_test)
y_train=sc.fit_transform(y_train)
'''
#Model inşası
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train, y_train)

y_pred=lr.predict(x_test)

x_train=x_train.sort_index()
y_train=y_train.sort_index()

plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))

plt.title('Aylara Göre Satış')
plt.xlabel('Aylar')
plt.ylabel('Satışlar')
