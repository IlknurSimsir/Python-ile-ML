import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd


# Verileri okuma
veriler = pd.read_csv('maaslar.csv')
X = veriler.iloc[:, 1:2].values
y = veriler.iloc[:, 2:].values

from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=10,random_state=0)
rf.fit(X,y.ravel())

plt.scatter(X, y)
plt.plot(X, rf.predict(X))

print(rf.predict([[11]]))
print(rf.predict([[6.5]]))