import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd


# Verileri okuma
veriler = pd.read_csv('maaslar.csv')
X = veriler.iloc[:, 1:2].values
y = veriler.iloc[:, 2:].values

from sklearn.tree import DecisionTreeRegressor
r_dt=DecisionTreeRegressor(random_state=0)
r_dt.fit(X,y)

plt.scatter(X, y)
plt.plot(X, r_dt.predict(X))

print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))