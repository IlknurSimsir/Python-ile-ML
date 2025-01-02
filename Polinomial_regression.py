import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd


veriler =pd.read_csv('maaslar.csv')
X= veriler.iloc[:,1:2]
y= veriler.iloc[:,2:]

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X.values, y.values)

plt.scatter(X.values,y.values)
plt.plot(X,lr.predict(X.values))
plt.show()
#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
pr=PolynomialFeatures(degree=3)
X_poly=pr.fit_transform(X)

lr2=LinearRegression()
lr2.fit(X_poly,y)

plt.scatter(X.values,y.values)
plt.plot(X,lr2.predict(pr.fit_transform(X.values)))
plt.show()

#♥tahminler
print(lr.predict([[11]]))
print(lr.predict([[6.6]]))

print(lr2.predict(pr.fit_transform([[6.6]])))
print(lr2.predict(pr.fit_transform([[11]])))