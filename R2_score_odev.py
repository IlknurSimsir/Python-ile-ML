import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
import statsmodels.api as sm

veriler =pd.read_csv('maaslar_yeni.csv')
X=veriler.iloc[:,2:5].values
y=veriler.iloc[:,5:].values


#LINEAR REGRESSION
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X,y)

print('Linear OLS :')
model=sm.OLS(lr.predict(X),X)
print(model.fit().summary())

print('Linear R2 :')
print(r2_score(y, lr.predict(X)))


#POLYNOMIAL REGRESSION
from sklearn.preprocessing import PolynomialFeatures
pf=PolynomialFeatures(degree=4)
X_poly=pf.fit_transform(X)

pr=LinearRegression()
pr.fit(X_poly, y)

print('POLYNOMIAL OLS :')
model2=sm.OLS(pr.predict(pf.fit_transform(X)),X)
print(model2.fit().summary())

print('Polynomial R2 :')
print(r2_score(y, pr.predict(pf.fit_transform(X))))


#SUPPORT VECTOR REGRESSION
from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
X_scaled = sc1.fit_transform(X)
sc2 = StandardScaler()
y_scaled = sc2.fit_transform(y)

from sklearn.svm import SVR
svr_reg = SVR(kernel='rbf')
svr_reg.fit(X_scaled, y_scaled.ravel())

print('SVR OLS :')
model3=sm.OLS(svr_reg.predict(X_scaled),X_scaled)
print(model3.fit().summary())

print('SVR R2 :')
print(r2_score(y, svr_reg.predict(X_scaled)))


#DECISION TREE REGRESSION
from sklearn.tree import DecisionTreeRegressor
r_dt=DecisionTreeRegressor(random_state=0)
r_dt.fit(X,y)

print('Decision Tree OLS :')
model4=sm.OLS(r_dt.predict(X),X)
print(model4.fit().summary())

print('Decision Tree R2 :')
print(r2_score(y, r_dt.predict(X)))


#RANDOM FOREST REGRESSION
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=10,random_state=0)
rf.fit(X,y.ravel())

print('Decision Tree OLS :')
model5=sm.OLS(rf.predict(X),X)
print(model5.fit().summary())

print('Decision Tree R2 :')
print(r2_score(y, rf.predict(X)))


















