import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veri=pd.read_csv('odev_tenis.csv')

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le_part=veri.iloc[:,-2:]
le=le_part.apply(LabelEncoder().fit_transform)

ohe=OneHotEncoder()
wheather=veri.iloc[:,:1]
wheather=ohe.fit_transform(wheather).toarray()

wheather_forecast=pd.DataFrame(data=wheather,index=range(14),columns=['o','r','s'])

last_data=pd.concat([wheather_forecast,veri.iloc[:,1:3],le],axis=1)

y=last_data[['humidity']]
X=last_data.drop(y,axis=1).values


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X, y,test_size=0.3,random_state=0)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train, y_train)
y_pred=lr.predict(X_test)

#Backpropagation
import statsmodels.api as sm
X=np.append(arr=np.ones((14,1)).astype(int), values=last_data.iloc[:,:-1],axis=1)
X_l=last_data.iloc[:,[0,1,2,3,4,5]].values
X_l=np.array(X_l,dtype=float)

model=sm.OLS(last_data.iloc[:,-1:],X_l).fit()
print(model.summary())

