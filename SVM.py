import pandas as pd

veriler=pd.read_csv('veriler_full.csv')
X=veriler.iloc[:,1:4].values
y=veriler.iloc[:,4:].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X, y,test_size=0.33,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)

from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
svc=SVC(kernel='rbf')#poly , rbf , linear
svc.fit(X_train,y_train)


y_pred=svc.predict(X_test)
cm=confusion_matrix(y_test, y_pred)
print(cm)