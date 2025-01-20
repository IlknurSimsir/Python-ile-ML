import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

# Veriyi yükleme
veriler = pd.read_csv('Churn_Modelling.csv')
X = veriler.iloc[:, 3:13].values
y = veriler.iloc[:, 13].values

# Label Encoding
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
X[:, 1] = le.fit_transform(X[:, 1])
X[:, 2] = le.fit_transform(X[:, 2])

# OneHot Encoding
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ohe = ColumnTransformer([("ohe", OneHotEncoder(dtype=float), [1])], remainder="passthrough")
X = ohe.fit_transform(X)
X = X[:, 1:]

# Veriyi eğitim ve test olarak ayırma
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# Özellik ölçeklendirme
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Yapay Sinir Ağı Modeli
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

classifier = Sequential()

# İlk gizli katman
classifier.add(Dense(6, kernel_initializer='uniform', activation="relu", input_dim=11))

# İkinci gizli katman
classifier.add(Dense(6, kernel_initializer='uniform', activation="relu"))

# Çıkış katmanı
classifier.add(Dense(1, kernel_initializer='uniform', activation="sigmoid"))

# Modeli derleme
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Modeli eğitme
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

y_pred=classifier.predict(X_test)

y_pred=(y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
print(cm)

