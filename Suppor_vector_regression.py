"""
SVR outlierlara karşı aşırı hassas bir model bu sebeple scaler kullanmak zorundayız.
"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd


# Verileri okuma
veriler = pd.read_csv('maaslar.csv')
X = veriler.iloc[:, 1:2].values
y = veriler.iloc[:, 2:].values

# Verileri ölçeklendirme
from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
X_scaled = sc1.fit_transform(X)
sc2 = StandardScaler()
y_scaled = sc2.fit_transform(y)

# SVR modelini oluşturma
from sklearn.svm import SVR
svr_reg = SVR(kernel='rbf')
svr_reg.fit(X_scaled, y_scaled.ravel())  # y_scaled'i 1D yapıyoruz

plt.scatter(X_scaled, y_scaled)
plt.plot(X_scaled,svr_reg.predict(X_scaled))

print(svr_reg.predict([[11]]))
print(svr_reg.predict([[6.6]]))



