#ARM (Association Rule Mining)
# Apriori
# Importing the libraries
import pandas as pd

veriler = pd.read_csv('sepet.csv', header = None)

t = []
for i in range (0,7501):
    t.append([str(veriler.values[i,j]) for j in range (0,20)])

from apyori import apriori #eklediğimiz dosyadan çektik
kurallar = apriori(t,min_support=0.01, min_confidence=0.2, min_lift = 3, min_length=2)

print(list(kurallar))

