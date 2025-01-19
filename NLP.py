import numpy as np 
import pandas as pd

comments = pd.read_csv('Restaurant_Reviews.csv',on_bad_lines='skip')

import re
import nltk 
stopword=nltk.download('stopwords')
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

derlem=[]
for i in range(1000):
    comment=re.sub('[^a-zA-Z]',' ',comments['Review'][0])
    comment=comment.lower()
    
    comment=comment.split() #kelimeleri listeye attık
    comment=[ps.stem(word) for word in comment if not word in set(stopwords.words('english'))]
    
    comment = ' '.join(comment)
    derlem.append(comment)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2000)
X=cv.fit_transform(derlem).toarray() #Bağımsız değişken
y=comments.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X, y,test_size=0.2,random_state=0)

from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train,y_train)
y_pred=gnb.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
print(cm)