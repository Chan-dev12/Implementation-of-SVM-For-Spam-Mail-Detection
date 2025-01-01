# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and preprocess the customer data (normalize or scale the features).
2.Choose the number of clusters (k) and initialize the KMeans model.
3.Fit the KMeans model to the customer data to perform clustering.
4.Predict the cluster labels for each customer in the dataset. 5.Analyze and visualize the resulting customer segments. 

## Program:
```

Program to implement the SVM For Spam Mail Detection..
Developed by: CHANTHRU V
RegisterNumber: 24900997

import chardet
file=(r'C:\Users\admin\Downloads\spam.csv')
with open(file,'rb')as rawdata:
    result=chardet.detect(rawdata.read(100000))
print(result)
import pandas as pd
data=pd.read_csv(r'C:\Users\admin\Downloads\spam.csv',encoding='Windows-1252')
print(data.head())
print(data.info())
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
print(y_pred)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print(accuracy)

```

## Output:
![Screenshot 2024-12-22 230026](https://github.com/user-attachments/assets/ab3b117c-d7e1-4102-8bca-319a1a02d598)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
