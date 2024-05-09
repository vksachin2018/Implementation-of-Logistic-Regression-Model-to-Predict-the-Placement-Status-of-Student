# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
STEP 1 START

STEP 2:Import the required packages and print the present data.

STEP 3: Print the placement data and salary data.

STEP 4:Find the null and duplicate values.

STEP 5:Using logistic regression find the predicted values of accuracy , confusion matrices.

STEP 6:Display the results.

STEP 7:

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: gokul sachin.k
RegisterNumber:  2122232220025

import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])


```

## Output:
## Placement data:
![ml401](https://github.com/vksachin2018/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149366019/d4d1f77d-1b23-4c69-9bbc-4726df13f305)

## Salary data:
![ml402](https://github.com/vksachin2018/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149366019/1efca0bf-f897-44f2-9db8-1bbc450af8ec)

## Checking the null() function:
![ml403](https://github.com/vksachin2018/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149366019/e63e192b-ac26-4f01-973a-4dedf3ff4949)

## Data duplicate:
![ml404](https://github.com/vksachin2018/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149366019/e4ff9aae-b1e6-4caf-a232-b15f60a4d9b3)

## Print data:
![ml405](https://github.com/vksachin2018/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149366019/5cf2ee36-dbec-498d-accb-ed3ac9746107)

## Data-status:
![ml406](https://github.com/vksachin2018/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149366019/5eb296a8-b76f-4faa-83fc-432fd122b2ac)

## y_prediction array:
![ml407](https://github.com/vksachin2018/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149366019/8bde5ace-3f51-40ad-847a-4bf369333aed)

## Accuracy value:
![ml408](https://github.com/vksachin2018/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149366019/bcbdd699-d946-46d2-b254-128ca3d86b25)

## Confusion arrray:
![ml409](https://github.com/vksachin2018/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149366019/70c10e6e-bb94-44fb-a34b-cf070ac11cd5)

## Classification Report:
![ml410](https://github.com/vksachin2018/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149366019/d02e5ae4-e4f4-492c-bc9b-079718c10032)

## Prediction of LR:
![ml411](https://github.com/vksachin2018/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149366019/4fa35b90-b216-4504-ad91-f3a8d0c7cff3)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
