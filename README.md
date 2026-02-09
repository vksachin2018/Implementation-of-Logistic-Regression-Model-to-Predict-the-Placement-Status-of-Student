# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices
5. Display the results.


## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Gokul sachin k
RegisterNumber: 212223220025
*/
 import pandas as pd
 data=pd.read_csv("C:/Users/SEC/Downloads/Placement_Data.csv")
 data.head()

 data1=data.copy()
 data1=data1.drop(["sl_no","salary"],axis=1)
 data1.head()

 data1.isnull()
 data1.duplicated().sum()

 from sklearn .preprocessing import LabelEncoder
 le=LabelEncoder()
 data1["gender"]=le.fit_transform(data1["gender"])
 data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
 data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
 data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
 data1["degree_t"]=le.fit_transform(data1["degree_t"])
 data1["workex"]=le.fit_transform(data1["workex"])
 data1["specialisation"]=le.fit_transform(data1["specialisation"])
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

 from sklearn.metrics import classification_report
 classification_report1=classification_report(y_test,y_pred)
 print(classification_report1)
 lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:
<img width="1371" height="328" alt="image" src="https://github.com/user-attachments/assets/a3a5f64a-15bf-4955-b30b-b577bfcf6940" />
<img width="1201" height="344" alt="image" src="https://github.com/user-attachments/assets/8f1f70e6-8e5d-4925-af69-f69e491e4744" />
<img width="1109" height="811" alt="image" src="https://github.com/user-attachments/assets/7014121f-b625-4919-8a6b-5a91617176ae" />
<img width="907" height="637" alt="image" src="https://github.com/user-attachments/assets/72d03c71-b17c-4ecf-80cf-290991250a71" />
<img width="1680" height="391" alt="image" src="https://github.com/user-attachments/assets/54eba7b5-8b10-4bac-8b4e-5d610b980629" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
