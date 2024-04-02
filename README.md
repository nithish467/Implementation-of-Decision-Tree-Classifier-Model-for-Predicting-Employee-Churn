# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import pandas module and import the required data set.
2. Find the null values and count them.
3. Count number of left values.
4. From sklearn import LabelEncoder to convert string values to numerical values.
5. From sklearn.model_selection import train_test_split.

## Program:
```

Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: NITHISH KUMAR S 
RegisterNumber:212223240109

import pandas as pd
data = pd.read_csv("/content/Employee.csv")

data.info()

data.isnull().sum()

data["left"].value_counts

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x= data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]

x.head()
y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 100)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)

y_pred = dt.predict(x_test)
from sklearn import metrics

accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])


```

## Output:
![image](https://github.com/nithish467/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/150232274/1d906166-50f9-43de-b86c-4165c5f5b22a)
![image](https://github.com/nithish467/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/150232274/62cdad62-9774-4f82-abc5-551f72f67a35)
![image](https://github.com/nithish467/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/150232274/e4998ef4-96bd-42e8-aab5-f6b5c30bb639)
![image](https://github.com/nithish467/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/150232274/03659916-de50-4931-bfd8-1e501d482126)
![image](https://github.com/nithish467/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/150232274/8c3156af-b234-4fd2-9f74-1796c0abf894)
![image](https://github.com/nithish467/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/150232274/fb572554-f8be-431e-9fef-5b13703b899c)
![image](https://github.com/nithish467/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/150232274/b5e5f73d-ac43-46f2-85fd-8534f56b3bb1)
![image](https://github.com/nithish467/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/150232274/91eca786-38c4-41ef-b131-a9ebfa14eb88)







## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
