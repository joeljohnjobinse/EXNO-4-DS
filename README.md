# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv('income(1) (1).csv',na_values=[" ?"])
data
```
![image](https://github.com/user-attachments/assets/623558d7-da6e-447d-b74a-e81eca5390c4)
```
data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/4dcdcb16-3f5b-4708-8e51-8ef4b607c4b1)
```
missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/user-attachments/assets/569867be-5d5d-4694-be5a-e428d4f49cf4)
```
data2 = data.dropna(axis=0)
data2
```
![image](https://github.com/user-attachments/assets/796ad030-872b-433e-9ff2-535aa27f64c4)
```
sal=data['SalStat']
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/user-attachments/assets/6294f19f-b70e-4b0b-a39c-7f61d1cf9b96)
```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/user-attachments/assets/1cde3dd6-7563-4574-ae73-3fc07bdb07f8)
```
data2
```
![image](https://github.com/user-attachments/assets/7f241d99-47e3-4f30-9bcd-af10833a0135)
```
new_data=pd.get_dummies(data2,drop_first=True)
new_data
```
![image](https://github.com/user-attachments/assets/ccb5a1a1-0383-4980-bdee-8a04095c4c49)
```
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/user-attachments/assets/8f3bca4f-ac1b-4ac5-abe7-781db7f67e24)
```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/user-attachments/assets/b0c76804-50be-42ad-89c9-fb5ee50eac21)
```
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/user-attachments/assets/fff4cc7c-b18b-44d0-bd38-08596afe1644)
```
x=new_data[features].values
print(x)
```
![image](https://github.com/user-attachments/assets/1007bf72-2821-4c8d-ad2d-546255ccf59e)
```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier = KNeighborsClassifier(n_neighbors=5)
KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/user-attachments/assets/ca550450-16ec-4ef5-a9e8-ac7c7de8b7a8)
```
prediction = KNN_classifier.predict(test_x)
confusionMmatrix = confusion_matrix(test_y, prediction)
print(confusionMmatrix)
```
![image](https://github.com/user-attachments/assets/f22d4c9f-04a8-4ccd-832e-7b31ae962236)
```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![image](https://github.com/user-attachments/assets/3e025246-2885-48e1-8009-8c85c280a477)
```
print('Misclassified samples: %d' % (test_y != prediction).sum())
```
![image](https://github.com/user-attachments/assets/014476ac-232b-420f-8f9d-a401bf5fe384)
```
data.shape
```
![image](https://github.com/user-attachments/assets/8039a266-79c4-4796-8032-782c23710933)

# RESULT:
Thus, the program for feature selection and scaling has been implemented.
