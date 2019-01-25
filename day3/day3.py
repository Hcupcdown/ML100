import pandas as pd
import numpy as np

#导入数据集
dataset=pd.read_csv('50_Startups.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,4].values

#将类别变量数据化,虚变量
from sklearn.preprocessing import LabelEncoder ,OneHotEncoder
labelencoder =LabelEncoder()
X[:,3]=labelencoder.fit_transform(X[:,3])

#categorical_features需要编码的列索引
onehotencoder =OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()

#躲避虚拟变量陷阱
X=X[:,1:]

#拆分数据集weight训练集和测试集
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

y_pred=regressor.predict(X_test)
print(y_pred)
