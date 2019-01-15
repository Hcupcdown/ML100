import numpy as np
import pandas as pd


#导入数据集
dataset=pd.read_csv('Data.csv')

#iloc参数(-行选择器，-列选择器),-1表示最后一行或一列
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,3].values  #第3列的数据
print("原始输入数据：",X)

#处理丢失数据
from sklearn.preprocessing import Imputer
imputer =Imputer(missing_values="NaN",strategy="mean",axis=0)
imputer=imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])
print("处理丢失数据后:",X)

#分析导入数据
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#将X的第一列编码解析为数字
labelencoder_X = LabelEncoder()
X[ : ,0] = labelencoder_X.fit_transform(X[ : , 0])
print("将非数值类数据解析为数值数据:",X)

#将X编码为独热编码
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
print("编码为独热编码:",X)

#将Y编码解析为数字
labelencoder_Y = LabelEncoder()
Y =  labelencoder_Y.fit_transform(Y)


#拆分数据集为训练集合和测试集合
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X , Y , test_size = 0.2, random_state = 0)


#特征缩放 特征缩放时，我们通常的目的是将特征的取值约束到−1到+1的范围内
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
print("特征缩放后:",X_train)