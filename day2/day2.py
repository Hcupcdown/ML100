import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

#数据预处理
dataset = pd.read_csv('studentscores.csv')
X=dataset.iloc[: , :1].values
Y=dataset.iloc[ : , 1].values
print(X);
print(Y);

from sklearn.model_selection import train_test_split
X-train,X_test,Y_train,Y_test=train_test_split(X，Y,test_size=1/4,random_state=0)
