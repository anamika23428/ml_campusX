import pandas as pd
import numpy as np
class meralr:
    def __init__(self):
        self.m=None
        self.b=None
    def fit(self,X_train,y_train):
        num=0
        den=0
        for i in range(X_train.shape[0]):
            num=num+((X_train[i]-X_train.mean())*(y_train[i]-y_train.mean()))
            den =den+((X_train[i]-X_train.mean())*(X_train[i]-X_train.mean()))
        print("Model is Trained")

        self.m=num/den
        self.b= y_train.mean()-(self.m*X_train.mean())
        print(self.m ," " , self.b)
    def predict(self,X_test):
        return self.m*X_test + self.b
    
df= pd.read_csv('placement.csv')
from sklearn.model_selection import train_test_split
X=df.iloc[:,0]
y=df.iloc[:,-1]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

X_train_list = np.array(X_train.values)
y_train_list=np.array(y_train.values)

lr =meralr()
lr.fit(X_train_list,y_train_list)
print(lr.predict(9.8))


