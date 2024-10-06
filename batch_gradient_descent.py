import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

df = pd.read_csv('diabetes.csv')
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)
reg = LinearRegression()
reg.fit(X_train,y_train)
print(reg.coef_)
print(reg.intercept_)
y_pred = reg.predict(X_test)
print(r2_score(y_test,y_pred))


print("custom mode")
## custom
class gdreg:
    def __init__(self,learning_rate=0.01,iterations=100):
        self.lr = learning_rate
        self.iterations = iterations
        self.coef_=None
        self.intercept_=None
    
    def fit(self,X_train,y_train):
        self.intercept_=0
        self.coef_=np.zeros(X_train.shape[1])
        for i in range(self.iterations):
            
            ##y_cap = b0 + b1(x11) + ...+b8(x18) = b0 + [x11 x12 .. x18][[b0] , [b1],...,[bm]]
            ## b0 + (1Xm)X(mX1) 
            y_cap = self.intercept_ + np.dot(X_train, self.coef_)
            intrcept_der = -2*(np.mean(y_train - y_cap))
            self.intercept_=self.intercept_-(self.lr)*(intrcept_der)

            coef_der = (-2/X_train.shape[0])*(np.dot((y_train-y_cap) ,X_train))
            self.coef_ = self.coef_ - self.lr*(coef_der)
        print(self.intercept_ , self.coef_)
    def predict(self,X_test):
        return np.dot(X_test , self.coef_) +self.intercept_
        


gd = gdreg(learning_rate=0.000000001,iterations=100000)
gd.fit(X_train,y_train)
y_pred2 = gd.predict(X_test)
print(r2_score(y_test,y_pred2))
