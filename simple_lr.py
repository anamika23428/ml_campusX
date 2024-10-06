import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

df = pd.read_csv('placement.csv')
X=df.iloc[:,0]
y=df.iloc[:,-1]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)
X_train = X_train.values.reshape(-1, 1)  # Reshape to 2D with 1 column
y_train = y_train.values.reshape(-1, 1)  # Reshape to 2D with 1 column
y_test= y_test.values.reshape(-1, 1) 

lr= LinearRegression() ## object is created named lr
lr.fit(X_train,y_train) ## fit attribute is used to train the model 
print(lr.predict(X_test.iloc[0].reshape(1,1))) ## since .iloc will give 1d data but the input for function "predict" takes 2d data i.e 1 row and 1 column 
print(lr.predict([[9.8]]))

##plot
plt.scatter(df['cgpa'],df['package'])
plt.plot(X_train,lr.predict(X_train) , color='red')
plt.xlabel('CGPA')
plt.ylabel('package in lpa')
#plt.show()
## y=mx+b
m=lr.coef_
b=lr.intercept_
print("m=",m," ","b=",b)

##metrics

y_predicted =np.array(lr.predict(X_test.values.reshape(-1, 1)))
print("MAE" , mean_absolute_error(y_test,y_predicted))
print("MAE" , mean_squared_error(y_test,y_predicted))
print("MAE" , r2_score(y_test,y_predicted))


