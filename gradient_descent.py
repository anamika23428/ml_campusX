import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



class meragd:
    def __init__(self, learning_rate , epochs):
        self.m = 29.19
        self.b=-120
        self.lr = learning_rate
        self.epochs = epochs

    def fit(self,X,y):
        for i in range(self.epochs):
            # Compute gradients for m and b
            m_slope = -2 * np.sum((y - (self.m * X.ravel() + self.b)) * X.ravel())
            b_slope = -2 * np.sum(y - (self.m * X.ravel() + self.b))

            # Update m and b
            self.m = self.m - self.lr * m_slope
            self.b = self.b - self.lr * b_slope
        
        print(f"Final slope (m): {self.m}")
        print(f"Final intercept (b): {self.b}")
        print("Model is trained")

    def predict(self , x):
        return (self.m*x) + self.b
df = pd.read_csv('placement.csv')
X=df.iloc[:,0]
y=df.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# In-built LinearRegression
lr = LinearRegression()
lr.fit(X_train.values.reshape(-1, 1), y_train)  # Reshaping X_train for sklearn

print("m (slope) = ", lr.coef_[0])  # Slope (m)
print("b (intercept) = ", lr.intercept_)


model = meragd(0.0001 ,10)
model.fit(X_train,y_train)
#print(model.predict(X_test.iloc[0]))
#print(y_test.iloc[0])