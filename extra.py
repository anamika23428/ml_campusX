import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

class Meragd:
    def __init__(self, learning_rate, epochs):
        self.m = 100  
        self.b = -120  
        self.lr = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        n = len(X)
        for i in range(self.epochs):
            # Compute gradients for m and b
            y_pred = self.m * X.ravel() + self.b
            m_slope = (-2/n) * np.sum((y - y_pred) * X.ravel())  # Derivative wrt m
            b_slope = (-2/n) * np.sum(y - y_pred)  # Derivative wrt b

            # Update m and b
            self.m = self.m - self.lr * m_slope
            self.b = self.b - self.lr * b_slope
        
        print(f"Final slope (m): {self.m}")
        print(f"Final intercept (b): {self.b}")
        print("Custom model is trained")

    def predict(self, x):
        return (self.m * x) + self.b

# Load the dataset
df = pd.read_csv('placement.csv')
X = df.iloc[:, 0]  # Feature (1st column)
y = df.iloc[:, -1]  # Target (last column)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# In-built LinearRegression model
lr = LinearRegression()
lr.fit(X_train.values.reshape(-1, 1), y_train)  # Reshaping X_train for sklearn

print("In-built model slope (m) = ", lr.coef_[0])  # Slope (m)
print("In-built model intercept (b) = ", lr.intercept_)  # Intercept (b)

# Custom gradient descent model
model = Meragd(learning_rate=0.0001, epochs=1000)  # Increased epochs
model.fit(X_train, y_train)

# You can uncomment these to see predictions and actual values
print(model.predict(X_test.iloc[0]))
print(y_test.iloc[0])
