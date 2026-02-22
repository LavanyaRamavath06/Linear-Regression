**# Linear-Regression**

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
data = {
    'Experience (Years)': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Salary (in $1000)': [35, 37, 39, 45, 49, 52, 60, 62, 70, 75]
}
df = pd.DataFrame(data)
print("Dataset:")
print(df)
X = df[['Experience (Years)']]  
y = df['Salary (in $1000)']      
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nModel Evaluation:")
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, model.predict(X), color="red", label="Regression Line")
plt.title("Linear Regression: Experience vs Salary")
plt.xlabel("Experience (Years)")
plt.ylabel("Salary (in $1000)")
plt.legend()
plt.show()
