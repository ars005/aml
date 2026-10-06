# Implement the Least Square Regression algorithm for a given set of training data examples stored in a CSV file.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("test.csv")

print("Training Data:")
print(df)

X = df.iloc[:, 0].astype(float).values
Y = df.iloc[:, 1].astype(float).values

n = len(X)

sum_x = np.sum(X)
sum_y = np.sum(Y)
sum_xy = np.sum(X * Y)
sum_x2 = np.sum(X ** 2)

b1 = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)

b0 = (sum_y - b1 * sum_x) / n

# Display equation
print("\nIntercept (b0):", round(b0, 4))
print("Slope (b1):", round(b1, 4))
print("Regression Equation:")
print("Y =", round(b0, 4), "+", round(b1, 4), "X")

# Predict Y values
Y_pred = b0 + b1 * X

print("\nPredicted Values:")
print(Y_pred)

# Plot the data and regression line
plt.scatter(X, Y, label="Training Data")
plt.plot(X, Y_pred, label="Regression Line")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Least Square Regression")
plt.legend()
plt.grid(True)
plt.show()

# Predict for a new X value
new_x = float(input("\nEnter a new X value: "))

new_y = b0 + b1 * new_x

print("Predicted Y:", round(new_y, 4))