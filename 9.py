# Implement the non-parametric Locally Weighted Regression (LWR) algorithm to fit data points. Select an appropriate dataset and draw graphs.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# Create a non-linear dataset
X = np.linspace(0, 10, 100)
y = np.sin(X) + np.random.normal(0, 0.15, 100)

data = pd.DataFrame({"X": X, "Y": y})

print(data.head())

# Plot original data
plt.scatter(X, y, label="Data")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Original Dataset")
plt.legend()
plt.show()


# Locally Weighted Regression
def locally_weighted_regression(X, y, x_query, tau=0.5):
    X = np.asarray(X)
    y = np.asarray(y)

    # Design matrix
    X_design = np.column_stack((np.ones(len(X)), X))

    # Gaussian weights
    weights = np.exp(-((X - x_query) ** 2) / (2 * tau ** 2))
    W = np.diag(weights)

    # Weighted least squares
    theta = np.linalg.pinv(X_design.T @ W @ X_design) @ \
            (X_design.T @ W @ y)

    return np.array([1, x_query]) @ theta


# Generate LWR predictions
x_test = np.linspace(0, 10, 400)

y_lwr = np.array([
    locally_weighted_regression(X, y, x, tau=0.5)
    for x in x_test
])


# Plot LWR fit
plt.scatter(X, y, label="Data")
plt.plot(x_test, y_lwr, linewidth=2, label="LWR Fit")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Locally Weighted Regression")
plt.legend()
plt.show()


# Effect of different bandwidths
plt.scatter(X, y, alpha=0.5, label="Data")

for tau in [0.2, 0.5, 1.0, 2.0]:
    y_fit = np.array([
        locally_weighted_regression(X, y, x, tau)
        for x in x_test
    ])
    plt.plot(x_test, y_fit, label=f"Tau = {tau}")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Effect of Bandwidth on LWR")
plt.legend()
plt.show()