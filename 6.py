# Write a program to implement Decision Tree and Random Forest with: Prediction , Test Score and Confusion Matrix


import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target


# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ---------------- Decision Tree ----------------

dt_model = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42
)

# Train model
dt_model.fit(X_train, y_train)

# Prediction
dt_prediction = dt_model.predict(X_test)

# Test Score
dt_score = dt_model.score(X_test, y_test)

print("----- Decision Tree -----")
print("Prediction:", dt_prediction)
print("Test Score:", dt_score)

# Confusion Matrix
dt_cm = confusion_matrix(y_test, dt_prediction)

print("Confusion Matrix:")
print(dt_cm)

ConfusionMatrixDisplay(
    confusion_matrix=dt_cm,
    display_labels=iris.target_names
).plot()

plt.title("Decision Tree - Confusion Matrix")
plt.show()


# ---------------- Random Forest ----------------

rf_model = RandomForestClassifier(
    n_estimators=100,
    criterion="entropy",
    random_state=42
)

# Train model
rf_model.fit(X_train, y_train)

# Prediction
rf_prediction = rf_model.predict(X_test)

# Test Score
rf_score = rf_model.score(X_test, y_test)

print("\n----- Random Forest -----")
print("Prediction:", rf_prediction)
print("Test Score:", rf_score)

# Confusion Matrix
rf_cm = confusion_matrix(y_test, rf_prediction)

print("Confusion Matrix:")
print(rf_cm)

ConfusionMatrixDisplay(
    confusion_matrix=rf_cm,
    display_labels=iris.target_names
).plot()

plt.title("Random Forest - Confusion Matrix")
plt.show()