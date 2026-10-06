# Design a simple Machine Learning model to train the training instances and test the same.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = {
    "Age": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
    "Study Hours": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "Attendance": [60, 65, 70, 75, 80, 82, 85, 90, 92, 95],
    "Result": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)


# Features and Target
X = df[["Age", "Study Hours", "Attendance"]]
y = df["Result"]


# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

print("\nTraining Instances:", len(X_train))
print("Testing Instances:", len(X_test))


# Create the model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

print("\nModel Training Completed!")


# Test the model
y_pred = model.predict(X_test)

print("\nActual Results:", [int(x) for x in y_test])
print("Predicted Results:", [int(x) for x in y_pred])


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy * 100, "%")


# Predict a new student
new_student = [[26, 8, 88]]

prediction = model.predict(new_student)[0]

print("\nNew Student:")
print("Age:", 26)
print("Study Hours:", 8)
print("Attendance:", 88)

if int(prediction) == 1:
    print("Predicted Result: PASS")
else:
    print("Predicted Result: FAIL")