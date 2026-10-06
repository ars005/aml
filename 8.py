# Write a program to construct a Bayesian Network considering medical data and demonstrate diagnosis of heart patients using a standard Heart Disease Dataset.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination

# Load dataset
data = pd.read_csv("heart.csv")

# Convert Age and MaxHR into groups
data["AgeGroup"] = pd.cut(
    data["Age"], [0, 40, 50, 60, 100],
    labels=["Young", "Middle", "Senior", "Old"]
).astype(str)

data["MaxHRGroup"] = pd.cut(
    data["MaxHR"], [0, 100, 130, 160, 250],
    labels=["Low", "Moderate", "High", "VeryHigh"]
).astype(str)

# Convert values to string
for col in ["ChestPain", "ExAng", "Ca", "Thal", "AHD"]:
    data[col] = data[col].astype(str)

# Select columns
bn_data = data[
    ["AgeGroup", "ChestPain", "MaxHRGroup",
     "ExAng", "Ca", "Thal", "AHD"]
]

# Split data
train, test = train_test_split(
    bn_data, test_size=0.2, random_state=42
)

# Create Bayesian Network
model = DiscreteBayesianNetwork([
    ("AgeGroup", "AHD"),
    ("ChestPain", "AHD"),
    ("MaxHRGroup", "AHD"),
    ("ExAng", "AHD"),
    ("Ca", "AHD"),
    ("Thal", "AHD")
])

# Learn probabilities
estimator = BayesianEstimator(model, train)
model.add_cpds(*estimator.get_parameters(
    prior_type="BDeu",
    equivalent_sample_size=10
))

print("Bayesian Network:")
print(list(model.edges()))
print("Model is valid:", model.check_model())

# Diagnosis
inference = VariableElimination(model)

patient = {
    "AgeGroup": "Senior",
    "ChestPain": "asymptomatic",
    "MaxHRGroup": "Moderate",
    "ExAng": "1",
    "Ca": "1",
    "Thal": "reversable"
}

result = inference.query(["AHD"], evidence=patient)

print("\nPatient Diagnosis:")
print(result)

prediction = result.state_names["AHD"][
    np.argmax(result.values)
]

print("Diagnosis:", prediction)

# Accuracy
predictions = []
actual = []

for _, row in test.iterrows():
    evidence = {
        col: row[col]
        for col in ["AgeGroup", "ChestPain", "MaxHRGroup",
                    "ExAng", "Ca", "Thal"]
    }

    result = inference.query(["AHD"], evidence=evidence)

    pred = result.state_names["AHD"][
        np.argmax(result.values)
    ]

    predictions.append(pred)
    actual.append(row["AHD"])

accuracy = accuracy_score(actual, predictions)

print("\nAccuracy:", round(accuracy * 100, 2), "%")