# Implement and demonstrate the FIND-S algorithm for finding  the most specific hypothesis based on a given set of training data samples.
#  Read the training data from a CSV file.


import pandas as pd

data = pd.read_csv("training_data.csv")

print("Training Data:")
print(data)


# Separate features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]


# Start with the most specific hypothesis
hypothesis = ["0"] * len(X.columns)

print("\nInitial Hypothesis:")
print(hypothesis)


# FIND-S Algorithm
for i in range(len(X)):

    # Consider only positive examples
    if y[i] == "Yes":

        for j in range(len(X.columns)):

            # First positive example
            if hypothesis[j] == "0":
                hypothesis[j] = X.iloc[i, j]

            # If values are different
            elif hypothesis[j] != X.iloc[i, j]:
                hypothesis[j] = "?"


    print("After example", i + 1, ":", hypothesis)


# Final hypothesis
print("\nFinal Most Specific Hypothesis:")
print(hypothesis)