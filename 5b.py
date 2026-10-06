# .Implement and demonstrate the Candidate-Elimination algorithm using training data stored in a CSV file.


import pandas as pd

data = pd.read_csv("training_data.csv")

print("Training Data:")
print(data)


X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

n = X.shape[1]


S = ["0"] * n

G = [["?"] * n]


for i in range(len(X)):

    if y[i] == "Yes":

        if S == ["0"] * n:
            S = list(X[i])

        else:
            for j in range(n):
                if S[j] != X[i][j]:
                    S[j] = "?"

        G = [
            g for g in G
            if all(g[j] == "?" or g[j] == X[i][j] for j in range(n))
        ]


    elif y[i] == "No":

        new_G = []

        for g in G:

            for j in range(n):

                if g[j] == "?":

                    if S[j] != "?":
                        new_g = g.copy()
                        new_g[j] = S[j]

                        if new_g not in new_G:
                            new_G.append(new_g)

        G = new_G


    print("\nAfter Example", i + 1)
    print("Specific Boundary (S):", S)
    print("General Boundary (G):", G)


print("\n-----------------------------")
print("Final Candidate Boundaries")
print("-----------------------------")

print("Most Specific Hypothesis (S):")
print(S)

print("\nMost General Hypotheses (G):")
for g in G:
    print(g)