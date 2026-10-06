# A.Implement Conditional Probability and Joint Probability using Python.


import pandas as pd

data = {
    'Student': ['A','B','C','D','E','F','G','H'],
    'Passed':  [1,1,0,1,0,1,0,1],
    'Studied': [1,1,1,0,0,1,0,0]
}

df = pd.DataFrame(data)
print(df)

total = len(df)

P_pass = len(df[df['Passed']==1])/total

P_study = len(df[df['Studied']==1])/total

P_pass_and_study = len(df[(df['Passed']==1) & (df['Studied']==1)])/total

P_pass_given_study = P_pass_and_study/P_study

print("\nProbability of Passing:", round(P_pass,2))
print("Probability of Studying:", round(P_study,2))
print("Joint Probability P(Pass ∩ Study):", round(P_pass_and_study,2))
print("Conditional Probability P(Pass | Study):", round(P_pass_given_study,2))



# total_students = 100

# studied = 60

# studied_and_passed = 50

# passed = 70


# conditional_probability = studied_and_passed / passed

# print("Conditional Probability:")
# print("P(Studied | Passed) =", conditional_probability)


# joint_probability = studied_and_passed / total_students

# print("\nJoint Probability:")
# print("P(Studied AND Passed) =", joint_probability)