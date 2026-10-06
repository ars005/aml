# Design an Expert System using AIML
# Example: Expert system for responding to patient queries for identifying flu.


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = {
'Fever':[1,1,1,1,1,0,0,0,1,0,1,1,0,0,1,1,0,1,0,1],
'Cough':[1,1,1,0,1,1,1,0,0,0,1,1,1,0,1,1,0,0,1,1],
'SoreThroat':[1,1,0,1,1,1,0,1,0,0,1,0,1,0,1,1,1,0,0,1],
'Fatigue':[1,1,1,1,0,1,0,0,0,0,1,1,0,1,1,0,1,1,0,1],
'Headache':[1,1,1,1,1,0,0,0,1,0,0,0,1,0,1,0,0,1,1,1],
'BodyPain':[1,0,1,1,1,0,0,0,0,0,1,1,0,0,1,1,0,1,0,1],
'RunnyNose':[1,1,1,1,1,1,1,0,0,1,1,1,0,1,0,0,0,0,1,1],
'Chills':[1,1,1,0,1,0,0,0,0,0,0,1,0,0,1,1,0,1,0,0],
'Flu':[1,1,1,1,1,0,0,0,0,0,1,1,0,0,1,1,0,1,0,1]
}
df = pd.DataFrame(data)
df.head()
X = df.drop('Flu', axis=1)
y = df['Flu']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, pred))
features = ['Fever', 'Cough', 'Sore Throat', 'Fatigue', 'Headache', 'Body Pain', 'Runny Nose', 'Chills']

user_data = []

print("****** FLU EXPERT SYSTEM ******")

for symptom in features:
    value = int(input(f"Do you have {symptom}? (1=Yes, 0=No): "))
    user_data.append(value)

result = model.predict([user_data])
print("\nPrediction:")
if result[0] == 1:
  print("Patient is likely suffering from FLU.")
else:
  print("Patient is NOT likely suffering from FLU.")