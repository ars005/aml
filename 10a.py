# Use of Pandas AI: The Generative AI Python Library
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

print("Pandas version:", pd.__version__)

# Create a sample dataset
data = {
    "Student_ID": range(1, 16),
    "Name": [
        "Aarav","Diya","Rohan","Anaya","Vivaan",
        "Isha","Kabir","Meera","Arjun","Sara",
        "Aditya","Nisha","Rahul","Kiara","Dev"
    ],
    "Department": [
        "Computer Science","IT","Computer Science","Management","IT",
        "Computer Science","Management","IT","Computer Science","Management",
        "IT","Computer Science","Management","IT","Computer Science"
    ],
    "Attendance": [92,85,78,90,88,95,72,81,89,76,94,87,80,91,96],
    "Internal_Marks": [45,39,36,44,41,47,34,38,43,35,46,40,37,42,48],
    "Assignment_Marks": [18,17,15,19,16,20,14,16,18,15,19,17,16,18,20],
    "Final_Marks": [88,76,69,91,82,94,65,73,86,68,92,79,71,84,96]
}

df = pd.DataFrame(data)

print("\nDataset:")
print(df)

# Basic Pandas analysis
print("\nShape:", df.shape)
print("\nColumns:", list(df.columns))
print("\nAverage Final Marks:", round(df["Final_Marks"].mean(), 2))
print("\nHighest Final Marks:", df["Final_Marks"].max())

print("\nTop 5 students:")
print(df.nlargest(5, "Final_Marks")[["Name", "Department", "Final_Marks"]])