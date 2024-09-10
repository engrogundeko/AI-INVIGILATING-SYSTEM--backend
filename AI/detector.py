import csv
import pandas as pd

file = r"C:\Users\Admin\Desktop\AI INVIGILATING SYSTEM\dataset.csv"
df = pd.read_csv(file)

# df = pd.read_csv(filename)

# List of columns you want to keep (based on your header)
columns_to_keep = ["Avg Magnitude", "Std Magnitude", "Avg Direction", "Num Moving Points", "Object Size", "Label"]  # Example columns

# Drop columns not in the list of columns to keep
df = df[columns_to_keep]

# Save the cleaned DataFrame back to a new CSV file
df.to_csv("cleaned_features.csv", index=False)

print("Columns not in header have been removed.")
