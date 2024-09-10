import pandas as pd

file = r"C:\Users\Admin\Desktop\AI INVIGILATING SYSTEM\dataset.csv"
new_file = []
df = pd.read_csv(file)
dt = df["Avg Magnitude"].tolist()
counter = 0
for data in dt:
    csv = {"id": counter, "Avg Magnitude": data}
    new_file.append(csv)
    counter += 1

dta = pd.DataFrame(new_file)
print(dta)
# dta.to_csv("dataset_2.csv", index=False)
