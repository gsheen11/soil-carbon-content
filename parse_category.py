import pandas as pd
import csv

# Read the CSV file
df = pd.read_csv("csv_data/HWSD2_LAYERS_CLEANED.csv")

temp = pd.read_csv("csv_data/D_WRB_PHASES.csv")
mapping_10 = temp.set_index("1").to_dict()["0"]
mapping_10[""] = ""
df["10"] = df["10"].map(mapping_10)

temp = pd.read_csv("csv_data/D_WRB4.csv")
mapping_11 = temp.set_index("2").to_dict()["0"]
mapping_11[""] = ""
df["11"] = df["11"].map(mapping_11)

mapping_11 = dict()
with open("csv_data/D_WRB2.csv", "r") as file:
    reader = csv.reader(file)
    header = next(reader)
    row_num = 1
    for row in reader:
        mapping_11[row[0]] = row_num
        row_num += 1
mapping_11[""] = ""
df["12"] = df["12"].map(mapping_11)

# Write the filtered data to a new CSV file
df.to_csv("csv_data/HWSD2_LAYERS_CATEGORIZED.csv", index=False)
