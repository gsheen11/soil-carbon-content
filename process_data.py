import pandas as pd

# Read the CSV file
df = pd.read_csv("csv_data/HWSD2_LAYERS.csv")

# Remove columns 1-9
df = df.drop(df.columns[1:10], axis=1)

# Remove rows without a value in column 34 (ORG_CARBON)
df = df[df["34"].notna()]

# Write the filtered data to a new CSV file
df.to_csv("csv_data/HWSD2_LAYERS_CLEANED.csv", index=False)
