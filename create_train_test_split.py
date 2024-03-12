import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

df = pd.read_csv('csv_data/HWSD2_LAYERS.csv', dtype={2: str, 3: str})

first_column_name = df.columns[0]
df = df.drop(first_column_name, axis=1) # dropping row number

header_df = pd.read_csv('csv_data/HWSD2_LAYERS_METADATA.csv')
headers = header_df['Feature Symbol'].values
df.columns = headers

df_filtered = df[(df['ORG_CARBON'] >= 0) & (df['ORG_CARBON'].notna())]
# values_counts = df_filtered['ORG_CARBON'].value_counts()

# Split the data while keeping soil units together
gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
for train_idx, test_idx in gss.split(df_filtered, groups=df_filtered['ORG_CARBON']):
    train_set = df_filtered.iloc[train_idx]
    test_set = df_filtered.iloc[test_idx]

train_set.to_csv('data/train_set.csv', index=False)
test_set.to_csv('data/test_set.csv', index=False)