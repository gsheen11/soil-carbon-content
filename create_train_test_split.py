# import pandas as pd
# from sklearn.model_selection import train_test_split

# df = pd.read_csv('csv_data/HWSD2_LAYERS.csv')

# first_column_name = df.columns[0]
# df = df.drop(first_column_name, axis=1) # dropping row number

# labels = []
# with open('csv_data/HWSD2_LAYERS_METADATA.csv', 'r') as file:
#     for line in file.readlines()[1:]:
#         values = line.strip().split(',')
#         slug = values[1]
#         # name = values[3]
#         # data_type = values[4]
#         # np_dtype = data_type_to_np_dtype[data_type]
#         labels += [slug]

# df.columns = labels

# train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)


# train_set.to_csv('data/train_set.csv', index=False)
# test_set.to_csv('data/test_set.csv', index=False)