import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


# def load_dataset():
#     data_type_to_np_dtype = {
#         "String": str,
#         "Number": float
#     }

#     labels = [("Row Number", "ROW_NUM", "Number", float)]
#     with open('../csv_data/HWSD2_LAYERS_METADATA.csv', 'r') as file:
#         for line in file.readlines()[1:]:
#             values = line.strip().split(',')
#             slug = values[1]
#             name = values[3]
#             data_type = values[4]
#             np_dtype = data_type_to_np_dtype[data_type]
#             labels += [(name, slug, data_type, np_dtype)]
            

    
#     # Extract labels from the first line, assuming the first row contains headers
#     data = np.genfromtxt('../csv_data/HWSD2_LAYERS.csv', delimiter=',', skip_header=1, dtype=[(label[1], label[3]) for label in labels])
#     data = data[:] # remove row number from very left

#     return labels, data

def pre_process_data(df):
    float_features = df[["SAND", "SILT"]].astype(float)

    y = df[["ORG_CARBON"]].astype(float).values
    
    # categorical_feature = df[[""]]
    # encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')  # Initialize encoder, sparse=False returns a 2D array
    # one_hot_encoded_feature = encoder.fit_transform(categorical_feature)
    
    # preprocessed_features = np.concatenate([float_features.values, one_hot_encoded_feature], axis=1)
    x =  np.concatenate([float_features])
    
    return x, y


def load_training_data():
    df = pd.read_csv('data/train_set.csv')
    x, y = pre_process_data(df)
    return x, y

def load_test_data():
    df = pd.read_csv('data/test_set.csv')
    x, y = pre_process_data(df)
    return x, y


def main():
    x, y = load_training_data()
    


if __name__ == "__main__":
    main()