import pandas as pd
from sklearn.model_selection import train_test_split
from copy import deepcopy
import numpy as np

data_csv = 'csv_data/HWSD2_LAYERS.csv'

def filter_dataset(data_csv):
    """
    Removes ID columns and datapoints with invalid carbon content values
    """
    df = pd.read_csv(data_csv, dtype={2: str, 3: str})
    df = df.drop(df.columns[0], axis=1) # Dropping row number

    # Remove ID columns
    df = df.drop(df.columns[0:9], axis=1)

    # Filter NaN and negative carbon values
    df = df[(df['ORG_CARBON'] >= 0) & (df['ORG_CARBON'].notna())]

    # Remove duplicate datapoints
    df = df.drop_duplicates()

    return df


def split_and_save_dataset(df):
    """
    Splits the data into training and testing
    """
    train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

    train_set.to_csv('data/train_set.csv', index=False)
    test_set.to_csv('data/test_set.csv', index=False)


def create_subset_test_sets():
    test_set = pd.read_csv("./data/test_set.csv")

    num_features = test_set.shape[1]
    features_to_consider = test_set.columns.difference(["ORG_CARBON"])
    
    for i in [0, 5,10,15,20,25,30,35]:
        # Deep copy the test set to avoid modifying the original DataFrame
        test_copy = deepcopy(test_set)
        
        for index, row in test_copy.iterrows():
            # Randomly choose 'i' features to set to NaN
            nan_features = np.random.choice(features_to_consider, size=min(i, num_features), replace=False)
            test_copy.loc[index, nan_features] = np.nan
        
        test_copy.to_csv('data/test_set_' + str(i) + '.csv', index=False)


if __name__ == "__main__":
    # df = filter_dataset(data_csv)
    # split_and_save_dataset(df)
    create_subset_test_sets()