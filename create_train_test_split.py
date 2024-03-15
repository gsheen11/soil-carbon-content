import pandas as pd
from sklearn.model_selection import train_test_split

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

if __name__ == "__main__":
    df = filter_dataset(data_csv)
    split_and_save_dataset(df)