import pandas as pd
from create_train_test_split import filter_dataset
import seaborn as sns
import matplotlib.pyplot as plt

data_csv = 'data/train_set.csv'

def get_correlation_matrix(df):
    float_features = ['ORG_CARBON', 'TOPDEP', 'BOTDEP', 'COARSE', 'SAND', 'SILT', 'CLAY',
        'BULK', 'REF_BULK', 'PH_WATER', 'TOTAL_N', 'CN_RATIO', 'CEC_SOIL', 'CEC_CLAY',
        'CEC_EFF', 'TEB', 'BSAT', 'ALUM_SAT', 'ESP', 'TCARBON_EQ', 'GYPSUM', 'ELEC_COND']
    df_float = df[float_features].astype(float)
    correlation_matrix = df_float.corr()
    return correlation_matrix


def get_cooccurrence_matrix(df):
    categorical_features = ['WRB_PHASES', 'WRB4', 'WRB2', 'FAO90', 'ROOT_DEPTH',
        'PHASE1', 'PHASE2', 'ROOTS', 'IL', 'SWR', 'DRAINAGE', 'AWC', 'ADD_PROP',
        'LAYER', 'TEXTURE_USDA', 'TEXTURE_SOTER']

if __name__ == "__main__":
    df = pd.read_csv(data_csv)
    correlation_matrix = get_correlation_matrix(df)
    
    # Set up the matplotlib figure
    plt.figure(figsize=(12, 9))  # Adjust the size as needed

    # Generate a heatmap
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm',
                cbar_kws={'label': 'Correlation coefficient'},
                square=True, linewidths=.5)

    # Add title and adjust plot as needed
    plt.title('Correlation Matrix')
    plt.tight_layout()  # Adjusts subplot params for better layout

    # Show plot
    plt.show()