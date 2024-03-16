import pandas as pd
from create_train_test_split import filter_dataset
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import K
import util

train_data = 'data/train_set.csv'
test_data = 'data/test_set.csv'

def plot_correlation_matrix(df):
    continuous_features = K.CONTINUOUS_FEATURES
    continuous_features.append('ORG_CARBON')
    df_continuous = df[continuous_features].astype(float)
    correlation_matrix = df_continuous.corr()

    plt.figure(figsize=(12, 9))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm',
                cbar_kws={'label': 'Correlation coefficient'},
                square=True, linewidths=.5)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()

def plot_cooccurrence_matrix(df):
    binary_df = df.notnull().astype(int)
    co_occurrence_matrix = binary_df.T.dot(binary_df)
    column_maxes= co_occurrence_matrix.max(axis=0)
    cooccurrence_matrix = co_occurrence_matrix.div(column_maxes, axis='columns')

    plt.figure(figsize=(12, 10))
    sns.heatmap(cooccurrence_matrix, annot=False, fmt=".2f", cmap="coolwarm", annot_kws={"size": 8}) 
    plt.title("Conditional Probability Matrix of Feature Co-occurrence")
    plt.show()

def plot_categorical_feature_counts(df, categorical_feature):
    category_counts = df[categorical_feature].value_counts(dropna=False).sort_index()
    plt.figure(figsize=(10, 6))
    category_counts.plot(kind='bar', color='skyblue')
    plt.title(f'Counts of {categorical_feature}', fontsize=16)
    plt.xlabel(f'{categorical_feature} Values', fontsize=14)
    plt.ylabel('Count', fontsize=14)

    max_ticks = 20
    tick_positions = np.linspace(0, len(category_counts) - 1, min(len(category_counts), max_ticks), dtype=int)
    tick_labels = []
    for pos in tick_positions:
        label = category_counts.index[pos]
        if pd.isna(label):
            tick_labels.append('NaN')
        elif isinstance(label, (int, float, np.integer)):
            tick_labels.append(f'{int(label)}')
        else:
            tick_labels.append(label)
    plt.xticks(tick_positions, tick_labels, rotation=0)
    plt.xlim(-0.5, len(category_counts))  
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)  
    plt.show()

def plot_continuous_feature_distribution(df, continuous_feature):
    plt.figure(figsize=(10, 6))  # Adjust the size as needed
    sns.scatterplot(x=continuous_feature, y='ORG_CARBON', data=df, alpha=0.5)  # alpha controls the transparency of the dots
    plt.title(f'Distribution of Carbon Content for {continuous_feature}')    
    plt.xlabel(f'{continuous_feature}')
    plt.ylabel('ORG_CARBON')    
    plt.show()

def plot_categorical_feature_distribution(df, categorical_feature):
    df = df.copy()

    order = [str(item) for item in sorted(list(df[categorical_feature].dropna().unique()))]
    order.append('NaN')

    df.loc[:, categorical_feature] = df[categorical_feature].fillna('NaN')
    df.loc[:, categorical_feature] = df[categorical_feature].astype(str)

    plt.figure(figsize=(12, 6))

    sns.boxplot(x=categorical_feature, y='ORG_CARBON', data=df, order=order)

    max_ticks = 20
    current_tick_labels = [item.get_text() for item in plt.gca().get_xticklabels()]
    tick_positions = np.linspace(0, len(current_tick_labels) - 1, min(len(current_tick_labels), max_ticks), dtype=int)
    new_tick_labels = []
    for pos in tick_positions:
        label = current_tick_labels[pos]
        try:
            new_label = f'{int(float(label))}' if pd.notna(label) and label.replace('.', '', 1).isdigit() else label
            new_tick_labels.append(new_label)
        except ValueError: 
            new_tick_labels.append(label)
    plt.xticks(tick_positions, new_tick_labels, rotation=0)

    plt.xlabel(categorical_feature)
    plt.ylabel('Carbon Content')
    plt.title(f'Distribution of Carbon Content for {categorical_feature}')
    plt.show()

def plot_continuous_features(df):
    continuous_features = K.CONTINUOUS_FEATURES
    continuous_features.append('ORG_CARBON')
    df = df[continuous_features]
    for continuous_feature in K.CONTINUOUS_FEATURES:
        plot_continuous_feature_distribution(df, continuous_feature)

def plot_categorical_features(df):
    df = util.pre_process_categorical_feature(df)
    categorical_features = K.CATEGORICAL_FEATURES
    categorical_features.append('ORG_CARBON')
    df = df[categorical_features]
    for categorical_feature in K.CATEGORICAL_FEATURES[:2]:
        plot_categorical_feature_counts(df, categorical_feature)
        plot_categorical_feature_distribution(df, categorical_feature)

def plot_org_carbon(df):
    plt.figure(figsize=(10, 6))  # Adjust the size as needed
    plt.hist(df['ORG_CARBON'], bins=30, color='skyblue', edgecolor='black')  # You can change the number of bins
    plt.title('Histogram of ORG_CARBON')
    plt.xlabel('ORG_CARBON')
    plt.ylabel('Counts')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv(train_data)

    plot_correlation_matrix(df)
    # plot_cooccurrence_matrix(df)
    # plot_continuous_features(df)
    # plot_categorical_features(df)
    # plot_org_carbon(df)
