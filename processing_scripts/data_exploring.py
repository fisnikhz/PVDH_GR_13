import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file_path):
    """
    Load dataset from CSV and return dataframe.
    Also returns numeric columns.
    """
    df = pd.read_csv(file_path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    print(f"Numeric columns: {numeric_cols}")
    return df, numeric_cols

def summary_statistics(df, numeric_cols):
    """
    Compute summary statistics: mean, median, std, min, max
    """
    summary = df[numeric_cols].describe().T
    print("\nSummary Statistics:\n", summary)
    return summary

def multivariate_analysis(df, numeric_cols):
    """
    Compute correlation matrix and plot heatmap
    """
    corr_matrix = df[numeric_cols].corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()
    return corr_matrix

def categorical_visualization(df, cat_cols):
    """
    Plot countplots for categorical columns
    """
    for col in cat_cols:
        plt.figure(figsize=(8,4))
        sns.countplot(data=df, x=col, order=df[col].value_counts().index)
        plt.title(f"Countplot for {col}")
        plt.xticks(rotation=45)
        plt.show()
