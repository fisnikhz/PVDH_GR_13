import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid") 

def load_data(file_path):
    """
    Load dataset from CSV and return dataframe.
    Also returns numeric columns.
    """
    df = pd.read_csv(file_path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    print(f"Numeric columns: {numeric_cols}")
    print(f"Categorical columns: {categorical_cols}")
    return df, numeric_cols, categorical_cols

def quick_report(df, numeric_cols):
    print("\n--- Quick Report ---")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Numeric columns: {numeric_cols}")
    print("\nDataFrame Info:")
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head())

def summary_statistics(df, numeric_cols):
    summary = df[numeric_cols].describe().T
    print("\nSummary Statistics:\n", summary)
    return summary

def missing_values_report(df):
    missing = df.isnull().sum()
    missing_percent = (df.isnull().mean() * 100).round(2)
    missing_df = pd.DataFrame({"missing_count": missing, "missing_percent": missing_percent})
    print("\n--- Missing Values Report ---")
    print(missing_df.sort_values("missing_percent", ascending=False))
    return missing_df

def numeric_visualizations(df, numeric_cols):
    for col in numeric_cols:
        plt.figure(figsize=(6,4))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        plt.show()

        plt.figure(figsize=(6,4))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        plt.show()

def categorical_visualizations(df, categorical_cols):
    for col in categorical_cols:
        plt.figure(figsize=(8,4))
        sns.countplot(data=df, x=col, order=df[col].value_counts().index)
        plt.title(f"Countplot of {col}")
        plt.xticks(rotation=45)
        plt.show()

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

def pairplot_visualization(df, numeric_cols):
    if len(numeric_cols) <= 10:
        sns.pairplot(df[numeric_cols])
        plt.show()
    else:
        print("Pairplot skipped: too many numeric columns (>10)")

def full_data_exploring(file_path, cat_cols=None):
    df, numeric_cols, categorical_cols = load_data(file_path)
    quick_report(df, numeric_cols)
    summary_statistics(df, numeric_cols)
    missing_values_report(df)
    numeric_visualizations(df, numeric_cols)
    if cat_cols:
        categorical_visualizations(df, cat_cols)
    elif categorical_cols:
        categorical_visualizations(df, categorical_cols)
    multivariate_analysis(df, numeric_cols)
    pairplot_visualization(df, numeric_cols)
    print("\n--- Data Exploring Complete ---")
    return df
