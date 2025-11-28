import pandas as pd
import numpy as np

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
