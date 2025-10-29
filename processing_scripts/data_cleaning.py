# processing_scripts/data_cleaning.py
import pandas as pd
import numpy as np
from pathlib import Path

class DataCleaner:
    def __init__(self, input_dir="../unprocessed_datasets", output_dir="../processed_datasets"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.data = None

    def load_data(self, file_path):
        """Load a single CSV file."""
        try:
            df = pd.read_csv(file_path)
            df["source_file"] = file_path.stem
            self.data = df
            print(f"Loaded {file_path.name} with {len(df)} rows")
            return df
        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")
            return None

    def clean_column_names(self):
        self.data.columns = [col.strip().lower().replace(" ", "_") for col in self.data.columns]
        print("Standardized column names.")
        return self.data

    def handle_missing_values(self, strategy="mean"):
        num_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if strategy == "mean":
                self.data[col] = self.data[col].fillna(self.data[col].mean())
            elif strategy == "median":
                self.data[col] = self.data[col].fillna(self.data[col].median())
        # For non-numeric columns
        self.data = self.data.fillna("Unknown")
        print("Handled missing values.")
        return self.data

    def remove_duplicates(self):
        before = len(self.data)
        self.data.drop_duplicates(inplace=True)
        print(f"Removed {before - len(self.data)} duplicate rows.")
        return self.data

    def save_cleaned_data(self, filename=None):
        if filename is None:
            # use source_file to make a unique name
            filename = f"cleaned_{self.data['source_file'].iloc[0]}.csv"
        output_path = self.output_dir / filename
        self.data.to_csv(output_path, index=False)
        print(f"Saved cleaned data to {output_path.resolve()}")
        return output_path

    def summary(self):
        print("\nData Summary After Cleaning:")
        print(self.data.info())
        print(self.data.describe(include='all').T.head(10))

def main():
    cleaner = DataCleaner()
    files = list(cleaner.input_dir.glob("*.csv"))

    for file in files:
        df = cleaner.load_data(file)
        if df is not None:
            cleaner.clean_column_names()
            cleaner.handle_missing_values(strategy="mean")
            cleaner.remove_duplicates()
            cleaner.save_cleaned_data()  # no need to pass filename now
            print(f"Data cleaning completed for {file.name}!")

if __name__ == "__main__":
    main()
