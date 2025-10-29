# processing_scripts/data_scaling.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pathlib import Path


class DataScaler:
    def __init__(self, input_path="../processed_datasets/processed_data.csv",
                 output_path="../processed_datasets/scaled_data.csv"):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.data = None

    def load_data(self):
        if not self.input_path.exists():
            print(f"File not found: {self.input_path}")
            return None
        self.data = pd.read_csv(self.input_path)
        print(f"Loaded data with shape {self.data.shape}")
        return self.data

    def scale_numeric_features(self, method="standard"):
        if self.data is None:
            print("No data loaded.")
            return None

        numeric_cols = self.data.select_dtypes(include=["float64", "int64"]).columns
        print(f"Numeric columns to scale: {list(numeric_cols)}")

        if method == "minmax":
            scaler = MinMaxScaler()
            scaled_values = scaler.fit_transform(self.data[numeric_cols])
        else:
            scaler = StandardScaler()
            scaled_values = scaler.fit_transform(self.data[numeric_cols])

        # Replace numeric columns with scaled versions
        self.data[numeric_cols] = scaled_values
        print(f"Applied {method} scaling to {len(numeric_cols)} numeric columns.")
        return self.data

    def save_scaled_data(self):
        if self.data is not None:
            self.data.to_csv(self.output_path, index=False)
            print(f"Saved scaled data to {self.output_path}")
            return self.output_path
        else:
            print("No data to save.")
            return None

def main():
    scaler = DataScaler()  # or "minmax"
    scaler.load_data()
    scaler.scale_numeric_features()
    scaler.save_scaled_data()

if __name__ == "__main__":
    main()

