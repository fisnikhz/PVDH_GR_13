# processing_scripts/data_reduction.py
import pandas as pd
from sklearn.decomposition import PCA
from pathlib import Path

class DataReducer:
    def __init__(self,
                 input_path="../processed_datasets/scaled_data.csv",
                 output_path="../processed_datasets/reduced_data.csv"):
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

    def correlation_reduction(self, threshold=0.9):
        if self.data is None:
            print("No data loaded.")
            return None

        numeric_data = self.data.select_dtypes(include=["float64", "int64"])
        corr_matrix = numeric_data.corr().abs()

        # Identify columns to drop
        upper_tri = corr_matrix.where(
            pd.np.triu(pd.np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]

        self.data.drop(columns=to_drop, inplace=True)
        print(f"🧹 Dropped {len(to_drop)} highly correlated features (threshold={threshold}).")
        return self.data

    def pca_reduction(self, variance_threshold=0.95):
        if self.data is None:
            print("No data loaded.")
            return None

        numeric_data = self.data.select_dtypes(include=["float64", "int64"])
        pca = PCA(n_components=variance_threshold)
        reduced_values = pca.fit_transform(numeric_data)

        reduced_df = pd.DataFrame(
            reduced_values,
            columns=[f"PC{i+1}" for i in range(reduced_values.shape[1])]
        )

        non_numeric = self.data.select_dtypes(exclude=["float64", "int64"])
        self.data = pd.concat([non_numeric.reset_index(drop=True), reduced_df], axis=1)

        print(f"📉 PCA reduced data to {reduced_df.shape[1]} components "
              f"({variance_threshold*100:.0f}% variance retained).")
        return self.data

    def save_reduced_data(self):
        if self.data is not None:
            self.data.to_csv(self.output_path, index=False)
            print(f"Saved reduced data to {self.output_path}")
            return self.output_path
        else:
            print("No data to save.")
            return None


# Example usage
# def main():
#     reducer = DataReducer()
#     reducer.load_data()
#     reducer.correlation_reduction(threshold=0.9)
#     # or reducer.pca_reduction(variance_threshold=0.95)
#     reducer.save_reduced_data()
#
# if __name__ == "__main__":
#     main()
