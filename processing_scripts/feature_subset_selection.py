import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from data_preprocessing import DataPreprocessor

class FeatureSelector:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def select_top_features(self, target_column: str, k: int = 5) -> pd.DataFrame:
        numeric_cols = self.data.select_dtypes(include=[float, int]).columns
        X = self.data[numeric_cols].drop(columns=[target_column], errors='ignore')
        y = self.data[target_column] if target_column in self.data.columns else None

        if y is None:
            print(f"Target column '{target_column}' not found. Skipping feature selection.")
            return self.data

        selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
        selector.fit(X, y)
        selected_columns = X.columns[selector.get_support()]

        print(f"Selected top {len(selected_columns)} features based on '{target_column}': {list(selected_columns)}")
        return self.data[selected_columns.tolist() + [target_column]]


def main():
    preprocessor = DataPreprocessor()
    data_file = "../unprocessed_datasets/Crimes_2024.csv"
    preprocessor.load_data(data_file)
    
    if preprocessor.data is not None:
        preprocessor.assess_data_quality()
        preprocessor.clean_data()
        preprocessor.handle_missing_values(strategy='mean')
        
        selector = FeatureSelector(preprocessor.data)
        selected_data = selector.select_top_features(target_column='Arrest', k=5)
        
        output_file = "../processed_datasets/crimes_2024_selected_features.csv"
        selected_data.to_csv(output_file, index=False)
        print(f"\nSelected features data saved to: {output_file}")

if __name__ == "__main__":
    main()
