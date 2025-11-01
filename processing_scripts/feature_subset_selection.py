import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from data_preprocessing import DataPreprocessor

class FeatureSelector:
    def __init__(self, data=None, csv_path=None):
        if csv_path is not None:
            self.data = pd.read_csv(csv_path)
        elif data is not None:
            self.data = data.copy()
        else:
            self.data = None

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

        print(f"Selected top {len(selected_columns)} features based on '{target_column}' (SelectKBest): {list(selected_columns)}")
        return self.data[selected_columns.tolist() + [target_column]]

    def select_features_rfe(self, target_column: str, k: int = 5) -> pd.DataFrame:
        numeric_cols = self.data.select_dtypes(include=[float, int]).columns
        X = self.data[numeric_cols].drop(columns=[target_column], errors='ignore')
        y = self.data[target_column] if target_column in self.data.columns else None

        if y is None:
            print(f"Target column '{target_column}' not found. Skipping RFE feature selection.")
            return self.data

        estimator = LogisticRegression(max_iter=1000)
        selector = RFE(estimator, n_features_to_select=min(k, X.shape[1]))
        selector.fit(X, y)
        selected_columns = X.columns[selector.support_]

        print(f"Selected top {len(selected_columns)} features based on '{target_column}' (RFE): {list(selected_columns)}")
        return self.data[selected_columns.tolist() + [target_column]]

    def select_features_rf_importance(self, target_column: str, k: int = 5) -> pd.DataFrame:
        numeric_cols = self.data.select_dtypes(include=[float, int]).columns
        X = self.data[numeric_cols].drop(columns=[target_column], errors='ignore')
        y = self.data[target_column] if target_column in self.data.columns else None

        if y is None:
            print(f"Target column '{target_column}' not found. Skipping Random Forest feature selection.")
            return self.data

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
        selected_columns = feature_importances.nlargest(min(k, X.shape[1])).index

        print(f"Selected top {len(selected_columns)} features based on '{target_column}' (Random Forest): {list(selected_columns)}")
        return self.data[selected_columns.tolist() + [target_column]]

# def main():
#     preprocessor = DataPreprocessor()
#     data_file = "../unprocessed_datasets/Crimes_2024.csv"
#     preprocessor.load_data(data_file)
#
#     if preprocessor.data is not None:
#         preprocessor.assess_data_quality()
#         preprocessor.clean_data()
#         preprocessor.handle_missing_values(strategy='mean')
#
#         selector = FeatureSelector(preprocessor.data)
#
#         selected_data_kbest = selector.select_top_features(target_column='Arrest', k=5)
#         output_file_kbest = "../processed_datasets/crimes_2024_selected_features_kbest.csv"
#         selected_data_kbest.to_csv(output_file_kbest, index=False)
#         print(f"\nSelected features (SelectKBest) saved to: {output_file_kbest}")
#
#         selected_data_rfe = selector.select_features_rfe(target_column='Arrest', k=5)
#         output_file_rfe = "../processed_datasets/crimes_2024_selected_features_rfe.csv"
#         selected_data_rfe.to_csv(output_file_rfe, index=False)
#         print(f"Selected features (RFE) saved to: {output_file_rfe}")
#
#         selected_data_rf = selector.select_features_rf_importance(target_column='Arrest', k=5)
#         output_file_rf = "../processed_datasets/crimes_2024_selected_features_rf.csv"
#         selected_data_rf.to_csv(output_file_rf, index=False)
#         print(f"Selected features (Random Forest) saved to: {output_file_rf}")
#
# if __name__ == "__main__":
#     main()

def main():
    # Path to the feature-engineered CSV
    fe_csv = "../processed_datasets/feature_engineered.csv"

    # Load the feature-engineered data directly
    selector = FeatureSelector(csv_path=fe_csv)

    # Target column (your label)
    target_col = 'Arrest'

    # Select top features using different methods
    selected_data_kbest = selector.select_top_features(target_column=target_col, k=5)
    selected_data_rfe = selector.select_features_rfe(target_column=target_col, k=5)
    selected_data_rf = selector.select_features_rf_importance(target_column=target_col, k=5)

    # Save the results
    selected_data_kbest.to_csv("../processed_datasets/selected_features_kbest.csv", index=False)
    selected_data_rfe.to_csv("../processed_datasets/selected_features_rfe.csv", index=False)
    selected_data_rf.to_csv("../processed_datasets/selected_features_rf.csv", index=False)

    print("Feature selection complete. CSVs saved to processed_datasets.")
