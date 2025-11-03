import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')

#gh sample preprocessing
class DataPreprocessor:
    def __init__(self):
        self.data = None
        self.original_data = None
        self.scaler = None
        # Basic configuration for upcoming outlier detection
        self.outlier_method = "zscore"  # or "iqr"
        self.outlier_zscore_threshold = 3.0
        self.outlier_iqr_factor = 1.5

    def load_data(self, file_path):
        try:
            self.data = pd.read_csv(file_path)
            self.original_data = self.data.copy()
            print(f"Data loaded successfully: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
            return self.data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def assess_data_quality(self):
        print(f"Dataset shape: {self.data.shape}")
        print(f"Memory usage: {self.data.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

        missing_data = self.data.isnull().sum()
        missing_percent = (missing_data / len(self.data)) * 100

        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Values': missing_data.values,
            'Percentage': missing_percent.values
        }).sort_values('Missing Values', ascending=False)

        print("\nMissing values:")
        print(missing_df[missing_df['Missing Values'] > 0])

        duplicates = self.data.duplicated().sum()
        print(f"\nDuplicates: {duplicates} rows ({duplicates / len(self.data) * 100:.2f}%)")

        return missing_df

    def handle_missing_values(self, strategy='mean'):
        missing_before = self.data.isnull().sum().sum()
        print(f"Missing values before: {missing_before}")

        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns

        if len(numeric_cols) > 0:
            if strategy in ['mean', 'median']:
                imputer = SimpleImputer(strategy=strategy)
                self.data[numeric_cols] = imputer.fit_transform(self.data[numeric_cols])

        if len(categorical_cols) > 0:
            for col in categorical_cols:
                if self.data[col].isnull().sum() > 0:
                    mode_val = self.data[col].mode()
                    if len(mode_val) > 0:
                        self.data[col].fillna(mode_val[0], inplace=True)

        missing_after = self.data.isnull().sum().sum()
        print(f"Missing values after: {missing_after}")
        return self.data

    def define_data_types(self):
        conversions = {
            'Date': 'datetime64[ns]',
            'Arrest': 'bool',
            'Domestic': 'bool',
            'Year': 'int32'
        }

        for col, dtype in conversions.items():
            if col in self.data.columns:
                try:
                    if dtype == 'datetime64[ns]':
                        self.data[col] = pd.to_datetime(self.data[col], errors='coerce')
                    elif dtype == 'bool':
                        self.data[col] = self.data[col].astype(str).str.lower() == 'true'
                    else:
                        self.data[col] = self.data[col].astype(dtype)
                except:
                    pass

        return self.data.dtypes

    def create_features(self):
        created_features = []

        if 'Date' in self.data.columns:
            self.data['Hour'] = pd.to_datetime(self.data['Date'], errors='coerce').dt.hour
            self.data['DayOfWeek'] = pd.to_datetime(self.data['Date'], errors='coerce').dt.dayofweek
            self.data['Month'] = pd.to_datetime(self.data['Date'], errors='coerce').dt.month
            created_features.extend(['Hour', 'DayOfWeek', 'Month'])

        if all(col in self.data.columns for col in ['Latitude', 'Longitude']):
            chicago_lat, chicago_lon = 41.8781, -87.6298
            self.data['DistanceFromCenter'] = np.sqrt(
                (self.data['Latitude'] - chicago_lat) ** 2 +
                (self.data['Longitude'] - chicago_lon) ** 2
            )
            created_features.append('DistanceFromCenter')

        print(f"Created {len(created_features)} new features: {created_features}")
        return created_features

    def detect_outliers(self):
        """Placeholder for numeric outlier detection logic.

        This will later be expanded to support multiple detection
        strategies (e.g. IQR and z-score) and optional removal.
        Currently, it just returns the data unchanged.
        """
        if self.data is None:
            print("No data loaded for outlier detection.")
            return None

        print("Outlier detection not yet implemented — data unchanged.")
        return self.data

    def clean_data(self):
        duplicates_before = self.data.duplicated().sum()
        self.data.drop_duplicates(inplace=True)
        duplicates_after = self.data.duplicated().sum()
        print(f"Duplicates removed: {duplicates_before} -> {duplicates_after}")

        text_cols = self.data.select_dtypes(include=['object']).columns
        for col in text_cols:
            if self.data[col].dtype == 'object':
                self.data[col] = self.data[col].astype(str).str.strip().str.upper()

        return self.data

    def sample_data(self, method='random', n=10000):
        original_size = len(self.data)

        if method == 'random':
            sampled_data = self.data.sample(n=min(n, len(self.data)), random_state=42)
            print(f"Random sampling: {len(sampled_data)} from {original_size}")
        elif method == 'systematic':
            step = len(self.data) // n
            indices = list(range(0, len(self.data), step))[:n]
            sampled_data = self.data.iloc[indices]
            print(f"Systematic sampling: {len(sampled_data)} from {original_size}")
        else:
            sampled_data = self.data

        self.data = sampled_data.copy()
        return sampled_data

    def normalize_features(self):
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            scaler = MinMaxScaler()
            self.data[numeric_cols] = scaler.fit_transform(self.data[numeric_cols])
            print(f"Normalized {len(numeric_cols)} numeric columns using MinMaxScaler")
        else:
            print("No numeric columns found for normalization")
        return self.data

    def encode_categorical(self):
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if self.data[col].nunique() < 50:
                le = LabelEncoder()
                self.data[f'{col}_encoded'] = le.fit_transform(self.data[col].astype(str))
                print(f"Encoded categorical column: {col}")
            else:
                print(f"Encoding skipped for {col} due to {self.data[col].nunique()} unique values.")
                return self.data

    def reduce_dimensionality(self, n_components=2):
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) < n_components:
            print("Not enough numeric columns for PCA.")
            return None

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data[numeric_cols])

        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(scaled_data)

        for i in range(n_components):
            self.data[f'PCA_{i + 1}'] = pca_result[:, i]

        explained_var = np.sum(pca.explained_variance_ratio_) * 100
        print(f"PCA reduction complete ({n_components} components) — Explained variance: {explained_var:.2f}%")
        return self.data

    def generate_report(self):
        print("\n" + "=" * 50)
        print("DATA PREPROCESSING REPORT")
        print("=" * 50)

        print(f"Original data: {self.original_data.shape[0]} rows, {self.original_data.shape[1]} columns")
        print(f"Processed data: {self.data.shape[0]} rows, {self.data.shape[1]} columns")

        rows_change = self.data.shape[0] - self.original_data.shape[0]
        cols_change = self.data.shape[1] - self.original_data.shape[1]

        print(f"Rows change: {rows_change:+d}")
        print(f"Columns change: {cols_change:+d}")
        print(f"Missing values: {self.data.isnull().sum().sum()}")
        print(f"Duplicates: {self.data.duplicated().sum()}")

        return {
            'original_shape': self.original_data.shape,
            'final_shape': self.data.shape,
            'missing_values': self.data.isnull().sum().sum(),
            'duplicates': self.data.duplicated().sum()
        }


def main():
    preprocessor = DataPreprocessor()

    data_file = "../processed_datasets/integrated_data.csv"
    preprocessor.load_data(data_file)

    if preprocessor.data is not None:
        preprocessor.define_data_types()
        preprocessor.assess_data_quality()
        preprocessor.sample_data(method='random', n=5000)
        preprocessor.handle_missing_values(strategy='mean')
        preprocessor.clean_data()
        preprocessor.create_features()
        preprocessor.normalize_features()
        preprocessor.encode_categorical()
        preprocessor.reduce_dimensionality(n_components=3)
        report = preprocessor.generate_report()

        output_file = "../processed_datasets/CrimesChicagoDatasetPreprocessedSample.csv"
        preprocessor.data.to_csv(output_file, index=False)
        print(f"\nProcessed data saved to: {output_file}")

    return preprocessor


if __name__ == "__main__":
    preprocessor = main()