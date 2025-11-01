import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, KBinsDiscretizer, Binarizer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self, unprocessed_dir="../unprocessed_datasets", processed_dir="../processed_datasets"):
        self.unprocessed_dir = unprocessed_dir
        self.processed_dir = processed_dir
        self.data = None
        self.original_data = None
        self.scaler = None
        self._feature_selector_external = None

        # ---------- Integration ----------
        def integrate_unprocessed_csvs(self, pattern="*.csv"):
            file_glob = os.path.join(self.unprocessed_dir, pattern)
            files = sorted(glob.glob(file_glob))
            if not files:
                raise FileNotFoundError(f"No CSV files found in {self.unprocessed_dir} with pattern {pattern}")

            df_list = []
            for f in files:
                try:
                    print(f"Loading {f} ...")
                    df = pd.read_csv(f, low_memory=False)
                    df['source_file'] = os.path.basename(f)
                    df_list.append(df)
                except Exception as e:
                    print(f"Warning: could not read {f}: {e}")

            self.data = pd.concat(df_list, axis=0, ignore_index=True)
            self.original_data = self.data.copy()
            print(f"Integrated {len(files)} files -> {self.data.shape[0]} rows, {self.data.shape[1]} columns")
            return self.data

        # ---------- Sampling decision ----------
        def choose_sample_or_full(self, sample_n=5000):
            choice = input(
                f"Process full dataset ({len(self.data)} rows) or sample {sample_n} rows? [full/sample] (default sample): ").strip().lower()
            if choice in ["full", "f", "no", "n"]:
                print("Processing full dataset.")
                return  # leave self.data intact
            else:
                n = min(sample_n, len(self.data))
                self.data = self.data.sample(n=n, random_state=42).reset_index(drop=True)
                print(f"Sampled {n} rows (random).")

        # ---------- Assessment ----------
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

            print("\nMissing values (top):")
            print(missing_df[missing_df['Missing Values'] > 0].head(20))

            duplicates = self.data.duplicated().sum()
            print(f"\nDuplicates: {duplicates} rows ({duplicates / len(self.data) * 100:.2f}%)")
            return missing_df

        # ---------- Cleaning ----------
        def clean_data(self):

            duplicates_before = self.data.duplicated().sum()
            self.data.drop_duplicates(inplace=True)
            duplicates_after = self.data.duplicated().sum()
            print(f"Duplicates removed: {duplicates_before} -> {duplicates_after}")

            # Standardize text columns: strip and uppercase
            text_cols = self.data.select_dtypes(include=['object']).columns
            for col in text_cols:
                # avoid converting big text blobs unintentionally
                try:
                    self.data[col] = self.data[col].astype(str).str.strip()
                except Exception:
                    pass

            if 'Date' in self.data.columns:
                self.data['Date_parsed'] = pd.to_datetime(self.data['Date'], errors='coerce')
                if 'Year' not in self.data.columns:
                    self.data['Year'] = self.data['Date_parsed'].dt.year
                if 'Month' not in self.data.columns:
                    self.data['Month'] = self.data['Date_parsed'].dt.month
                if 'Day' not in self.data.columns:
                    self.data['Day'] = self.data['Date_parsed'].dt.day
            for col in ["Ward", "Community Area"]:
                if col in self.data.columns:
                    self.data[col] = self.data[col].fillna("UNKNOWN").astype(str)

            rename_map = {
                'Primary Type': 'PrimaryType',
                'IUCR': 'IUCR',
                'FBI Code': 'FBI_Code',
                'Location Description': 'LocationDescription',
                'Latitude': 'Latitude',
                'Longitude': 'Longitude',
                'Arrest': 'Arrest',
                'Domestic': 'Domestic'
            }
            rename_map = {k: v for k, v in rename_map.items() if k in self.data.columns}
            if rename_map:
                self.data.rename(columns=rename_map, inplace=True)

            for col in ['Latitude', 'Longitude']:
                if col in self.data.columns:
                    self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

            print("Basic cleaning complete.")
            return self.data

        # ---------- Feature creation ----------
        def create_features(self):
            created = []
            if 'Date_parsed' in self.data.columns:
                self.data['Hour'] = self.data['Date_parsed'].dt.hour
                self.data['DayOfWeek'] = self.data['Date_parsed'].dt.dayofweek
                created.extend(['Hour', 'DayOfWeek'])

            if all(c in self.data.columns for c in ['Latitude', 'Longitude']):
                chicago_lat, chicago_lon = 41.8781, -87.6298
                self.data['DistanceFromCenter'] = np.sqrt(
                    (self.data['Latitude'] - chicago_lat) ** 2 +
                    (self.data['Longitude'] - chicago_lon) ** 2
                )
                created.append('DistanceFromCenter')

            if 'PrimaryType' in self.data.columns:
                violent_keywords = ['HOMICIDE', 'ROBBERY', 'ASSAULT', 'BATTERY', 'CRIM SEXUAL ASSAULT']
                self.data['IsViolent'] = self.data['PrimaryType'].astype(str).str.upper().isin(violent_keywords).astype(int)
                created.append('IsViolent')

            print(f"Created features: {created}")
            return created

        def normalize_numeric_minmax(self, numeric_cols=None, exclude_cols=None):
            if exclude_cols is None:
                exclude_cols = [
                    'ID', 'CaseNumber', 'RecordID', 'Arrest', 'Domestic', 'IsViolent',
                    'Year', 'Month', 'Day', 'Hour', 'DayOfWeek'
                ]

            if numeric_cols is None:
                numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()

            numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

            if not numeric_cols:
                print("No numeric columns found for normalization.")
                return []

            scaler = MinMaxScaler()
            self.data[numeric_cols] = scaler.fit_transform(self.data[numeric_cols])
            print(f"Normalized {len(numeric_cols)} numeric columns using MinMaxScaler: {numeric_cols}")

            return numeric_cols

        # ---------- Encoding ----------
        def encode_categoricals(self, max_unique_for_label=50):
            cat_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
            encoded_cols = []
            for col in cat_cols:
                nunique = self.data[col].nunique(dropna=True)
                if nunique == 0:
                    continue
                if nunique <= max_unique_for_label:
                    le = LabelEncoder()
                    self.data[f"{col}_encoded"] = le.fit_transform(self.data[col].astype(str))
                    encoded_cols.append(f"{col}_encoded")
                else:
                    print(f"Skipping label encoding for {col} (unique={nunique})")
            print(f"Encoded columns: {encoded_cols}")
            return encoded_cols

        # ---------- Feature subset selection ----------
        def select_feature_subset(self, target_column=None, k=20):
            if self._feature_selector_external is not None:
                try:
                    print("Running external feature selector...")
                    selected_df = self._feature_selector_external(self.data.copy())
                    if isinstance(selected_df, pd.DataFrame):
                        self.data = selected_df
                        print("External selector returned a dataframe; replaced data.")
                        return self.data.columns.tolist()
                    elif isinstance(selected_df, (list, tuple, np.ndarray)):
                        print(f"External selector returned feature list: {len(selected_df)} features")
                        keep = [c for c in self.data.columns if c in selected_df]
                        self.data = self.data[keep].copy()
                        return keep
                except Exception as e:
                    print(f"External selector failed: {e}\nFalling back to internal selector.")

            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            if target_column and target_column in self.data.columns:
                print(f"Selecting top {k} features using SelectKBest with target '{target_column}'")
                tmp = self.data.dropna(subset=[target_column])
                X = tmp[numeric_cols].fillna(0)
                y = tmp[target_column]
                try:
                    skb = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
                    skb.fit(X, y)
                    mask = skb.get_support()
                    chosen = [c for c, m in zip(numeric_cols, mask) if m]
                except Exception:
                    chosen = sorted(numeric_cols, key=lambda c: self.data[c].var(), reverse=True)[:k]
            else:
                print(f"No target provided. Selecting top {k} numeric features by variance.")
                chosen = sorted(numeric_cols, key=lambda c: self.data[c].var(), reverse=True)[:k]

            keep_cols = chosen + [c for c in self.data.columns if c not in numeric_cols]
            self.data = self.data[keep_cols].copy()
            print(f"Selected features: {chosen}")
            return chosen
        # ---------- Aggregation ----------

        # ---------- Aggregation ----------
def aggregate_monthly_and_type_counts(self):
    if 'Year' not in self.data.columns or 'Month' not in self.data.columns:
        # try to extract from Date_parsed
        if 'Date_parsed' in self.data.columns:
            self.data['Year'] = self.data['Date_parsed'].dt.year
            self.data['Month'] = self.data['Date_parsed'].dt.month
        else:
            print("No Year/Month available for aggregation; skipping aggregation.")
            return []

    self.data['YearMonth'] = self.data['Year'].astype(str) + "-" + self.data['Month'].astype(str).str.zfill(2)

    monthly_counts = self.data.groupby('YearMonth').size().rename('MonthlyCrimeCount').reset_index()
    self.data = self.data.merge(monthly_counts, on='YearMonth', how='left')

    created = ['MonthlyCrimeCount']
    if 'PrimaryType' in self.data.columns:
        type_month_counts = self.data.groupby(['YearMonth', 'PrimaryType']).size().rename(
            'TypeMonthlyCount').reset_index()
        self.data = self.data.merge(type_month_counts, on=['YearMonth', 'PrimaryType'], how='left')
        created.append('TypeMonthlyCount')

    print(f"Aggregation complete. Created features: {created}")
    return created


# ---------- Discretization ----------
def discretize_numeric(self, numeric_cols=None, n_bins=5, strategy='quantile', exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = [
            'ID', 'CaseNumber', 'RecordID', 'Arrest', 'Domestic', 'IsViolent',
            'Year', 'Month', 'Day', 'Hour', 'DayOfWeek'
        ]

    if numeric_cols is None:
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()

    # Filter columns
    numeric_cols = [
        c for c in numeric_cols
        if c not in exclude_cols and self.data[c].nunique() > n_bins
    ]

    if not numeric_cols:
        print("No numeric columns suitable for discretization.")
        return []

    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
    discretized = est.fit_transform(self.data[numeric_cols].fillna(0))
    for i, col in enumerate(numeric_cols):
        self.data[f"{col}_bin"] = discretized[:, i].astype(int)

    print(f"Discretized {len(numeric_cols)} columns into {n_bins} bins (strategy={strategy}).")
    return [f"{c}_bin" for c in numeric_cols]


# ---------- Binarization ----------
def binarize_numeric(self):
    numeric_cols = ['Beat', 'District', 'DistanceFromCenter', 'MonthlyCrimeCount', 'TypeMonthlyCount']

    medians = self.data[numeric_cols].median()
    binarized = self.data[numeric_cols].apply(lambda col: (col > medians[col.name]).astype(int))
    for c in numeric_cols:
        self.data[f"{c}_bin01"] = binarized[c]

    print(f"Binarized {len(numeric_cols)} numeric columns using per-column median threshold.")
    return [f"{c}_bin01" for c in numeric_cols]


# ---------- PCA ----------
def apply_pca(self, n_components=3, numeric_cols=None, exclude_cols=None):

    if exclude_cols is None:
        exclude_cols = ['ID', 'CaseNumber', 'RecordID', 'Arrest', 'Domestic', 'IsViolent']

    if numeric_cols is None:
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()

    numeric_cols = [c for c in numeric_cols if c not in exclude_cols and self.data[c].nunique() > 2]

    if len(numeric_cols) < n_components:
        print(f"Not enough numeric columns for PCA (found {len(numeric_cols)}, need {n_components}).")
        return []

    scaler = StandardScaler()
    X = self.data[numeric_cols].fillna(0)
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(X_scaled)
    for i in range(n_components):
        self.data[f'PCA_{i + 1}'] = pcs[:, i]

    explained = np.sum(pca.explained_variance_ratio_) * 100
    print(
        f"PCA applied: {n_components} components, explained variance {explained:.2f}% on {len(numeric_cols)} columns")
    return [f'PCA_{i + 1}' for i in range(n_components)]
        # ---------- Discretization ----------
        # ---------- Binarization ----------
        # ---------- PCA ----------
        # ---------- Generate Report ----------
        # ---------- SAVE NEW DATASET ----------
def main():
    if __name__ == "__main__":
        preprocessor = main()


