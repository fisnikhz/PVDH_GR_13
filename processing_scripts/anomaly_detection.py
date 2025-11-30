# processing_scripts/anomaly_detection.py
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class AnomalyDetector:
    """
    Comprehensive anomaly detection module for crime data preprocessing.
    
    Implements multiple detection methods:
    - Z-Score based detection
    - IQR (Interquartile Range) based detection  
    - Isolation Forest
    - Local Outlier Factor (LOF)
    - DBSCAN-based detection
    
    Example usage:
        detector = AnomalyDetector()
        detector.load_data("../unprocessed_datasets/Crimes_2024.csv")
        detector.detect_zscore_anomalies(threshold=3.0)
        detector.detect_isolation_forest_anomalies()
        detector.get_summary()
        detector.save_cleaned_data()
    """
    
    def __init__(self, input_path=None, output_path="../processed_datasets/anomaly_cleaned_data.csv"):
        self.input_path = Path(input_path) if input_path else None
        self.output_path = Path(output_path)
        self.data = None
        self.original_data = None
        self.anomaly_results = {}
        self.detected_anomalies = {}
        self.numeric_cols = []
        
    def load_data(self, file_path=None):
        """Load data from CSV file."""
        if file_path:
            self.input_path = Path(file_path)
            
        if self.input_path is None or not self.input_path.exists():
            print(f"File not found: {self.input_path}")
            return None
            
        try:
            self.data = pd.read_csv(self.input_path)
            self.original_data = self.data.copy()
            self.numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            print(f"Loaded data: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
            print(f"Numeric columns available: {len(self.numeric_cols)}")
            return self.data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def set_data(self, data):
        """Set data directly from DataFrame."""
        self.data = data.copy()
        self.original_data = data.copy()
        self.numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        print(f"Data set: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        print(f"Numeric columns available: {len(self.numeric_cols)}")
        return self.data
    
    def _get_numeric_data(self, columns=None, exclude_cols=None):
        """Get numeric columns for analysis, optionally excluding specific columns."""
        if columns is None:
            columns = self.numeric_cols.copy()
        
        if exclude_cols is None:
            exclude_cols = ['ID', 'Case Number', 'Year', 'X Coordinate', 'Y Coordinate']
        
        columns = [c for c in columns if c in self.data.columns and c not in exclude_cols]
        return columns
    
    # ==================== Z-Score Method ====================
    def detect_zscore_anomalies(self, columns=None, threshold=3.0, exclude_cols=None):
        """
        Detect anomalies using Z-Score method.
        
        Points with |z-score| > threshold are considered anomalies.
        
        Parameters:
            columns: list of columns to analyze (default: all numeric)
            threshold: z-score threshold (default: 3.0)
            exclude_cols: columns to exclude from analysis
            
        Returns:
            DataFrame with anomaly flags
        """
        if self.data is None:
            print("No data loaded.")
            return None
        
        columns = self._get_numeric_data(columns, exclude_cols)
        if not columns:
            print("No numeric columns found for Z-Score analysis.")
            return None
        
        print(f"\n{'='*60}")
        print("Z-SCORE ANOMALY DETECTION")
        print(f"{'='*60}")
        print(f"Threshold: |z-score| > {threshold}")
        print(f"Analyzing {len(columns)} columns...")
        
        anomaly_mask = pd.Series([False] * len(self.data))
        column_anomalies = {}
        
        for col in columns:
            col_data = self.data[col].dropna()
            if len(col_data) == 0 or col_data.std() == 0:
                continue
                
            z_scores = np.abs((self.data[col] - col_data.mean()) / col_data.std())
            col_anomalies = z_scores > threshold
            anomaly_mask = anomaly_mask | col_anomalies.fillna(False)
            column_anomalies[col] = col_anomalies.sum()
        
        self.data['zscore_anomaly'] = anomaly_mask.astype(int)
        total_anomalies = anomaly_mask.sum()
        
        print(f"\nResults:")
        print(f"  Total anomalies detected: {total_anomalies} ({total_anomalies/len(self.data)*100:.2f}%)")
        
        # Show top columns with anomalies
        top_cols = sorted(column_anomalies.items(), key=lambda x: x[1], reverse=True)[:5]
        if top_cols:
            print(f"\n  Top columns with anomalies:")
            for col, count in top_cols:
                if count > 0:
                    print(f"    - {col}: {count} anomalies")
        
        self.anomaly_results['zscore'] = {
            'total_anomalies': total_anomalies,
            'threshold': threshold,
            'column_anomalies': column_anomalies
        }
        self.detected_anomalies['zscore'] = anomaly_mask
        
        return self.data
    
    # ==================== IQR Method ====================
    def detect_iqr_anomalies(self, columns=None, factor=1.5, exclude_cols=None):
        """
        Detect anomalies using Interquartile Range (IQR) method.
        
        Points outside [Q1 - factor*IQR, Q3 + factor*IQR] are anomalies.
        
        Parameters:
            columns: list of columns to analyze (default: all numeric)
            factor: IQR multiplier (default: 1.5, use 3.0 for extreme outliers)
            exclude_cols: columns to exclude from analysis
            
        Returns:
            DataFrame with anomaly flags
        """
        if self.data is None:
            print("No data loaded.")
            return None
        
        columns = self._get_numeric_data(columns, exclude_cols)
        if not columns:
            print("No numeric columns found for IQR analysis.")
            return None
        
        print(f"\n{'='*60}")
        print("IQR ANOMALY DETECTION")
        print(f"{'='*60}")
        print(f"Factor: {factor} (bounds = Q1/Q3 ± {factor}*IQR)")
        print(f"Analyzing {len(columns)} columns...")
        
        anomaly_mask = pd.Series([False] * len(self.data))
        column_anomalies = {}
        
        for col in columns:
            col_data = self.data[col].dropna()
            if len(col_data) == 0:
                continue
                
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR == 0:  # All values in same range
                continue
            
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            col_anomalies = (self.data[col] < lower_bound) | (self.data[col] > upper_bound)
            anomaly_mask = anomaly_mask | col_anomalies.fillna(False)
            column_anomalies[col] = col_anomalies.sum()
        
        self.data['iqr_anomaly'] = anomaly_mask.astype(int)
        total_anomalies = anomaly_mask.sum()
        
        print(f"\nResults:")
        print(f"  Total anomalies detected: {total_anomalies} ({total_anomalies/len(self.data)*100:.2f}%)")
        
        top_cols = sorted(column_anomalies.items(), key=lambda x: x[1], reverse=True)[:5]
        if top_cols:
            print(f"\n  Top columns with anomalies:")
            for col, count in top_cols:
                if count > 0:
                    print(f"    - {col}: {count} anomalies")
        
        self.anomaly_results['iqr'] = {
            'total_anomalies': total_anomalies,
            'factor': factor,
            'column_anomalies': column_anomalies
        }
        self.detected_anomalies['iqr'] = anomaly_mask
        
        return self.data
    
    # ==================== Isolation Forest ====================
    def detect_isolation_forest_anomalies(self, columns=None, contamination=0.1, 
                                          n_estimators=100, exclude_cols=None, random_state=42):
        """
        Detect anomalies using Isolation Forest algorithm.
        
        Isolation Forest isolates anomalies by randomly selecting features
        and splitting values, requiring fewer splits for anomalies.
        
        Parameters:
            columns: list of columns to analyze (default: all numeric)
            contamination: expected proportion of anomalies (default: 0.1)
            n_estimators: number of trees in forest (default: 100)
            exclude_cols: columns to exclude from analysis
            random_state: random seed for reproducibility
            
        Returns:
            DataFrame with anomaly flags and scores
        """
        if self.data is None:
            print("No data loaded.")
            return None
        
        columns = self._get_numeric_data(columns, exclude_cols)
        if not columns:
            print("No numeric columns found for Isolation Forest analysis.")
            return None
        
        print(f"\n{'='*60}")
        print("ISOLATION FOREST ANOMALY DETECTION")
        print(f"{'='*60}")
        print(f"Contamination: {contamination} ({contamination*100:.1f}%)")
        print(f"Estimators: {n_estimators}")
        print(f"Analyzing {len(columns)} columns...")
        
        # Prepare data (handle missing values)
        X = self.data[columns].copy()
        X = X.fillna(X.median())
        
        # Scale data for better performance
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        
        predictions = iso_forest.fit_predict(X_scaled)
        scores = iso_forest.decision_function(X_scaled)
        
        # -1 = anomaly, 1 = normal
        anomaly_mask = predictions == -1
        
        self.data['isolation_forest_anomaly'] = (predictions == -1).astype(int)
        self.data['isolation_forest_score'] = scores
        
        total_anomalies = anomaly_mask.sum()
        
        print(f"\nResults:")
        print(f"  Total anomalies detected: {total_anomalies} ({total_anomalies/len(self.data)*100:.2f}%)")
        print(f"  Anomaly scores range: [{scores.min():.4f}, {scores.max():.4f}]")
        if (~anomaly_mask).any():
            print(f"  Mean score (normal): {scores[~anomaly_mask].mean():.4f}")
        if anomaly_mask.any():
            print(f"  Mean score (anomaly): {scores[anomaly_mask].mean():.4f}")
        
        self.anomaly_results['isolation_forest'] = {
            'total_anomalies': total_anomalies,
            'contamination': contamination,
            'n_estimators': n_estimators,
            'score_range': (scores.min(), scores.max())
        }
        self.detected_anomalies['isolation_forest'] = anomaly_mask
        
        return self.data
    
    # ==================== Local Outlier Factor ====================
    def detect_lof_anomalies(self, columns=None, n_neighbors=20, contamination=0.1, exclude_cols=None):
        """
        Detect anomalies using Local Outlier Factor (LOF).
        
        LOF compares local density of a point with neighbors' densities.
        Lower density relative to neighbors indicates an anomaly.
        
        Parameters:
            columns: list of columns to analyze (default: all numeric)
            n_neighbors: number of neighbors for density estimation (default: 20)
            contamination: expected proportion of anomalies (default: 0.1)
            exclude_cols: columns to exclude from analysis
            
        Returns:
            DataFrame with anomaly flags and LOF scores
        """
        if self.data is None:
            print("No data loaded.")
            return None
        
        columns = self._get_numeric_data(columns, exclude_cols)
        if not columns:
            print("No numeric columns found for LOF analysis.")
            return None
        
        print(f"\n{'='*60}")
        print("LOCAL OUTLIER FACTOR (LOF) ANOMALY DETECTION")
        print(f"{'='*60}")
        print(f"Neighbors: {n_neighbors}")
        print(f"Contamination: {contamination} ({contamination*100:.1f}%)")
        print(f"Analyzing {len(columns)} columns...")
        
        # Prepare data
        X = self.data[columns].copy()
        X = X.fillna(X.median())
        
        # Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit LOF
        lof = LocalOutlierFactor(
            n_neighbors=min(n_neighbors, len(X) - 1),
            contamination=contamination,
            n_jobs=-1
        )
        
        predictions = lof.fit_predict(X_scaled)
        scores = lof.negative_outlier_factor_
        
        anomaly_mask = predictions == -1
        
        self.data['lof_anomaly'] = (predictions == -1).astype(int)
        self.data['lof_score'] = scores
        
        total_anomalies = anomaly_mask.sum()
        
        print(f"\nResults:")
        print(f"  Total anomalies detected: {total_anomalies} ({total_anomalies/len(self.data)*100:.2f}%)")
        print(f"  LOF scores range: [{scores.min():.4f}, {scores.max():.4f}]")
        if (~anomaly_mask).any():
            print(f"  Mean LOF (normal): {scores[~anomaly_mask].mean():.4f}")
        if anomaly_mask.any():
            print(f"  Mean LOF (anomaly): {scores[anomaly_mask].mean():.4f}")
        
        self.anomaly_results['lof'] = {
            'total_anomalies': total_anomalies,
            'n_neighbors': n_neighbors,
            'contamination': contamination,
            'score_range': (scores.min(), scores.max())
        }
        self.detected_anomalies['lof'] = anomaly_mask
        
        return self.data
    
    # ==================== DBSCAN-based Detection ====================
    def detect_dbscan_anomalies(self, columns=None, eps=0.5, min_samples=5, exclude_cols=None):
        """
        Detect anomalies using DBSCAN clustering.
        
        Points not belonging to any cluster (label = -1) are considered anomalies.
        
        Parameters:
            columns: list of columns to analyze (default: all numeric)
            eps: maximum distance between samples in same neighborhood (default: 0.5)
            min_samples: minimum samples in neighborhood for core point (default: 5)
            exclude_cols: columns to exclude from analysis
            
        Returns:
            DataFrame with anomaly flags and cluster labels
        """
        if self.data is None:
            print("No data loaded.")
            return None
        
        columns = self._get_numeric_data(columns, exclude_cols)
        if not columns:
            print("No numeric columns found for DBSCAN analysis.")
            return None
        
        print(f"\n{'='*60}")
        print("DBSCAN ANOMALY DETECTION")
        print(f"{'='*60}")
        print(f"Epsilon (eps): {eps}")
        print(f"Min samples: {min_samples}")
        print(f"Analyzing {len(columns)} columns...")
        
        # Prepare data
        X = self.data[columns].copy()
        X = X.fillna(X.median())
        
        # Scale data (important for DBSCAN)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = dbscan.fit_predict(X_scaled)
        
        # Points with label -1 are noise/anomalies
        anomaly_mask = labels == -1
        
        self.data['dbscan_anomaly'] = anomaly_mask.astype(int)
        self.data['dbscan_cluster'] = labels
        
        total_anomalies = anomaly_mask.sum()
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        print(f"\nResults:")
        print(f"  Total anomalies (noise): {total_anomalies} ({total_anomalies/len(self.data)*100:.2f}%)")
        print(f"  Number of clusters found: {n_clusters}")
        
        if n_clusters > 0:
            cluster_sizes = pd.Series(labels[labels != -1]).value_counts().sort_index()
            print(f"  Cluster sizes: {dict(cluster_sizes.head(5))}")
        
        if total_anomalies == len(self.data):
            print("  Warning: All points marked as noise. Try increasing eps value.")
        
        self.anomaly_results['dbscan'] = {
            'total_anomalies': total_anomalies,
            'eps': eps,
            'min_samples': min_samples,
            'n_clusters': n_clusters
        }
        self.detected_anomalies['dbscan'] = anomaly_mask
        
        return self.data
    
    # ==================== Modified Z-Score (MAD) ====================
    def detect_mad_anomalies(self, columns=None, threshold=3.5, exclude_cols=None):
        """
        Detect anomalies using Modified Z-Score based on Median Absolute Deviation (MAD).
        
        More robust than standard Z-Score for data with extreme outliers.
        Points with |modified z-score| > threshold are anomalies.
        
        Parameters:
            columns: list of columns to analyze (default: all numeric)
            threshold: modified z-score threshold (default: 3.5)
            exclude_cols: columns to exclude from analysis
            
        Returns:
            DataFrame with anomaly flags
        """
        if self.data is None:
            print("No data loaded.")
            return None
        
        columns = self._get_numeric_data(columns, exclude_cols)
        if not columns:
            print("No numeric columns found for MAD analysis.")
            return None
        
        print(f"\n{'='*60}")
        print("MODIFIED Z-SCORE (MAD) ANOMALY DETECTION")
        print(f"{'='*60}")
        print(f"Threshold: |modified z-score| > {threshold}")
        print(f"Analyzing {len(columns)} columns...")
        
        anomaly_mask = pd.Series([False] * len(self.data))
        column_anomalies = {}
        
        for col in columns:
            col_data = self.data[col].dropna()
            if len(col_data) == 0:
                continue
                
            median = col_data.median()
            mad = np.median(np.abs(col_data - median))
            
            if mad == 0:
                continue
            
            # Modified Z-Score formula
            modified_z = 0.6745 * (self.data[col] - median) / mad
            col_anomalies = np.abs(modified_z) > threshold
            anomaly_mask = anomaly_mask | col_anomalies.fillna(False)
            column_anomalies[col] = col_anomalies.sum()
        
        self.data['mad_anomaly'] = anomaly_mask.astype(int)
        total_anomalies = anomaly_mask.sum()
        
        print(f"\nResults:")
        print(f"  Total anomalies detected: {total_anomalies} ({total_anomalies/len(self.data)*100:.2f}%)")
        
        top_cols = sorted(column_anomalies.items(), key=lambda x: x[1], reverse=True)[:5]
        if top_cols:
            print(f"\n  Top columns with anomalies:")
            for col, count in top_cols:
                if count > 0:
                    print(f"    - {col}: {count} anomalies")
        
        self.anomaly_results['mad'] = {
            'total_anomalies': total_anomalies,
            'threshold': threshold,
            'column_anomalies': column_anomalies
        }
        self.detected_anomalies['mad'] = anomaly_mask
        
        return self.data
    
    # ==================== Combined Detection ====================
    def detect_combined_anomalies(self, methods=None, min_votes=2):
        """
        Combine multiple anomaly detection methods using voting.
        
        A point is flagged as anomaly if detected by at least min_votes methods.
        
        Parameters:
            methods: list of methods to combine (default: all detected)
            min_votes: minimum methods that must flag a point (default: 2)
            
        Returns:
            DataFrame with combined anomaly flag
        """
        if not self.detected_anomalies:
            print("No anomaly detection has been run yet.")
            return None
        
        if methods is None:
            methods = list(self.detected_anomalies.keys())
        
        available_methods = [m for m in methods if m in self.detected_anomalies]
        
        if len(available_methods) < min_votes:
            print(f"Not enough methods available ({len(available_methods)}) for min_votes={min_votes}")
            return None
        
        print(f"\n{'='*60}")
        print("COMBINED ANOMALY DETECTION (VOTING)")
        print(f"{'='*60}")
        print(f"Methods used: {available_methods}")
        print(f"Minimum votes required: {min_votes}")
        
        # Count votes for each point
        votes = pd.DataFrame({m: self.detected_anomalies[m] for m in available_methods})
        vote_counts = votes.sum(axis=1)
        
        combined_anomalies = vote_counts >= min_votes
        self.data['combined_anomaly'] = combined_anomalies.astype(int)
        self.data['anomaly_votes'] = vote_counts
        
        total_anomalies = combined_anomalies.sum()
        
        print(f"\nResults:")
        print(f"  Total combined anomalies: {total_anomalies} ({total_anomalies/len(self.data)*100:.2f}%)")
        print(f"\n  Vote distribution:")
        for v in range(len(available_methods) + 1):
            count = (vote_counts == v).sum()
            pct = count / len(self.data) * 100
            print(f"    {v} votes: {count} records ({pct:.2f}%)")
        
        self.anomaly_results['combined'] = {
            'total_anomalies': total_anomalies,
            'methods_used': available_methods,
            'min_votes': min_votes
        }
        self.detected_anomalies['combined'] = combined_anomalies
        
        return self.data
    
    # ==================== Remove Anomalies ====================
    def remove_anomalies(self, method='combined', keep_flagged=False):
        """
        Remove detected anomalies from the dataset.
        
        Parameters:
            method: which detection method's results to use
            keep_flagged: if True, keep anomaly flag columns; if False, remove them
            
        Returns:
            Cleaned DataFrame without anomalies
        """
        if method not in self.detected_anomalies:
            print(f"Method '{method}' has not been run. Available: {list(self.detected_anomalies.keys())}")
            return self.data
        
        anomaly_mask = self.detected_anomalies[method]
        rows_before = len(self.data)
        
        self.data = self.data[~anomaly_mask].reset_index(drop=True)
        rows_after = len(self.data)
        
        print(f"\nRemoved {rows_before - rows_after} anomalies using '{method}' method")
        print(f"Dataset size: {rows_before} → {rows_after} rows")
        
        if not keep_flagged:
            anomaly_cols = [col for col in self.data.columns if 'anomaly' in col.lower() 
                          or 'score' in col.lower() or 'votes' in col.lower() 
                          or 'cluster' in col.lower()]
            self.data.drop(columns=anomaly_cols, inplace=True, errors='ignore')
            print(f"Removed {len(anomaly_cols)} anomaly flag columns")
        
        return self.data
    
    # ==================== Summary & Reporting ====================
    def get_summary(self):
        """Generate comprehensive summary of all anomaly detection results."""
        print("\n" + "="*70)
        print("ANOMALY DETECTION SUMMARY REPORT")
        print("="*70)
        
        if self.original_data is not None:
            print(f"\nDataset Information:")
            print(f"  Original rows: {len(self.original_data)}")
            print(f"  Current rows: {len(self.data)}")
            print(f"  Numeric columns analyzed: {len(self.numeric_cols)}")
        
        if not self.anomaly_results:
            print("\nNo anomaly detection methods have been run yet.")
            return None
        
        print(f"\nDetection Methods Applied: {len(self.anomaly_results)}")
        print("-"*70)
        
        summary_data = []
        for method, results in self.anomaly_results.items():
            total = results.get('total_anomalies', 0)
            pct = total / len(self.original_data) * 100 if self.original_data is not None else 0
            
            print(f"\n{method.upper()}:")
            print(f"  Anomalies detected: {total} ({pct:.2f}%)")
            
            for key, value in results.items():
                if key not in ['total_anomalies', 'column_anomalies']:
                    print(f"  {key}: {value}")
            
            summary_data.append({
                'Method': method,
                'Anomalies': total,
                'Percentage': f"{pct:.2f}%"
            })
        
        print("\n" + "="*70)
        print("COMPARISON TABLE")
        print("="*70)
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        return self.anomaly_results
    
    def get_anomaly_indices(self, method='combined'):
        """Get indices of detected anomalies for a specific method."""
        if method not in self.detected_anomalies:
            print(f"Method '{method}' not found. Available: {list(self.detected_anomalies.keys())}")
            return None
        
        return self.data[self.detected_anomalies[method]].index.tolist()
    
    def get_data(self):
        """Return the current state of the data."""
        return self.data
    
    def reset(self):
        """Reset detector to original data state, clearing all anomaly results."""
        if self.original_data is not None:
            self.data = self.original_data.copy()
            self.anomaly_results = {}
            self.detected_anomalies = {}
            print("Detector reset to original data state.")
        else:
            print("No original data to reset to.")
    
    def quick_summary(self):
        """Print a quick one-line summary of anomaly counts per method."""
        if not self.anomaly_results:
            print("No detection methods have been run.")
            return
        
        print("\nQuick Summary: ", end="")
        parts = []
        for method, results in self.anomaly_results.items():
            count = results.get('total_anomalies', 0)
            parts.append(f"{method}={count}")
        print(" | ".join(parts))
    
    def save_cleaned_data(self, filename=None, include_anomaly_flags=True):
        """Save the processed data to CSV."""
        if self.data is None:
            print("No data to save.")
            return None
        
        if filename:
            output_path = Path(filename)
        else:
            output_path = self.output_path
        
        data_to_save = self.data.copy()
        
        if not include_anomaly_flags:
            anomaly_cols = [col for col in data_to_save.columns if 'anomaly' in col.lower() 
                          or 'score' in col.lower() or 'votes' in col.lower()
                          or 'cluster' in col.lower()]
            data_to_save.drop(columns=anomaly_cols, inplace=True, errors='ignore')
        
        data_to_save.to_csv(output_path, index=False)
        print(f"\nSaved data to: {output_path.resolve()}")
        print(f"Shape: {data_to_save.shape[0]} rows × {data_to_save.shape[1]} columns")
        
        return output_path


def main():
    """Example usage of AnomalyDetector with Chicago Crime data."""
    print("="*70)
    print("ANOMALY DETECTION MODULE - DEMO")
    print("="*70)
    
    # Initialize detector
    detector = AnomalyDetector()
    
    # Load sample data from processed datasets
    data_file = "../processed_datasets/CrimesChicagoDatasetPreprocessed.csv"
    detector.load_data(data_file)
    
    if detector.data is None:
        print("Could not load data. Please check the file path.")
        return None
    
    # Take a sample for faster demonstration (if dataset is large)
    if len(detector.data) > 10000:
        detector.data = detector.data.sample(n=10000, random_state=42).reset_index(drop=True)
        detector.original_data = detector.data.copy()
        print(f"\nSampled 10,000 rows for demonstration")
    
    # Define columns to exclude from anomaly detection (IDs, text columns, etc.)
    exclude_columns = ['ID', 'Case Number', 'Year', 'Updated On', 'source_file']
    
    # Run multiple anomaly detection methods
    print("\n" + "="*70)
    print("RUNNING ANOMALY DETECTION METHODS...")
    print("="*70)
    
    # 1. Z-Score detection
    detector.detect_zscore_anomalies(threshold=3.0, exclude_cols=exclude_columns)
    
    # 2. IQR detection
    detector.detect_iqr_anomalies(factor=1.5, exclude_cols=exclude_columns)
    
    # 3. Modified Z-Score (MAD)
    detector.detect_mad_anomalies(threshold=3.5, exclude_cols=exclude_columns)
    
    # 4. Isolation Forest
    detector.detect_isolation_forest_anomalies(contamination=0.05, exclude_cols=exclude_columns)
    
    # 5. Local Outlier Factor
    detector.detect_lof_anomalies(n_neighbors=20, contamination=0.05, exclude_cols=exclude_columns)
    
    # 6. DBSCAN
    detector.detect_dbscan_anomalies(eps=0.5, min_samples=5, exclude_cols=exclude_columns)
    
    # 7. Combined detection (voting)
    detector.detect_combined_anomalies(min_votes=3)
    
    # Generate summary report
    detector.get_summary()
    
    # Save results with anomaly flags
    detector.save_cleaned_data(
        "../processed_datasets/crimes_with_anomaly_flags.csv",
        include_anomaly_flags=True
    )
    
    # Optionally remove anomalies and save clean data
    print("\n" + "="*70)
    print("CREATING CLEAN DATASET (REMOVING ANOMALIES)")
    print("="*70)
    
    # Create a copy and remove anomalies
    clean_detector = AnomalyDetector()
    clean_detector.set_data(detector.original_data)
    clean_detector.detected_anomalies = detector.detected_anomalies.copy()
    clean_detector.remove_anomalies(method='combined', keep_flagged=False)
    clean_detector.save_cleaned_data("../processed_datasets/crimes_anomaly_cleaned.csv")
    
    print("\n" + "="*70)
    print("ANOMALY DETECTION DEMO COMPLETED!")
    print("="*70)
    
    return detector


if __name__ == "__main__":
    detector = main()

