# processing_scripts/data_exploration.py
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class DataExplorer:
    """
    Comprehensive data exploration module for crime data analysis.
    
    Provides various exploration and profiling capabilities:
    - Basic dataset information (shape, dtypes, memory)
    - Missing value analysis
    - Duplicate detection
    - Descriptive statistics
    - Distribution analysis
    - Correlation analysis
    - Categorical variable analysis
    - Outlier detection summary
    - Data quality scoring
    - Report generation
    
    Example usage:
        explorer = DataExplorer()
        explorer.load_data("../processed_datasets/CrimesChicagoDatasetPreprocessed.csv")
        explorer.basic_info()
        explorer.analyze_missing_values()
        explorer.descriptive_statistics()
        explorer.correlation_analysis()
        explorer.generate_report()
    """
    
    def __init__(self, input_path=None):
        self.input_path = Path(input_path) if input_path else None
        self.data = None
        self.exploration_results = {}
        self.numeric_cols = []
        self.categorical_cols = []
        self.datetime_cols = []
        
    def load_data(self, file_path=None, nrows=None):
        """
        Load data from CSV file.
        
        Parameters:
            file_path: path to CSV file
            nrows: number of rows to load (None for all)
            
        Returns:
            Loaded DataFrame
        """
        if file_path:
            self.input_path = Path(file_path)
            
        if self.input_path is None or not self.input_path.exists():
            print(f"File not found: {self.input_path}")
            return None
            
        try:
            self.data = pd.read_csv(self.input_path, nrows=nrows)
            self._identify_column_types()
            print(f"Data loaded successfully: {self.data.shape[0]:,} rows × {self.data.shape[1]} columns")
            return self.data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def set_data(self, data):
        """Set data directly from DataFrame."""
        self.data = data.copy()
        self._identify_column_types()
        print(f"Data set: {self.data.shape[0]:,} rows × {self.data.shape[1]} columns")
        return self.data
    
    def _identify_column_types(self):
        """Identify numeric, categorical, and datetime columns."""
        self.numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_cols = self.data.select_dtypes(include=['datetime64']).columns.tolist()
    
    # ==================== Basic Information ====================
    def basic_info(self):
        """Display basic dataset information."""
        if self.data is None:
            print("No data loaded.")
            return None
        
        print("\n" + "="*70)
        print("BASIC DATASET INFORMATION")
        print("="*70)
        
        # Shape
        rows, cols = self.data.shape
        print(f"\nDataset Shape:")
        print(f"  Rows: {rows:,}")
        print(f"  Columns: {cols}")
        
        # Memory usage
        memory_mb = self.data.memory_usage(deep=True).sum() / (1024 ** 2)
        print(f"\nMemory Usage: {memory_mb:.2f} MB")
        
        # Column types
        print(f"\nColumn Types:")
        print(f"  Numeric: {len(self.numeric_cols)}")
        print(f"  Categorical: {len(self.categorical_cols)}")
        print(f"  Datetime: {len(self.datetime_cols)}")
        
        # Data types breakdown
        print(f"\nDetailed Data Types:")
        dtype_counts = self.data.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"  {dtype}: {count} columns")
        
        # Index info
        print(f"\nIndex Type: {type(self.data.index).__name__}")
        try:
            print(f"Index Range: {self.data.index.min()} to {self.data.index.max()}")
        except (TypeError, ValueError):
            print(f"Index Range: N/A (non-numeric index)")
        
        self.exploration_results['basic_info'] = {
            'rows': rows,
            'columns': cols,
            'memory_mb': memory_mb,
            'numeric_cols': len(self.numeric_cols),
            'categorical_cols': len(self.categorical_cols),
            'datetime_cols': len(self.datetime_cols)
        }
        
        return self.exploration_results['basic_info']
    
    def column_info(self):
        """Display detailed information about each column."""
        if self.data is None:
            print("No data loaded.")
            return None
        
        print("\n" + "="*70)
        print("COLUMN INFORMATION")
        print("="*70)
        
        column_data = []
        for col in self.data.columns:
            col_info = {
                'Column': col,
                'Type': str(self.data[col].dtype),
                'Non-Null': self.data[col].notna().sum(),
                'Null': self.data[col].isna().sum(),
                'Null%': f"{self.data[col].isna().mean()*100:.1f}%",
                'Unique': self.data[col].nunique(),
                'Memory': f"{self.data[col].memory_usage(deep=True)/1024:.1f} KB"
            }
            column_data.append(col_info)
        
        column_df = pd.DataFrame(column_data)
        print(f"\n{column_df.to_string(index=False)}")
        
        self.exploration_results['column_info'] = column_df
        return column_df
    
    # ==================== Missing Value Analysis ====================
    def analyze_missing_values(self):
        """Comprehensive missing value analysis."""
        if self.data is None:
            print("No data loaded.")
            return None
        
        print("\n" + "="*70)
        print("MISSING VALUE ANALYSIS")
        print("="*70)
        
        total_cells = self.data.size
        total_missing = self.data.isnull().sum().sum()
        total_missing_pct = (total_missing / total_cells) * 100
        
        print(f"\nOverall Missing Values:")
        print(f"  Total cells: {total_cells:,}")
        print(f"  Missing cells: {total_missing:,}")
        print(f"  Missing percentage: {total_missing_pct:.2f}%")
        
        # Per-column analysis
        missing_data = []
        for col in self.data.columns:
            missing = self.data[col].isnull().sum()
            if missing > 0:
                missing_data.append({
                    'Column': col,
                    'Missing': missing,
                    'Percentage': f"{missing/len(self.data)*100:.2f}%",
                    'Present': len(self.data) - missing
                })
        
        if missing_data:
            print(f"\nColumns with Missing Values ({len(missing_data)}):")
            missing_df = pd.DataFrame(missing_data).sort_values('Missing', ascending=False)
            print(missing_df.to_string(index=False))
        else:
            print("\n✓ No missing values found in any column!")
        
        # Missing value patterns
        rows_with_missing = self.data.isnull().any(axis=1).sum()
        complete_rows = len(self.data) - rows_with_missing
        
        print(f"\nRow-wise Missing Patterns:")
        print(f"  Complete rows: {complete_rows:,} ({complete_rows/len(self.data)*100:.1f}%)")
        print(f"  Rows with missing: {rows_with_missing:,} ({rows_with_missing/len(self.data)*100:.1f}%)")
        
        self.exploration_results['missing_values'] = {
            'total_missing': total_missing,
            'total_missing_pct': total_missing_pct,
            'columns_with_missing': len(missing_data),
            'complete_rows': complete_rows,
            'rows_with_missing': rows_with_missing
        }
        
        return self.exploration_results['missing_values']
    
    # ==================== Duplicate Analysis ====================
    def analyze_duplicates(self, subset=None):
        """Analyze duplicate rows in the dataset."""
        if self.data is None:
            print("No data loaded.")
            return None
        
        print("\n" + "="*70)
        print("DUPLICATE ANALYSIS")
        print("="*70)
        
        # Full duplicates
        full_duplicates = self.data.duplicated().sum()
        full_dup_pct = (full_duplicates / len(self.data)) * 100
        
        print(f"\nFull Row Duplicates:")
        print(f"  Duplicate rows: {full_duplicates:,}")
        print(f"  Percentage: {full_dup_pct:.2f}%")
        print(f"  Unique rows: {len(self.data) - full_duplicates:,}")
        
        # Subset duplicates (if specified)
        if subset:
            subset = [c for c in subset if c in self.data.columns]
            if subset:
                subset_duplicates = self.data.duplicated(subset=subset).sum()
                subset_dup_pct = (subset_duplicates / len(self.data)) * 100
                print(f"\nDuplicates by subset {subset}:")
                print(f"  Duplicate rows: {subset_duplicates:,}")
                print(f"  Percentage: {subset_dup_pct:.2f}%")
        
        # First duplicate examples
        if full_duplicates > 0:
            print(f"\nFirst 3 duplicate examples:")
            dup_mask = self.data.duplicated(keep=False)
            print(self.data[dup_mask].head(3).to_string())
        
        self.exploration_results['duplicates'] = {
            'full_duplicates': full_duplicates,
            'duplicate_pct': full_dup_pct,
            'unique_rows': len(self.data) - full_duplicates
        }
        
        return self.exploration_results['duplicates']
    
    # ==================== Descriptive Statistics ====================
    def descriptive_statistics(self, percentiles=None):
        """Generate descriptive statistics for numeric columns."""
        if self.data is None:
            print("No data loaded.")
            return None
        
        if not self.numeric_cols:
            print("No numeric columns found.")
            return None
        
        print("\n" + "="*70)
        print("DESCRIPTIVE STATISTICS (NUMERIC COLUMNS)")
        print("="*70)
        
        if percentiles is None:
            percentiles = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
        
        stats_df = self.data[self.numeric_cols].describe(percentiles=percentiles).T
        stats_df['missing'] = self.data[self.numeric_cols].isnull().sum()
        stats_df['missing%'] = (stats_df['missing'] / len(self.data) * 100).round(2)
        stats_df['zeros'] = (self.data[self.numeric_cols] == 0).sum()
        stats_df['zeros%'] = (stats_df['zeros'] / len(self.data) * 100).round(2)
        
        # Calculate skewness and kurtosis (handle NaN)
        stats_df['skewness'] = self.data[self.numeric_cols].skew()
        stats_df['kurtosis'] = self.data[self.numeric_cols].kurtosis()
        stats_df['skewness'] = stats_df['skewness'].fillna(0)
        stats_df['kurtosis'] = stats_df['kurtosis'].fillna(0)
        
        print(f"\nAnalyzing {len(self.numeric_cols)} numeric columns:\n")
        
        # Display basic stats
        basic_stats = stats_df[['count', 'mean', 'std', 'min', '50%', 'max']].round(4)
        print("Basic Statistics:")
        print(basic_stats.to_string())
        
        print("\n" + "-"*70)
        print("\nDistribution Metrics:")
        dist_stats = stats_df[['skewness', 'kurtosis', 'missing%', 'zeros%']].round(4)
        print(dist_stats.to_string())
        
        self.exploration_results['descriptive_stats'] = stats_df
        return stats_df
    
    # ==================== Distribution Analysis ====================
    def distribution_analysis(self, column=None, bins=10):
        """Analyze the distribution of numeric columns."""
        if self.data is None:
            print("No data loaded.")
            return None
        
        columns = [column] if column else self.numeric_cols[:5]  # Limit to first 5
        columns = [c for c in columns if c in self.numeric_cols]
        
        if not columns:
            print("No valid numeric columns specified.")
            return None
        
        print("\n" + "="*70)
        print("DISTRIBUTION ANALYSIS")
        print("="*70)
        
        results = {}
        for col in columns:
            print(f"\n--- {col} ---")
            
            data_col = self.data[col].dropna()
            
            if len(data_col) == 0:
                print(f"  No valid data (all NaN)")
                continue
            
            # Basic distribution metrics
            print(f"  Count: {len(data_col):,}")
            print(f"  Range: [{data_col.min():.4f}, {data_col.max():.4f}]")
            print(f"  Mean: {data_col.mean():.4f}")
            print(f"  Median: {data_col.median():.4f}")
            
            if len(data_col) > 1:
                std_val = data_col.std()
                print(f"  Std Dev: {std_val:.4f}")
                print(f"  Variance: {data_col.var():.4f}")
                if not np.isnan(data_col.skew()):
                    print(f"  Skewness: {data_col.skew():.4f}")
                if not np.isnan(data_col.kurtosis()):
                    print(f"  Kurtosis: {data_col.kurtosis():.4f}")
            else:
                print(f"  Std Dev: N/A (only 1 value)")
                print(f"  Variance: N/A")
                print(f"  Skewness: N/A")
                print(f"  Kurtosis: N/A")
            
            # Value distribution (histogram-like)
            if len(data_col) > 0 and data_col.min() != data_col.max():
                hist, bin_edges = np.histogram(data_col, bins=bins)
                print(f"\n  Value Distribution ({bins} bins):")
                for i in range(len(hist)):
                    pct = hist[i] / len(data_col) * 100 if len(data_col) > 0 else 0
                    bar = "█" * int(pct / 2)
                    print(f"    [{bin_edges[i]:8.2f} - {bin_edges[i+1]:8.2f}]: {hist[i]:5} ({pct:5.1f}%) {bar}")
            else:
                print(f"\n  Value Distribution: All values are identical")
            
            results[col] = {
                'count': len(data_col),
                'min': data_col.min(),
                'max': data_col.max(),
                'mean': data_col.mean(),
                'median': data_col.median(),
                'std': data_col.std(),
                'skewness': data_col.skew(),
                'kurtosis': data_col.kurtosis()
            }
        
        self.exploration_results['distribution'] = results
        return results
    
    # ==================== Correlation Analysis ====================
    def correlation_analysis(self, method='pearson', threshold=0.7):
        """Analyze correlations between numeric columns."""
        if self.data is None:
            print("No data loaded.")
            return None
        
        if len(self.numeric_cols) < 2:
            print("Need at least 2 numeric columns for correlation analysis.")
            return None
        
        print("\n" + "="*70)
        print(f"CORRELATION ANALYSIS ({method.upper()})")
        print("="*70)
        
        # Handle NaN values before correlation
        numeric_data = self.data[self.numeric_cols].copy()
        numeric_data = numeric_data.dropna()
        
        if len(numeric_data) < 2:
            print("Not enough valid data points for correlation analysis (need at least 2 rows).")
            return None
        
        corr_matrix = numeric_data.corr(method=method)
        
        print(f"\nCorrelation Matrix ({len(self.numeric_cols)} columns):")
        
        # Format correlation matrix for display
        corr_display = corr_matrix.round(3)
        if len(self.numeric_cols) <= 10:
            print(corr_display.to_string())
        else:
            print(f"Matrix too large to display fully. Showing first 10x10:")
            print(corr_display.iloc[:10, :10].to_string())
        
        # Find highly correlated pairs
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= threshold:
                    high_corr.append({
                        'Feature 1': corr_matrix.columns[i],
                        'Feature 2': corr_matrix.columns[j],
                        'Correlation': corr_val
                    })
        
        print(f"\nHighly Correlated Pairs (|r| ≥ {threshold}):")
        if high_corr:
            high_corr_df = pd.DataFrame(high_corr).sort_values('Correlation', key=abs, ascending=False)
            print(high_corr_df.to_string(index=False))
        else:
            print(f"  No pairs found with |correlation| ≥ {threshold}")
        
        # Summary statistics
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        all_corrs = upper_tri.stack().values
        all_corrs = all_corrs[~np.isnan(all_corrs)]  # Remove NaN values
        
        if len(all_corrs) > 0:
            print(f"\nCorrelation Summary:")
            print(f"  Mean absolute correlation: {np.abs(all_corrs).mean():.4f}")
            print(f"  Max correlation: {all_corrs.max():.4f}")
            print(f"  Min correlation: {all_corrs.min():.4f}")
            print(f"  Highly correlated pairs: {len(high_corr)}")
        else:
            print(f"\nCorrelation Summary: No valid correlations computed.")
        
        self.exploration_results['correlation'] = {
            'matrix': corr_matrix,
            'high_correlations': high_corr,
            'mean_abs_correlation': np.abs(all_corrs).mean()
        }
        
        return corr_matrix
    
    # ==================== Categorical Analysis ====================
    def categorical_analysis(self, max_unique=50):
        """Analyze categorical columns."""
        if self.data is None:
            print("No data loaded.")
            return None
        
        if not self.categorical_cols:
            print("No categorical columns found.")
            return None
        
        print("\n" + "="*70)
        print("CATEGORICAL VARIABLE ANALYSIS")
        print("="*70)
        
        results = {}
        for col in self.categorical_cols:
            print(f"\n--- {col} ---")
            
            unique_count = self.data[col].nunique()
            missing = self.data[col].isnull().sum()
            
            print(f"  Unique values: {unique_count}")
            print(f"  Missing: {missing} ({missing/len(self.data)*100:.1f}%)")
            
            if unique_count <= max_unique:
                value_counts = self.data[col].value_counts()
                print(f"\n  Value Distribution:")
                for i, (val, count) in enumerate(value_counts.items()):
                    if i >= 10:
                        print(f"    ... and {len(value_counts) - 10} more values")
                        break
                    pct = count / len(self.data) * 100
                    bar = "█" * int(pct / 2)
                    val_str = str(val)[:30] + "..." if len(str(val)) > 30 else str(val)
                    print(f"    {val_str:35}: {count:6} ({pct:5.1f}%) {bar}")
            else:
                print(f"  (Too many unique values to display - showing top 5)")
                top_5 = self.data[col].value_counts().head(5)
                for val, count in top_5.items():
                    pct = count / len(self.data) * 100
                    print(f"    {str(val)[:35]:35}: {count:6} ({pct:5.1f}%)")
            
            results[col] = {
                'unique_count': unique_count,
                'missing': missing,
                'top_values': self.data[col].value_counts().head(5).to_dict()
            }
        
        self.exploration_results['categorical'] = results
        return results
    
    # ==================== Outlier Summary ====================
    def outlier_summary(self, method='iqr', factor=1.5):
        """Quick outlier summary for numeric columns."""
        if self.data is None:
            print("No data loaded.")
            return None
        
        if not self.numeric_cols:
            print("No numeric columns found.")
            return None
        
        print("\n" + "="*70)
        print(f"OUTLIER SUMMARY ({method.upper()} Method)")
        print("="*70)
        
        outlier_data = []
        for col in self.numeric_cols:
            data_col = self.data[col].dropna()
            
            if len(data_col) == 0:
                continue
            
            if method == 'iqr':
                Q1 = data_col.quantile(0.25)
                Q3 = data_col.quantile(0.75)
                IQR = Q3 - Q1
                if IQR == 0:
                    continue  # Skip if all values are the same
                lower = Q1 - factor * IQR
                upper = Q3 + factor * IQR
                outliers = ((data_col < lower) | (data_col > upper)).sum()
            elif method == 'zscore':
                if data_col.std() == 0:
                    continue  # Skip if no variance
                z_scores = np.abs((data_col - data_col.mean()) / data_col.std())
                outliers = (z_scores > 3).sum()
                lower = data_col.mean() - 3 * data_col.std()
                upper = data_col.mean() + 3 * data_col.std()
            else:
                continue
            
            outlier_data.append({
                'Column': col,
                'Outliers': outliers,
                'Percentage': f"{outliers/len(data_col)*100:.2f}%",
                'Lower Bound': f"{lower:.4f}",
                'Upper Bound': f"{upper:.4f}"
            })
        
        outlier_df = pd.DataFrame(outlier_data)
        outlier_df = outlier_df.sort_values('Outliers', ascending=False)
        
        print(f"\n{outlier_df.to_string(index=False)}")
        
        total_outliers = outlier_df['Outliers'].sum()
        cols_with_outliers = (outlier_df['Outliers'] > 0).sum()
        
        print(f"\nSummary:")
        print(f"  Total outlier instances: {total_outliers:,}")
        print(f"  Columns with outliers: {cols_with_outliers}/{len(self.numeric_cols)}")
        
        self.exploration_results['outliers'] = {
            'method': method,
            'total_outliers': total_outliers,
            'cols_with_outliers': cols_with_outliers,
            'details': outlier_df
        }
        
        return outlier_df
    
    # ==================== Data Quality Score ====================
    def data_quality_score(self):
        """Calculate overall data quality score."""
        if self.data is None:
            print("No data loaded.")
            return None
        
        print("\n" + "="*70)
        print("DATA QUALITY SCORE")
        print("="*70)
        
        scores = {}
        
        # 1. Completeness (0-100)
        completeness = (1 - self.data.isnull().sum().sum() / self.data.size) * 100
        scores['Completeness'] = completeness
        
        # 2. Uniqueness (0-100) - percentage of unique rows
        uniqueness = (1 - self.data.duplicated().sum() / len(self.data)) * 100
        scores['Uniqueness'] = uniqueness
        
        # 3. Consistency (simplified - check for uniform data types)
        consistency = 100  # Start with perfect score
        for col in self.categorical_cols:
            # Check for mixed case or formatting issues
            if self.data[col].astype(str).str.contains(r'^\s|\s$', regex=True).any():
                consistency -= 5
        scores['Consistency'] = max(0, consistency)
        
        # 4. Validity (check numeric ranges)
        validity = 100
        for col in self.numeric_cols:
            # Check for infinite values
            if np.isinf(self.data[col]).any():
                validity -= 10
        scores['Validity'] = max(0, validity)
        
        # Overall score (weighted average)
        weights = {'Completeness': 0.4, 'Uniqueness': 0.2, 'Consistency': 0.2, 'Validity': 0.2}
        overall = sum(scores[k] * weights[k] for k in scores)
        
        print(f"\nQuality Dimensions:")
        for dim, score in scores.items():
            bar = "█" * int(score / 5)
            status = "✓" if score >= 90 else "△" if score >= 70 else "✗"
            print(f"  {dim:15}: {score:6.2f}% {bar} {status}")
        
        print(f"\n{'='*40}")
        print(f"  OVERALL SCORE: {overall:.2f}%")
        
        if overall >= 90:
            grade = "EXCELLENT"
        elif overall >= 80:
            grade = "GOOD"
        elif overall >= 70:
            grade = "ACCEPTABLE"
        elif overall >= 60:
            grade = "NEEDS IMPROVEMENT"
        else:
            grade = "POOR"
        
        print(f"  GRADE: {grade}")
        print(f"{'='*40}")
        
        self.exploration_results['quality_score'] = {
            'dimensions': scores,
            'overall': overall,
            'grade': grade
        }
        
        return self.exploration_results['quality_score']
    
    # ==================== Sample Data ====================
    def show_sample(self, n=5, method='head'):
        """Display sample rows from the dataset."""
        if self.data is None:
            print("No data loaded.")
            return None
        
        print("\n" + "="*70)
        print(f"SAMPLE DATA ({method.upper()}, n={n})")
        print("="*70)
        
        if method == 'head':
            sample = self.data.head(n)
        elif method == 'tail':
            sample = self.data.tail(n)
        elif method == 'random':
            sample = self.data.sample(n=min(n, len(self.data)), random_state=42)
        else:
            sample = self.data.head(n)
        
        print(f"\n{sample.to_string()}")
        return sample
    
    # ==================== Value Ranges ====================
    def value_ranges(self):
        """Display value ranges for all columns."""
        if self.data is None:
            print("No data loaded.")
            return None
        
        print("\n" + "="*70)
        print("VALUE RANGES")
        print("="*70)
        
        range_data = []
        for col in self.data.columns:
            col_data = self.data[col]
            
            if col in self.numeric_cols:
                col_numeric = pd.to_numeric(col_data, errors='coerce').dropna()
                if len(col_numeric) > 0:
                    range_data.append({
                        'Column': col,
                        'Type': 'Numeric',
                        'Min': f"{col_numeric.min():.4f}",
                        'Max': f"{col_numeric.max():.4f}",
                        'Range': f"{col_numeric.max() - col_numeric.min():.4f}"
                    })
                else:
                    range_data.append({
                        'Column': col,
                        'Type': 'Numeric',
                        'Min': 'N/A',
                        'Max': 'N/A',
                        'Range': 'N/A'
                    })
            elif col in self.categorical_cols:
                mode_val = col_data.mode()
                range_data.append({
                    'Column': col,
                    'Type': 'Categorical',
                    'Min': f"{col_data.nunique()} unique",
                    'Max': f"Most: {mode_val.iloc[0] if len(mode_val) > 0 else 'N/A'}",
                    'Range': f"N/A"
                })
            else:
                range_data.append({
                    'Column': col,
                    'Type': str(col_data.dtype),
                    'Min': str(col_data.min())[:20],
                    'Max': str(col_data.max())[:20],
                    'Range': 'N/A'
                })
        
        range_df = pd.DataFrame(range_data)
        print(f"\n{range_df.to_string(index=False)}")
        
        return range_df
    
    # ==================== Generate Full Report ====================
    def generate_report(self, save_path=None):
        """Generate comprehensive exploration report."""
        if self.data is None:
            print("No data loaded.")
            return None
        
        print("\n" + "="*70)
        print("COMPREHENSIVE DATA EXPLORATION REPORT")
        print("="*70)
        print(f"Generated for: {self.input_path}")
        print(f"Report Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run all analyses
        self.basic_info()
        self.analyze_missing_values()
        self.analyze_duplicates()
        self.descriptive_statistics()
        self.correlation_analysis()
        self.categorical_analysis()
        self.outlier_summary()
        self.data_quality_score()
        
        print("\n" + "="*70)
        print("REPORT COMPLETE")
        print("="*70)
        
        # Save report to file if path provided
        if save_path:
            report_lines = []
            report_lines.append("DATA EXPLORATION REPORT")
            report_lines.append(f"File: {self.input_path}")
            report_lines.append(f"Date: {pd.Timestamp.now()}")
            report_lines.append("")
            
            for key, value in self.exploration_results.items():
                report_lines.append(f"=== {key.upper()} ===")
                if isinstance(value, dict):
                    for k, v in value.items():
                        if not isinstance(v, (pd.DataFrame, np.ndarray)):
                            report_lines.append(f"  {k}: {v}")
                report_lines.append("")
            
            with open(save_path, 'w') as f:
                f.write('\n'.join(report_lines))
            print(f"\nReport saved to: {save_path}")
        
        return self.exploration_results
    
    def get_data(self):
        """Return the current data."""
        return self.data
    
    def get_results(self):
        """Return all exploration results."""
        return self.exploration_results
    
    def quick_info(self):
        """Print a quick one-line summary of the dataset."""
        if self.data is None:
            print("No data loaded.")
            return
        
        missing_pct = (self.data.isnull().sum().sum() / self.data.size) * 100
        duplicates = self.data.duplicated().sum()
        
        print(f"Quick Info: {self.data.shape[0]:,} rows × {self.data.shape[1]} cols | "
              f"Missing: {missing_pct:.1f}% | Duplicates: {duplicates:,} | "
              f"Numeric: {len(self.numeric_cols)} | Categorical: {len(self.categorical_cols)}")
    
    def reset(self):
        """Clear all exploration results, keeping only the data."""
        self.exploration_results = {}
        print("Exploration results cleared. Data remains loaded.")


def main():
    """Example usage of DataExplorer with Chicago Crime data."""
    print("="*70)
    print("DATA EXPLORATION MODULE - DEMO")
    print("="*70)
    
    # Initialize explorer
    explorer = DataExplorer()
    
    # Load data from processed datasets
    data_file = "../processed_datasets/CrimesChicagoDatasetPreprocessed.csv"
    explorer.load_data(data_file)
    
    if explorer.data is None:
        print("Could not load data. Please check the file path.")
        return None
    
    # Run comprehensive exploration
    print("\n" + "="*70)
    print("RUNNING DATA EXPLORATION...")
    print("="*70)
    
    # Basic information
    explorer.basic_info()
    
    # Column details
    explorer.column_info()
    
    # Missing value analysis
    explorer.analyze_missing_values()
    
    # Duplicate analysis
    explorer.analyze_duplicates()
    
    # Descriptive statistics
    explorer.descriptive_statistics()
    
    # Distribution analysis (first 3 numeric columns)
    if explorer.numeric_cols:
        explorer.distribution_analysis(column=explorer.numeric_cols[0])
    
    # Correlation analysis
    explorer.correlation_analysis(threshold=0.5)
    
    # Categorical analysis
    explorer.categorical_analysis()
    
    # Outlier summary
    explorer.outlier_summary(method='iqr')
    
    # Show sample data
    explorer.show_sample(n=5, method='random')
    
    # Data quality score
    explorer.data_quality_score()
    
    # Value ranges
    explorer.value_ranges()
    
    print("\n" + "="*70)
    print("DATA EXPLORATION COMPLETED!")
    print("="*70)
    
    return explorer


if __name__ == "__main__":
    explorer = main()

