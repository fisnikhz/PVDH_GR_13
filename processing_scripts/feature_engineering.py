import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    
    def __init__(self, data):
        self.data = data.copy()
        self.original_data = data.copy()
        self.created_features = []
    
    def create_datetime_features(self, date_column):
        if date_column not in self.data.columns:
            print(f"Column '{date_column}' not found")
            return None
        
        try:
            dt = pd.to_datetime(self.data[date_column], errors='coerce')
            
            self.data[f'{date_column}_year'] = dt.dt.year
            self.data[f'{date_column}_month'] = dt.dt.month
            self.data[f'{date_column}_day'] = dt.dt.day
            self.data[f'{date_column}_hour'] = dt.dt.hour
            self.data[f'{date_column}_dayofweek'] = dt.dt.dayofweek
            self.data[f'{date_column}_quarter'] = dt.dt.quarter
            self.data[f'{date_column}_is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
            
            features = [f'{date_column}_year', f'{date_column}_month', f'{date_column}_day', 
                       f'{date_column}_hour', f'{date_column}_dayofweek', f'{date_column}_quarter',
                       f'{date_column}_is_weekend']
            
            self.created_features.extend(features)
            print(f"Created {len(features)} datetime features from '{date_column}'")
            return features
        
        except Exception as e:
            print(f"Error creating datetime features: {e}")
            return None
    
    def create_polynomial_features(self, columns, degree=2, include_bias=False):
        if not all(col in self.data.columns for col in columns):
            print("Some columns not found")
            return None
        
        try:
            poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
            poly_data = poly.fit_transform(self.data[columns])
            
            feature_names = poly.get_feature_names_out(columns)
            
            for i, name in enumerate(feature_names):
                if name not in columns:
                    self.data[name] = poly_data[:, i]
                    self.created_features.append(name)
            
            print(f"Created {len(feature_names) - len(columns)} polynomial features (degree={degree})")
            return feature_names
        
        except Exception as e:
            print(f"Error creating polynomial features: {e}")
            return None
    
    def create_interaction_features(self, col1, col2, operation='multiply'):
        if col1 not in self.data.columns or col2 not in self.data.columns:
            print("Columns not found")
            return None
        
        try:
            feature_name = f'{col1}_{operation}_{col2}'
            
            if operation == 'multiply':
                self.data[feature_name] = self.data[col1] * self.data[col2]
            elif operation == 'add':
                self.data[feature_name] = self.data[col1] + self.data[col2]
            elif operation == 'subtract':
                self.data[feature_name] = self.data[col1] - self.data[col2]
            elif operation == 'divide':
                self.data[feature_name] = self.data[col1] / (self.data[col2] + 1e-10)
            else:
                print(f"Unknown operation: {operation}")
                return None
            
            self.created_features.append(feature_name)
            print(f"Created interaction feature: {feature_name}")
            return feature_name
        
        except Exception as e:
            print(f"Error creating interaction feature: {e}")
            return None
    
    def create_aggregation_features(self, group_col, agg_col, agg_func='mean'):
        if group_col not in self.data.columns or agg_col not in self.data.columns:
            print("Columns not found")
            return None
        
        try:
            feature_name = f'{agg_col}_{agg_func}_by_{group_col}'
            
            if agg_func == 'mean':
                agg_values = self.data.groupby(group_col)[agg_col].transform('mean')
            elif agg_func == 'sum':
                agg_values = self.data.groupby(group_col)[agg_col].transform('sum')
            elif agg_func == 'count':
                agg_values = self.data.groupby(group_col)[agg_col].transform('count')
            elif agg_func == 'std':
                agg_values = self.data.groupby(group_col)[agg_col].transform('std')
            elif agg_func == 'max':
                agg_values = self.data.groupby(group_col)[agg_col].transform('max')
            elif agg_func == 'min':
                agg_values = self.data.groupby(group_col)[agg_col].transform('min')
            else:
                print(f"Unknown aggregation function: {agg_func}")
                return None
            
            self.data[feature_name] = agg_values
            self.created_features.append(feature_name)
            print(f"Created aggregation feature: {feature_name}")
            return feature_name
        
        except Exception as e:
            print(f"Error creating aggregation feature: {e}")
            return None
    
    def create_ratio_features(self, numerator_col, denominator_col):
        if numerator_col not in self.data.columns or denominator_col not in self.data.columns:
            print("Columns not found")
            return None
        
        try:
            feature_name = f'{numerator_col}_to_{denominator_col}_ratio'
            self.data[feature_name] = self.data[numerator_col] / (self.data[denominator_col] + 1e-10)
            
            self.created_features.append(feature_name)
            print(f"Created ratio feature: {feature_name}")
            return feature_name
        
        except Exception as e:
            print(f"Error creating ratio feature: {e}")
            return None
    
    def create_log_features(self, columns):
        if isinstance(columns, str):
            columns = [columns]
        
        created = []
        for col in columns:
            if col not in self.data.columns:
                print(f"Column '{col}' not found")
                continue
            
            try:
                feature_name = f'{col}_log'
                self.data[feature_name] = np.log1p(np.abs(self.data[col]))
                self.created_features.append(feature_name)
                created.append(feature_name)
            except Exception as e:
                print(f"Error creating log feature for '{col}': {e}")
        
        if created:
            print(f"Created {len(created)} log features")
        return created
    
    def create_sqrt_features(self, columns):
        if isinstance(columns, str):
            columns = [columns]
        
        created = []
        for col in columns:
            if col not in self.data.columns:
                print(f"Column '{col}' not found")
                continue
            
            try:
                feature_name = f'{col}_sqrt'
                self.data[feature_name] = np.sqrt(np.abs(self.data[col]))
                self.created_features.append(feature_name)
                created.append(feature_name)
            except Exception as e:
                print(f"Error creating sqrt feature for '{col}': {e}")
        
        if created:
            print(f"Created {len(created)} sqrt features")
        return created
    
    def create_binned_features(self, column, bins=5, labels=None):
        if column not in self.data.columns:
            print(f"Column '{column}' not found")
            return None
        
        try:
            feature_name = f'{column}_binned'
            
            if labels is None:
                labels = [f'bin_{i}' for i in range(bins)]
            
            self.data[feature_name] = pd.cut(self.data[column], bins=bins, labels=labels, duplicates='drop')
            self.created_features.append(feature_name)
            print(f"Created binned feature: {feature_name}")
            return feature_name
        
        except Exception as e:
            print(f"Error creating binned feature: {e}")
            return None
    
    def create_distance_features(self, lat_col, lon_col, ref_lat, ref_lon):
        if lat_col not in self.data.columns or lon_col not in self.data.columns:
            print("Location columns not found")
            return None
        
        try:
            feature_name = f'distance_from_reference'
            
            self.data[feature_name] = np.sqrt(
                (self.data[lat_col] - ref_lat)**2 + 
                (self.data[lon_col] - ref_lon)**2
            )
            
            self.created_features.append(feature_name)
            print(f"Created distance feature: {feature_name}")
            return feature_name
        
        except Exception as e:
            print(f"Error creating distance feature: {e}")
            return None
    
    def create_frequency_encoding(self, column):
        if column not in self.data.columns:
            print(f"Column '{column}' not found")
            return None
        
        try:
            feature_name = f'{column}_frequency'
            freq_map = self.data[column].value_counts(normalize=True).to_dict()
            self.data[feature_name] = self.data[column].map(freq_map)
            
            self.created_features.append(feature_name)
            print(f"Created frequency encoding: {feature_name}")
            return feature_name
        
        except Exception as e:
            print(f"Error creating frequency encoding: {e}")
            return None
    
    def get_summary(self):
        print("\n" + "="*60)
        print("FEATURE ENGINEERING SUMMARY")
        print("="*60)
        print(f"\nOriginal features: {self.original_data.shape[1]}")
        print(f"Current features: {self.data.shape[1]}")
        print(f"Created features: {len(self.created_features)}")
        
        if self.created_features:
            print(f"\nNew features added:")
            for i, feat in enumerate(self.created_features[:20], 1):
                print(f"  {i}. {feat}")
            if len(self.created_features) > 20:
                print(f"  ... and {len(self.created_features) - 20} more")
        
        return {
            'original_features': self.original_data.shape[1],
            'current_features': self.data.shape[1],
            'created_features': len(self.created_features),
            'feature_names': self.created_features
        }
    
    def get_data(self):
        return self.data


if __name__ == "__main__":
    print("Feature Engineering Module Demo\n")
    
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=100, freq='D'),
        'Sales': np.random.randint(100, 1000, 100),
        'Cost': np.random.randint(50, 500, 100),
        'Latitude': np.random.uniform(41.5, 42.0, 100),
        'Longitude': np.random.uniform(-88.0, -87.5, 100),
        'Category': np.random.choice(['A', 'B', 'C'], 100),
        'Temperature': np.random.uniform(0, 35, 100)
    })
    
    print("Sample data:")
    print(sample_data.head())
    
    fe = FeatureEngineer(sample_data)
    
    print("\n" + "="*60)
    print("Creating features...")
    print("="*60 + "\n")
    
    fe.create_datetime_features('Date')
    fe.create_interaction_features('Sales', 'Cost', 'multiply')
    fe.create_ratio_features('Sales', 'Cost')
    fe.create_aggregation_features('Category', 'Sales', 'mean')
    fe.create_log_features(['Sales', 'Cost'])
    fe.create_distance_features('Latitude', 'Longitude', 41.8781, -87.6298)
    fe.create_frequency_encoding('Category')
    
    summary = fe.get_summary()
    
    print("\n" + "="*60)
    print("RESULTS PREVIEW")
    print("="*60)
    print(fe.get_data().head())
    
    print("\n✓ Demo completed!")
