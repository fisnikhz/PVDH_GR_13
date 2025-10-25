import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer, Binarizer
import warnings
warnings.filterwarnings('ignore')


class DiscretizationBinarization:
    
    def __init__(self, data):
        self.data = data.copy()
        self.original_data = data.copy()
        self.discretization_info = {}
        self.binarization_info = {}
    
    def equal_width_discretization(self, column, n_bins=5, labels=None):
        if column not in self.data.columns:
            print(f"Error: Column '{column}' not found")
            return None
        
        if labels is None:
            labels = [f'Bin_{i+1}' for i in range(n_bins)]
        
        try:
            self.data[f'{column}_equal_width'] = pd.cut(
                self.data[column], 
                bins=n_bins, 
                labels=labels,
                duplicates='drop'
            )
            
            bin_edges = pd.cut(self.data[column], bins=n_bins, retbins=True, duplicates='drop')[1]
            
            self.discretization_info[f'{column}_equal_width'] = {
                'method': 'equal_width',
                'n_bins': n_bins,
                'bin_edges': bin_edges.tolist()
            }
            
            print(f"✓ Equal-width discretization applied to '{column}' ({n_bins} bins)")
            return self.data[f'{column}_equal_width']
        
        except Exception as e:
            print(f"Error during equal-width discretization: {e}")
            return None
    
    def equal_frequency_discretization(self, column, n_bins=5, labels=None):
        if column not in self.data.columns:
            print(f"Error: Column '{column}' not found")
            return None
        
        if labels is None:
            labels = [f'Quantile_{i+1}' for i in range(n_bins)]
        
        try:
            self.data[f'{column}_equal_freq'] = pd.qcut(
                self.data[column], 
                q=n_bins, 
                labels=labels,
                duplicates='drop'
            )
            
            quantile_edges = pd.qcut(self.data[column], q=n_bins, retbins=True, duplicates='drop')[1]
            
            self.discretization_info[f'{column}_equal_freq'] = {
                'method': 'equal_frequency',
                'n_bins': n_bins,
                'quantile_edges': quantile_edges.tolist()
            }
            
            print(f"✓ Equal-frequency discretization applied to '{column}' ({n_bins} bins)")
            return self.data[f'{column}_equal_freq']
        
        except Exception as e:
            print(f"Error during equal-frequency discretization: {e}")
            return None
    
    def kmeans_discretization(self, column, n_bins=5):
        if column not in self.data.columns:
            print(f"Error: Column '{column}' not found")
            return None
        
        try:
            discretizer = KBinsDiscretizer(
                n_bins=n_bins, 
                encode='ordinal', 
                strategy='kmeans'
            )
            
            values = self.data[[column]].values
            discretized = discretizer.fit_transform(values)
            
            self.data[f'{column}_kmeans'] = discretized.astype(int)
            
            self.discretization_info[f'{column}_kmeans'] = {
                'method': 'kmeans',
                'n_bins': n_bins,
                'bin_edges': discretizer.bin_edges_[0].tolist()
            }
            
            print(f"✓ K-means discretization applied to '{column}' ({n_bins} bins)")
            return self.data[f'{column}_kmeans']
        
        except Exception as e:
            print(f"Error during k-means discretization: {e}")
            return None
    
    def custom_discretization(self, column, bin_edges, labels=None):
        if column not in self.data.columns:
            print(f"Error: Column '{column}' not found")
            return None
        
        if labels is None:
            labels = [f'Custom_{i+1}' for i in range(len(bin_edges)-1)]
        
        try:
            self.data[f'{column}_custom'] = pd.cut(
                self.data[column], 
                bins=bin_edges, 
                labels=labels,
                include_lowest=True
            )
            
            self.discretization_info[f'{column}_custom'] = {
                'method': 'custom',
                'bin_edges': bin_edges
            }
            
            print(f"✓ Custom discretization applied to '{column}' ({len(bin_edges)-1} bins)")
            return self.data[f'{column}_custom']
        
        except Exception as e:
            print(f"Error during custom discretization: {e}")
            return None
    
    def binarize(self, column, threshold=0.5):
        if column not in self.data.columns:
            print(f"Error: Column '{column}' not found")
            return None
        
        try:
            binarizer = Binarizer(threshold=threshold)
            values = self.data[[column]].values
            binarized = binarizer.transform(values)
            
            self.data[f'{column}_binary'] = binarized.astype(int)
            
            self.binarization_info[f'{column}_binary'] = {
                'threshold': threshold,
                'original_column': column
            }
            
            count_ones = (self.data[f'{column}_binary'] == 1).sum()
            count_zeros = (self.data[f'{column}_binary'] == 0).sum()
            
            print(f"✓ Binarization applied to '{column}' (threshold={threshold})")
            print(f"  → 1s: {count_ones} ({count_ones/len(self.data)*100:.1f}%)")
            print(f"  → 0s: {count_zeros} ({count_zeros/len(self.data)*100:.1f}%)")
            
            return self.data[f'{column}_binary']
        
        except Exception as e:
            print(f"Error during binarization: {e}")
            return None
    
    def binarize_multiple(self, columns, threshold=0.5):
        for col in columns:
            self.binarize(col, threshold)
        
        return self.data
    
    def get_summary(self):
        print("\n" + "="*60)
        print("DISCRETIZATION & BINARIZATION SUMMARY")
        print("="*60)
        
        print(f"\nOriginal columns: {self.original_data.shape[1]}")
        print(f"Current columns: {self.data.shape[1]}")
        print(f"New columns added: {self.data.shape[1] - self.original_data.shape[1]}")
        
        if self.discretization_info:
            print(f"\n📊 Discretization operations: {len(self.discretization_info)}")
            for col, info in self.discretization_info.items():
                print(f"  • {col}: {info['method']} ({info.get('n_bins', 'N/A')} bins)")
        
        if self.binarization_info:
            print(f"\n🔢 Binarization operations: {len(self.binarization_info)}")
            for col, info in self.binarization_info.items():
                print(f"  • {col}: threshold={info['threshold']}")
        
        return {
            'discretization': self.discretization_info,
            'binarization': self.binarization_info
        }
    
    def get_data(self):
        return self.data


# Demo usage
if __name__ == "__main__":
    print("Discretization & Binarization Module Demo\n")
    
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'Age': np.random.randint(18, 80, 1000),
        'Income': np.random.normal(50000, 20000, 1000),
        'Score': np.random.uniform(0, 100, 1000),
        'Distance': np.random.exponential(10, 1000)
    })
    
    print("Sample data created:")
    print(sample_data.head())
    print(f"\nShape: {sample_data.shape}")
    
    # Initialize processor
    processor = DiscretizationBinarization(sample_data)
    
    # Apply different discretization methods
    print("\n" + "="*60)
    print("Applying discretization techniques...")
    print("="*60 + "\n")
    
    processor.equal_width_discretization('Age', n_bins=5, labels=['Young', 'Adult', 'Middle', 'Senior', 'Elder'])
    processor.equal_frequency_discretization('Income', n_bins=4, labels=['Low', 'Medium', 'High', 'VeryHigh'])
    processor.kmeans_discretization('Score', n_bins=3)
    processor.custom_discretization('Distance', bin_edges=[0, 5, 10, 20, 100], labels=['Close', 'Medium', 'Far', 'VeryFar'])
    
    # Apply binarization
    print("\n" + "="*60)
    print("Applying binarization...")
    print("="*60 + "\n")
    
    processor.binarize('Score', threshold=50)
    processor.binarize('Distance', threshold=10)
    
    # Get summary
    summary = processor.get_summary()
    
    # Show results
    print("\n" + "="*60)
    print("RESULTS PREVIEW")
    print("="*60)
    print(processor.get_data().head(10))
    
    print("\n✓ Demo completed successfully!")
