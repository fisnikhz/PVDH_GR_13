import pandas as pd
import numpy as np
from typing import Optional, Union

class DataSampler:
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.original_size = len(data)
    
    def random_sampling(self, sample_size: int, random_state: int = 42) -> pd.DataFrame:
        sample_size = min(sample_size, len(self.data))
        sampled = self.data.sample(n=sample_size, random_state=random_state)
        print(f"Random sampling: {len(sampled)} from {self.original_size}")
        return sampled
    
    def systematic_sampling(self, sample_size: int) -> pd.DataFrame:
        if sample_size >= len(self.data):
            return self.data
        
        step = len(self.data) // sample_size
        indices = list(range(0, len(self.data), step))[:sample_size]
        sampled = self.data.iloc[indices]
        print(f"Systematic sampling: {len(sampled)} from {self.original_size}")
        return sampled
    
    def stratified_sampling(self, column: str, sample_size: int, random_state: int = 42) -> pd.DataFrame:
        if column not in self.data.columns:
            print(f"Column {column} not found. Using random sampling instead.")
            return self.random_sampling(sample_size, random_state)
        
        value_counts = self.data[column].value_counts()
        proportions = value_counts / len(self.data)
        
        sampled_parts = []
        for value, proportion in proportions.items():
            stratum_data = self.data[self.data[column] == value]
            stratum_sample_size = max(1, int(sample_size * proportion))
            stratum_sample_size = min(stratum_sample_size, len(stratum_data))
            
            if len(stratum_data) > 0:
                stratum_sample = stratum_data.sample(n=stratum_sample_size, random_state=random_state)
                sampled_parts.append(stratum_sample)
        
        sampled = pd.concat(sampled_parts, ignore_index=True)
        print(f"Stratified sampling on {column}: {len(sampled)} from {self.original_size}")
        return sampled
    
    def sample_by_percentage(self, percentage: float, random_state: int = 42) -> pd.DataFrame:
        sample_size = int(len(self.data) * percentage)
        return self.random_sampling(sample_size, random_state)

def demonstrate_sampling(data: pd.DataFrame):
    print("Sampling Techniques Demonstration")
    print("-" * 40)
    
    sampler = DataSampler(data)
    
    print(f"Original data size: {len(data)}")
    
    random_sample = sampler.random_sampling(1000)
    systematic_sample = sampler.systematic_sampling(1000)
    percentage_sample = sampler.sample_by_percentage(0.1)
    
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        stratified_sample = sampler.stratified_sampling(categorical_cols[0], 1000)
    
    print("Sampling demonstration completed.")

if __name__ == "__main__":
    try:
        sample_data = pd.read_csv("../unprocessed_datasets/Crimes_2024.csv", nrows=10000)
        demonstrate_sampling(sample_data)
    except FileNotFoundError:
        print("Sample data file not found. Create synthetic data for testing.")
        synthetic_data = pd.DataFrame({
            'id': range(5000),
            'category': np.random.choice(['A', 'B', 'C'], 5000),
            'value': np.random.normal(100, 15, 5000)
        })
        demonstrate_sampling(synthetic_data)
