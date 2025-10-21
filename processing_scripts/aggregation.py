import pandas as pd
import numpy as np
from typing import List, Optional


class DataAggregator:
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.original_shape = data.shape

    def temporal_aggregation(self, date_column: str, freq: str = 'D',
                             numeric_columns: Optional[List[str]] = None) -> pd.DataFrame:
        if date_column not in self.data.columns:
            print(f"Date column '{date_column}' not found.")
            return self.data

        try:
            df_temp = self.data.copy()
            df_temp[date_column] = pd.to_datetime(df_temp[date_column], errors='coerce')

            if numeric_columns is None:
                numeric_columns = df_temp.select_dtypes(include=[np.number]).columns.tolist()

            numeric_columns = [col for col in numeric_columns if col != date_column]
            agg_dict = {col: 'sum' for col in numeric_columns}

            aggregated = df_temp.set_index(date_column).resample(freq).agg(agg_dict).reset_index()

            print(f"Temporal aggregation ({freq}): {self.original_shape[0]} rows → {len(aggregated)} rows")
            return aggregated
        except Exception as e:
            print(f"Error in temporal aggregation: {e}")
            return self.data

    def categorical_aggregation(self, group_columns: List[str],
                                numeric_columns: Optional[List[str]] = None,
                                agg_func: str = 'mean') -> pd.DataFrame:
        if not all(col in self.data.columns for col in group_columns):
            print(f"Some group columns not found.")
            return self.data

        df_temp = self.data.copy()

        if numeric_columns is None:
            numeric_columns = df_temp.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_columns = [col for col in numeric_columns if col in df_temp.columns]

        try:
            aggregated = df_temp.groupby(group_columns)[numeric_columns].agg(agg_func).reset_index()
            print(f"Categorical aggregation on {group_columns}: {self.original_shape[0]} rows → {len(aggregated)} rows")
            return aggregated
        except Exception as e:
            print(f"Error in categorical aggregation: {e}")
            return self.data

    def spatial_aggregation(self, lat_col: str, lon_col: str, grid_size: float = 0.01) -> pd.DataFrame:
        if lat_col not in self.data.columns or lon_col not in self.data.columns:
            print(f"Latitude or longitude columns not found.")
            return self.data

        try:
            df_temp = self.data.copy()
            df_temp['lat_grid'] = (df_temp[lat_col] / grid_size).astype(int) * grid_size
            df_temp['lon_grid'] = (df_temp[lon_col] / grid_size).astype(int) * grid_size

            numeric_cols = df_temp.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col not in ['lat_grid', 'lon_grid', lat_col, lon_col]]

            agg_dict = {col: 'sum' for col in numeric_cols}
            agg_dict[lat_col] = 'mean'
            agg_dict[lon_col] = 'mean'

            aggregated = df_temp.groupby(['lat_grid', 'lon_grid']).agg(agg_dict).reset_index()
            aggregated = aggregated.drop(columns=['lat_grid', 'lon_grid'])

            print(f"Spatial aggregation (grid_size={grid_size}): {self.original_shape[0]} rows → {len(aggregated)} rows")
            return aggregated
        except Exception as e:
            print(f"Error in spatial aggregation: {e}")
            return self.data

    def aggregation_by_primary_type(self, output_path: str = "../processed_datasets/aggregation_sample.csv") -> pd.DataFrame:
        if 'Primary Type' not in self.data.columns:
            print("Column 'Primary Type' not found.")
            return pd.DataFrame()

        df = self.data.dropna(subset=['Primary Type'])

        numeric_cols = ['Ward', 'District', 'Community Area', 'X Coordinate', 'Y Coordinate']
        numeric_cols = [col for col in numeric_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

        if not numeric_cols:
            print("No valid numeric columns found for aggregation.")
            return pd.DataFrame()

        aggregated = df.groupby('Primary Type')[numeric_cols].agg(['count', 'mean', 'min', 'max'])
        aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns.values]
        aggregated.to_csv(output_path)
        print(f"Saved aggregation by 'Primary Type' to: {output_path}")

        return aggregated

    def sampling_based_aggregation(self, sample_size: int, random_state: int = 42) -> pd.DataFrame:
        if sample_size >= len(self.data):
            print(f"Sample size ({sample_size}) >= dataset size ({len(self.data)}). No aggregation needed.")
            return self.data

        sampled = self.data.sample(n=sample_size, random_state=random_state)
        print(f"Sampling aggregation: {self.original_shape[0]} rows → {len(sampled)} rows")
        return sampled

def main():
    print("=== Data Aggregation Module ===")

    try:
        data = pd.read_csv("../unprocessed_datasets/Crimes_2024.csv", nrows=5000)
        print(f"Data loaded: {data.shape}")

        aggregator = DataAggregator(data)

        # # 1. Aggregation by primary type
        # primary_agg = aggregator.aggregation_by_primary_type()
        
        # # 2. Temporal aggregation (monthly)
        # temporal_agg = aggregator.temporal_aggregation(date_column='Date', freq='M')
        # temporal_agg.to_csv("../processed_datasets/temporal_aggregation.csv", index=False)

        # # 3. Categorical aggregation
        # categorical_agg = aggregator.categorical_aggregation(group_columns=['Location Description'])
        # categorical_agg.to_csv("../processed_datasets/categorical_aggregation.csv", index=False)

        # # 4. Spatial aggregation
        # spatial_agg = aggregator.spatial_aggregation(lat_col='Latitude', lon_col='Longitude', grid_size=0.01)
        # spatial_agg.to_csv("../processed_datasets/spatial_aggregation.csv", index=False)

        # # 5. Sampling aggregation
        # sampled_agg = aggregator.sampling_based_aggregation(sample_size=500)
        # sampled_agg.to_csv("../processed_datasets/sample_aggregation.csv", index=False)

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
