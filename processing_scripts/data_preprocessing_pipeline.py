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
        # ---------- Sampling decision ----------
        # ---------- Assessment ----------
        # ---------- Cleaning ----------
        # ---------- Feature creation ----------
        # ---------- Encoding ----------
        # ---------- Feature subset selection ----------
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
        # ---------- Aggregation ----------
        # ---------- Discretization ----------
        # ---------- Binarization ----------
        # ---------- PCA ----------
        # ---------- Generate Report ----------
        # ---------- SAVE NEW DATASET ----------
def main():
    if __name__ == "__main__":
        preprocessor = main()


