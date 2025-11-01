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
        # ---------- Aggregation ----------
        # ---------- Discretization ----------
        # ---------- Binarization ----------
        # ---------- PCA ----------
        # ---------- Generate Report ----------
        # ---------- SAVE NEW DATASET ----------
def main():
    if __name__ == "__main__":
        preprocessor = main()


