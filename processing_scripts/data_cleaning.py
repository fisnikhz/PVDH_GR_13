# processing_scripts/data_cleaning.py
import pandas as pd
import numpy as np
from pathlib import Path
 #file needs work
#class DataCleaner:
#    def __init__(self, input_dir="../unprocessed_datasets", output_dir="../processed_datasets"):
 #       self.input_dir = Path(input_dir)
  #      self.output_dir = Path(output_dir)
   #     self.data = None

    #def load_data(self, pattern="*.csv"):
     #   files = list(self.input_dir.glob(pattern))
      #  if not files:
       #     print("No files found in unprocessed_datasets.")
        #    return None

        #dfs = []
        #for file in files:
         #   try:
          #      df = pd.read_csv(file)
           #     df["source_file"] = file.stem
            #    dfs.append(df)
             #   print(f"Loaded {file.name}")
            #except Exception as e:
             #   print(f"Error loading {file.name}: {e}")

        #self.data = pd.concat(dfs, ignore_index=True)
        #print(f"Combined {len(files)} files, total {len(self.data)} rows.")
       # return self.data

    #def clean_column_names(self):
     #   self.data.columns = [col.strip().lower().replace(" ", "_") for col in self.data.columns]
      #  print("Standardized column names.")
       # return self.data

    #def handle_missing_values(self, strategy="mean"):
     #   num_cols = self.data.select_dtypes(include=[np.number]).columns
      #  for col in num_cols:
       #     if strategy == "mean":
        #        self.data[col].fillna(self.data[col].mean(), inplace=True)
         #   elif strategy == "median":
          #      self.data[col].fillna(self.data[col].median(), inplace=True)
        #self.data.fillna("Unknown", inplace=True)
       # print("Handled missing values.")
        #return self.data

    #def remove_duplicates(self):
     #   before = len(self.data)
      #  self.data.drop_duplicates(inplace=True)
       # print(f"Removed {before - len(self.data)} duplicate rows.")
        #return self.data

    #def handle_outliers(self, z_thresh=3):
     #   numeric_cols = self.data.select_dtypes(include=[np.number]).columns
      #  for col in numeric_cols:
       #     z_scores = np.abs((self.data[col] - self.data[col].mean()) / self.data[col].std())
        #    self.data = self.data[z_scores < z_thresh]
        #print("Removed outliers using Z-score threshold.")
       # return self.data

    #def save_cleaned_data(self, filename="cleaned_data.csv"):
     #   output_path = self.output_dir / filename
      #  self.data.to_csv(output_path, index=False)
       # print(f"Saved cleaned data to {output_path}")
       # return output_path


#def main():
 #   cleaner = DataCleaner()
  #  df = cleaner.load_data()
   # if df is not None:
    #    cleaner.clean_column_names()
     #   cleaner.handle_missing_values(strategy="mean")
      #  cleaner.remove_duplicates()
       # cleaner.handle_outliers()
        #cleaner.save_cleaned_data()

#if __name__ == "__main__":
 #   main()
