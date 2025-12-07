<table>
  <tr>
    <td width="150" align="center" valign="center">
      <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/University_of_Prishtina_logo.svg/1200px-University_of_Prishtina_logo.svg.png" width="120" alt="University Logo" />
    </td>
    <td valign="top">
      <p><strong>Universiteti i Prishtinës</strong></p>
      <p>Fakulteti i Inxhinierisë Elektrike dhe Kompjuterike</p>
      <p>Inxhinieri Kompjuterike dhe Softuerike - Programi Master</p>
      <p><strong>Projekti nga lënda:</strong> “Përgatitja dhe vizualizimi i të dhënave”</p>
      <p><strong>Profesor:</strong> PhD Mërgim Hoti</p>
      <p><strong>Studentët (Gr. 13):</strong></p>
      <ul>
        <li>Fisnik Hazrolli</li>
        <li>Altin Pajaziti</li>
        <li>Olta Pllana</li>
        <li>Rajmondë Shllaku</li>
      </ul>
    </td>
  </tr>
</table>

---

## Table of Contents

- [Project Overview](#-project-overview)
- [Repository Structure](#-repository-structure)
- [Dataset Description](#-dataset-description)
- [Implemented Modules](#-implemented-modules)
- [Technologies Used](#-technologies-used)
- [Installation & Setup](#-installation--setup)
- [Results](#-results)
- [Key Takeaways](#-key-takeaways)
---

## Project Overview

This repository implements a comprehensive data preparation pipeline for the **Chicago Crimes from 2001 to Present Dataset**. 
The project demonstrates advanced data science techniques including cleaning, feature engineering, aggregation, sampling, scaling, and dimensionality reduction to transform raw historical data into analysis-ready formats.

### Project Goals
- Implement end-to-end data preprocessing pipelines
- Apply feature engineering techniques to enhance data quality
- Demonstrate multiple data transformation strategies
- Perform statistical analysis and aggregation
- Handle large-scale real-world datasets

### Repository Structure
This is a **data preparation project** that implements various preprocessing, transformation, and analysis techniques on Chicago crime data spanning 25 years (2001-2025). The repository includes:



```
PVDH_GR_13/
│
├── unprocessed_datasets/                # Raw crime data (2001-2025)
│   ├── Crimes_2001.csv                  # ~300K+ records per file
│   ├── Crimes_2002.csv
│   ├── ...
│   └── Crimes_2025.csv
│
├── processed_datasets/                  # Cleaned and transformed data
│   └── crimes_2024_processed.csv        # Sample processed output
│
├── processing_scripts/                  # Modular data processing scripts
│   ├── plots/                           # All visuals in pipeline
│   ├── data_cleaning.py                 # Cleaning and quality improvement
│   ├── data_integration.py              # Multi-file integration and merging
│   ├── data_preprocessing.py            # General preprocessing functions
│   ├── data_preprocessing_pipeline.py   # Main pipeline orchestrating all steps
│   ├── data_exploring.py                # Exploration of dataset
│   ├── anomaly_detection.py             # Statistical (IQR)
│   ├── data_aggregation.py              # Temporal/spatial/categorical aggregation
│   ├── feature_engineering.py           # Feature creation and transformation
│   ├── feature_subset_selection.py      # Feature selection algorithms
│   ├── data_scaling.py                  # Normalization and scaling
│   ├── sampling_techniques.py           # Random, systematic, stratified sampling
│   └── discretization_binarization.py   # Binning and binarization
│
├── LICENSE                              # MIT License
├── README.md                            # Project documentation
└── .gitignore                           # Git ignore rules
```

### Directory Breakdown

**`unprocessed_datasets/`**
- Contains 25 CSV files (Crimes_2001.csv through Crimes_2025.csv)
- Raw data from Chicago crime database
- Each file contains 100K-400K crime records
- Total dataset size: ~8.4 million+ crime records

**`processed_datasets/`**
- Output directory for cleaned and processed data
- Contains transformed datasets after applying various techniques
- Example: `crimes_2024_processed.csv` (preprocessed 2024 data)

**`processing_scripts/`**
- Python modules implementing different data processing techniques
- Each module is self-contained and reusable
- Can be used independently or as part of a pipeline
- 2 Preprocessing pipeline scripts implementing different flows
- Includes example usage in `main()` functions

---
## Dataset Description

### Data Attributes (22 columns)

| Column | Type | Description |
|--------|------|-------------|
| `ID` | Integer | Unique identifier for each crime record |
| `Case Number` | String | Official case tracking number |
| `Date` | DateTime | Date and time of incident (MM/DD/YYYY HH:MM:SS AM/PM) |
| `Block` | String | Partial street address (privacy-protected) |
| `IUCR` | String | Illinois Uniform Crime Reporting code |
| `Primary Type` | String | Primary classification (THEFT, ASSAULT, BATTERY, etc.) |
| `Description` | String | Detailed crime description |
| `Location Description` | String | Type of location (STREET, RESIDENCE, APARTMENT, etc.) |
| `Arrest` | Boolean | Whether arrest was made (true/false) |
| `Domestic` | Boolean | Domestic-related incident (true/false) |
| `Beat` | Integer | Police beat area number |
| `District` | Integer | Police district number (1-25) |
| `Ward` | Integer | City ward number (1-50) |
| `Community Area` | Integer | Community area code (1-77) |
| `FBI Code` | String | FBI crime classification code |
| `X Coordinate` | Float | Illinois State Plane coordinate |
| `Y Coordinate` | Float | Illinois State Plane coordinate |
| `Year` | Integer | Year of the crime |
| `Updated On` | DateTime | Last database update timestamp |
| `Latitude` | Float | Geographic latitude |
| `Longitude` | Float | Geographic longitude |
| `Location` | String | Lat/Long in string format |

---

## Data Processing Modules

This project implements a comprehensive data processing workflow consisting of multiple stages:

| **Step** | **Stage**                    | **Description**                                    | **Key Actions**                                                                                                                                                                                         | **Module**                       |
|----------|------------------------------|----------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------|
| **1**    | **Data Collection**          | Load raw CSV crime datasets                        | - 25 years (2001–2025) <br> - ~300K records per year                                                                                                                                                    | —                                |
| **2**    | **Data Integration**         | Merge multiple yearly files into a unified dataset | - Combine datasets <br> - Add source tracking column <br> - Handle missing file errors                                                                                                                  | `data_integration.py`            |
| **3**    | **Data Quality Assessment**  | Analyze quality of raw data                        | - Missing value % <br> - Duplicate detection <br> - Memory usage analysis <br> - Generate reports                                                                                                       | `data_preprocessing.py`          |
| **4**    | **Data Cleaning**            | Clean and standardize dataset                      | - Standardize column names <br> - Remove duplicates (2–3%) <br> - Handle missing values (5–10%) <br> - Outlier removal (Z-score) <br> - Normalize text                                                  | `data_cleaning.py`               |
| **5**    | **Data Type Conversion**     | Optimize memory and type usage                     | - Convert dates to datetime <br> - Convert boolean fields <br> - Optimize numerics (int32/float32)                                                                                                      | `data_preprocessing.py`          |
| **6**    | **Feature Engineering**      | Create new meaningful features                     | **Temporal:** hour, month, quarter, is_weekend <br> **Spatial:** grid, distance_from_center <br> **Interaction:** feature combos <br> **Math:** log, sqrt, polynomial <br> **Encoding:** label, frequency | `feature_engineering.py`         |
| **7**    | **Data Transformation**      | Transform continuous and numerical data            | **Discretization:** equal-width, equal-frequency, k-means, custom <br> **Binarization:** threshold-based                                                                                                | `discretization_binarization.py` |
| **8**    | **Feature Selection**        | Reduce feature set to most important variables     | - SelectKBest <br> - RFE <br> - Random Forest feature importance                                                                                                                                        | `feature_subset_selection.py`    |
| **9**    | **Data Scaling**             | Normalize numeric features                         | - StandardScaler (Z-score) <br> - MinMaxScaler (0–1)                                                                                                                                                    | `data_scaling.py`                |
| **10**   | **Sampling**                 | Reduce dataset size when needed                    | - Random <br> - Systematic <br> - Stratified <br> - Percentage-based sampling                                                                                                                           | `sampling_techniques.py`         |
| **11**   | **Aggregation**              | Summarize data at different granularities          | **Temporal:** daily, monthly, yearly <br> **Spatial:** grid-based <br> **Categorical:** crime type, location <br> **Stats:** count, mean, sum, std                                                      | `aggregation.py`                 |
| **12**   | **Dimensionality Reduction** | Reduce feature dimensionality                      | - PCA <br> - Variance analysis                                                                                                                                                                          | `data_preprocessing.py`          |
| **13**   | **Anomaly Detection**        | Detect Anomalies in dataset                        | - Statistical: Z-Score, IQR, Modified Z-Score (MAD).                                                                                                                                                    | `anomaly_detection.py`           |
| **14**   | **Data Exploration**         | Data Quality Score                                 | comprehensive text-based report on data health.                                                                                                                                                         | `data_exploration.py`            |


## Implemented Modules
The **DataPreprocessor** class in the **data_preprocessing_pipeline.py** orchestrates the entire pipeline. 
Below is a detailed breakdown of each module's responsibility:

1. **Data Integration (integrate_unprocessed_csvs)**

* **Functionality:** Iteratively loads all CSV files matching the pattern (e.g., Crimes_20*.csv) from the input directory.
* **Logic:** Uses pd.read_csv(low_memory=False) to prevent type inference errors and appends a source_file column to track the origin year of each record.
* **Output:** A single concatenated DataFrame containing all historical data. 

<img src="ReadMe_Images/integration.png"></img>


2. **Sampling Decision (choose_sample_or_full)**

* **Functionality:** Offers an interactive choice to reduce dataset size for rapid prototyping and testing.
* **Logic:**
  * **Full Processing:** If selected, processes all ~8.4 million records.
  * **Random Sampling:** If selected (default), extracts a random subset of $N$ rows (default $N=5000$) using sample(n=n, random_state=42).
  * **Reproducibility:** A fixed random seed ensures the sample remains consistent across different runs.
<img src="ReadMe_Images/sample.png"></img>

3. **Quality Assessment (assess_data_quality)**

* **Functionality:** Performs a health check on the raw dataset.
* **Metrics:** Calculates total memory usage (MB), identifies columns with missing values, and counts duplicate rows based on ID.
<img src="ReadMe_Images/assess.png"></img>

4. **Outlier Detection (detect_outliers_iqr)**
* **Method:** Interquartile Range (IQR).
* **Logic:** Defines bounds as $[Q1 - 1.5 \times IQR, Q3 + 1.5 \times IQR]$.
* **Action:** Iterates through all numeric columns (like X Coordinate) and prints a count of records falling outside these statistical bounds. 
<img src="ReadMe_Images/outliers.png"></img>

5. **Exploratory Data Analysis (EDA) (explore_data)**
* **Functionality:** Generates statistical summaries to understand data distribution.
* **Outputs:** Descriptive 
* **Statistics:** Mean, Std, Min, Max for all numeric features.
* **Categorical Summary:** Unique value counts for text columns.
* **Correlation Matrix:** A Pearson correlation table showing relationships between numerical variables (e.g., correlation between Latitude and District).

<img src="ReadMe_Images/eda.png"></img>
<img src="ReadMe_Images/eda2.png"></img>
<img src="ReadMe_Images/skewnessdetection.png"></img>

6. **Adaptive Skewness Correction (`handle_skewed_features`)**
* **Functionality:** A "smart" diagnostic step that dynamically analyzes feature distributions before scaling.
* **Logic:**
    * Calculates the skewness coefficient for all continuous numeric variables.
    * **Threshold:** If skewness > 1.0 (highly skewed), it automatically applies a **Log Transformation** (`np.log1p`) to normalize the distribution.
    * **Safety Mechanism:** Explicitly excludes spatial coordinates (`Latitude`, `Longitude`) and cyclic time units (`Hour`) to prevent logical corruption of physical data.

<img src="ReadMe_Images/handleskewness.png"></img>

7. **Data Cleaning (clean_data)**

* **Functionality:** Fixes structural errors in the dataset.
* **Steps:**
  * **Deduplication:** Removes exact duplicate records.
  * **Text Normalization**: Strips leading/trailing whitespace from strings.
  * **Date Parsing**: Converts the string Date column into datetime objects and extracts Year, Month, Day.
  * **Imputation:** Fills NaN values in "Ward" and "Community Area" with "UNKNOWN".
<img src="ReadMe_Images/cleaning.png"></img>
  
8. **Feature Engineering (create_features)**
* **Functionality:** Derives new predictive features from raw data.
* **New Features:** Hour & DayOfWeek: Extracted from the parsed timestamp.
* **DistanceFromCenter:** Calculates Euclidean distance from Chicago's center ($41.8781, -87.6298$).
* **IsViolent:** Binary flag (1/0) checking if PrimaryType contains keywords like "HOMICIDE", "BATTERY", or "ASSAULT".
<img src="ReadMe_Images/featurengineering.png"></img>

9. **Data Filtering (remove_incorrect_values)**
* **Functionality:** Removes logically impossible data points after feature creation.
* **Rules:**
  * Latitude must be $[-90, 90]$.
  * Longitude must be $[-180, 180]$.
  * Year must be between $1990$ and $2030$.
  * DistanceFromCenter must be non-negative.
<img src="ReadMe_Images/incorrect.png"></img>

10. **Normalization (normalize_numeric_minmax)**

* **Method:** MinMax Scaling.
* **Logic:** Transforms numeric features (excluding targets/IDs) to the range $[0, 1]$ using the formula $X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}}$.
<img src="ReadMe_Images/normalization.png"></img>

11. **Categorical Encoding (encode_categoricals)**

* **Method**: Label Encoding.
* **Logic:** Converts low-cardinality categorical strings (unique values $\le 50$) into integer labels (e.g., "THEFT" $\rightarrow$ 4).
<img src="ReadMe_Images/encoding.png"></img>

12. **Aggregation (aggregate_monthly_and_type_counts)**

* **Functionality:** Adds context to individual rows based on monthly trends.
* **Features:**
  * **MonthlyCrimeCount:** Total crimes occurring in that specific month/year.
  * **TypeMonthlyCount:** Count of that specific crime type in that month.
<img src="ReadMe_Images/aggregation.png"></img>

13. **Discretization (discretize_numeric)**

* **Method:** Quantile Binning.
* **Logic:** Uses KBinsDiscretizer to sort continuous variables into 5 equal-frequency bins (e.g., X Coordinate_bin).
<img src="ReadMe_Images/discretization.png"></img>

14. **Binarization (binarize_numeric)**

* **Method:** Median Thresholding.
* **Logic:** Converts columns like Beat and District into binary (0/1) based on whether they are above the column's median value.
<img src="ReadMe_Images/binarization.png"></img>
15. **Feature Selection (select_feature_subset)**

* **Method:** Filter Method (ANOVA F-value).
* **Logic:** Uses SelectKBest with f_classif to retain the top 20 features most strongly correlated with the Arrest target.
<img src="ReadMe_Images/selectfeatures.png"></img>
16. **Dimensionality Reduction (apply_pca)**

* **Method:** Principal Component Analysis (PCA).
* **Logic:** Standardizes the data and projects it onto 3 orthogonal components (PCA_1, PCA_2, PCA_3) to reduce dimensionality while preserving variance.
<img src="ReadMe_Images/pca.png"></img>

17. **Reporting (generate_report)**

* **Functionality:** Produces a final summary of the preprocessing pipeline to validate data integrity.
* **Metrics:**
  * **Data Shape Comparison:** Compares original vs. processed row/column counts to track data loss or feature expansion.
  * **Memory Footprint:** Calculates the final memory usage in MB.
  * **Hygiene Check:** Confirms that zero missing values and zero duplicates remain in the final dataset.
  * **Statistical Snapshot:** Prints descriptive statistics (mean, std, min, max) for the transformed features.
<img src="ReadMe_Images/generatereport1.png"></img>
<img src="ReadMe_Images/generatereport2.png"></img>
17. **Data Export (save_processed)**

* Functionality: Persists the final, cleaned, and transformed dataframe to a file for use in downstream machine learning tasks.

* Logic:
  * Directory Management: Automatically creates the output directory (../processed_datasets) if it does not exist.
  * Format Support: Supports saving as either .csv (default) or .xlsx (Excel), determined by the format parameter.
  * Output: Generates the final file CrimesChicagoDatasetPreprocessed.csv.
<img src="ReadMe_Images/export.png"></img>
---

## Technologies Used

### Core Technologies
- **Python 3.x** - Primary programming language
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning and preprocessing

### Key Libraries & Modules

**Data Processing:**
- `pandas.DataFrame` - Data structures
- `pandas.read_csv()` - CSV file reading
- `pandas.concat()` - Data merging
- `pandas.groupby()` - Aggregation

**Preprocessing:**
- `sklearn.preprocessing.StandardScaler` - Z-score normalization
- `sklearn.preprocessing.MinMaxScaler` - Min-max scaling
- `sklearn.preprocessing.LabelEncoder` - Categorical encoding
- `sklearn.preprocessing.KBinsDiscretizer` - Discretization
- `sklearn.preprocessing.Binarizer` - Binarization
- `sklearn.preprocessing.PolynomialFeatures` - Polynomial features
- `sklearn.impute.SimpleImputer` - Missing value imputation

**Feature Selection:**
- `sklearn.feature_selection.SelectKBest` - Statistical selection
- `sklearn.feature_selection.RFE` - Recursive elimination
- `sklearn.feature_selection.f_classif` - ANOVA F-test

**Dimensionality Reduction:**
- `sklearn.decomposition.PCA` - Principal Component Analysis

**Machine Learning:**
- `sklearn.ensemble.RandomForestClassifier` - Feature importance
- `sklearn.linear_model.LogisticRegression` - RFE estimator

**Utilities:**
- `pathlib.Path` - File path handling
- `warnings` - Warning suppression
- `typing` - Type hints

---

## Installation & Setup

### Prerequisites
- Python 3.7 or higher
- pip package manager
- 2GB+ free disk space (for datasets)

### Step-by-Step Installation

**1. Clone the repository:**
```bash
git clone https://github.com/fisnikhz/PVDH_GR_13.git
cd PVDH_GR_13
```

**2. Create virtual environment (recommended):**
```bash
# Create virtual environment
python3 -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

**3. Install dependencies:**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

Or create `requirements.txt`:
```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

Then install:
```bash
pip install -r requirements.txt
```

**4. Verify installation:**
```bash
python -c "import pandas, numpy, sklearn; print('✓ All packages installed successfully!')"
```

**5. Verify dataset presence:**
```bash
ls unprocessed_datasets/ | head -5
# Should show: Crimes_2001.csv, Crimes_2002.csv, etc.
```

## Results and Analysis

The preprocessing pipeline successfully ingested and transformed 25 years of Chicago crime data. 
Below is a detailed breakdown of the data transformation, quality improvements, and statistical findings.
<img src="ReadMe_Images/results.png"></img>

####  Data Quality Improvements

**Cleaning:** The pipeline identified and removed 93,324 rows containing invalid data. This included records with missing coordinates (Latitude/Longitude), invalid dates, or location data falling outside Chicago's geographic boundaries.

**Outlier Detection:** Using the Interquartile Range (IQR) method, the script flagged approximately 40,265 outliers in the X Coordinate and Longitude features. These represent crimes occurring in geographically sparse or extreme areas relative to the city center distributions.

**Imputation:** High-missingness columns like Ward (~7.3% missing) and Community Area (~7.3% missing) were successfully handled, ensuring zero null values in the final output.

####  **Feature Engineering and Dimensionality Reduction Results**

#### **Feature Engineering**

The feature space was expanded from 23 raw columns to 39 features to capture temporal, spatial, and contextual patterns.

| Category | Count | Generated Features |
| :--- | :---: | :--- |
| **Temporal** | 2 | `Hour`, `DayOfWeek` |
| **Spatial** | 1 | `DistanceFromCenter` (Calculated Euclidean distance) |
| **Contextual** | 1 | `IsViolent` (Binary flag for Homicide, Assault, Battery) |
| **Aggregations** | 2 | `MonthlyCrimeCount`, `TypeMonthlyCount` |
| **Encoding** | 3 | `PrimaryType_encoded`, `FBI_Code_encoded`, `source_file_encoded` |
| **Discretization** | 12 | `X Coordinate_bin`, `Longitude_bin`, `MonthlyCrimeCount_bin`... |
| **Binarization** | 5 | `Beat_bin01`, `District_bin01`, `DistanceFromCenter_bin01`... |

#### **Dimensionality Reduction (PCA)**
  * Applied Principal Component Analysis to the numeric features.
  * **Result:** The top 3 Principal Components captured 74.59% of the total variance in the dataset. This allows for efficient modeling with fewer variables while retaining the majority of the information.
  
#### **Feature Selection**
* Using SelectKBest with the f_classif scoring function, the pipeline isolated the top 20 features most strongly correlated with the Arrest target variable. 
* Key selected features included:
  *** Derived:** IsViolent, Hour, DistanceFromCenter
  * **Transformed:** PrimaryType_encoded, MonthlyCrimeCount, PCA_1
  * **Original:** X Coordinate, Location Description, Year
  
#### Visual Analysis

The following visualizations demonstrate the statistical properties of the processed data.

 **Feature Correlation**
 
![Correlation Heatmap](processing_scripts/plots/correlation_heatmap.png) 

*Figure 1: Heatmap showing strong negative correlation (-0.53) between District and X Coordinate.*

**Outlier Detection**

![Outlier Detection](processing_scripts/plots/outlier_boxplot_X_Coordinate.png) 

*Figure 2: IQR detection identifying spatial outliers in 'X Coordinate'.* 

**Temporal Crime Patterns:**

![Temporal Distribution](processing_scripts/plots/insight_crime_by_hour.png)
*Figure 3: Distribution of crimes by Hour. The Kernel Density Estimate (KDE) curve reveals a sharp rise in criminal activity during evening hours (18:00–22:00).*

**Dataset Span:**

![Year Distribution](processing_scripts/plots/hist_kde_Year.png)
*Figure 5: Distribution of records across years, verifying the dataset covers the full 2001–2025 range.*

**Geographic & Administrative Density:**

| **Longitude Distribution** | **District Workload** |
| :---: | :---: |
| ![Longitude Density](processing_scripts/plots/hist_kde_Longitude.png) | ![District Distribution](processing_scripts/plots/hist_kde_District.png) |
| *Figure 6: Density plot of crime Longitudes. The bimodal peaks suggest two distinct high-crime vertical zones in the city.* | *Figure 7: Crime count by Police District. The uneven distribution indicates certain districts handle significantly higher case volumes.* |

**Community Analysis:**

| **Community Area Distribution** | **Community Area Boxplot** |
| :---: | :---: |
| ![Community Area Density](processing_scripts/plots/hist_kde_Community_Area.png) | ![Community Area Spread](processing_scripts/plots/boxplot_Community_Area.png) |
| *Figure 8: Distribution of crimes across Chicago's 77 Community Areas. The multi-modal peaks indicate specific neighborhoods with consistently higher incident rates.* | *Figure 9: Boxplot of Community Areas showing the spread and concentration of data.* |

**Adaptive Skewness Correction:**

![Skewness Correction](processing_scripts/plots/skew_correction_DistanceFromCenter.png)

*Figure 7: Impact of the automated Log Transformation on 'DistanceFromCenter'. The original distribution (Red, Skew=27.89) was successfully normalized (Green) to improve model performance.*

### View All Visualizations
To maintain a concise overview, only the most significant insights are displayed above. However, the pipeline generates **over 25 detailed visualizations**, including individual histograms, boxplots, and KDEs for every numeric feature. 

You can explore the complete collection of generated images in the [**processing_scripts/plots**](processing_scripts/plots) directory.
___

## Key Takeaways

### What This Project Demonstrates

 **End-to-End Data Pipeline**
- Complete workflow from raw data to analysis-ready datasets
- Modular architecture with a centralized `DataPreprocessor` class
- Production-ready code structure with error handling

 **Data Quality Management**
- Systematic handling of missing values (Imputation & Filtering)
- Outlier detection using statistical IQR thresholds
- Logic-based data validation (Coordinate bounds, Date ranges)
- Memory optimization using specific data types

 **Feature Engineering & Selection**
- **Temporal Extraction:** Hour, DayOfWeek
- **Spatial Logic:** Calculated Euclidean distance from City Center
- **Contextual Flags:** Binary indicators for violent crimes
- **Dimensionality Reduction:** PCA capturing ~75% variance
- **Feature Selection:** ANOVA F-test (SelectKBest)

 **Multiple Preprocessing Techniques**
- **Sampling:** Efficient random sampling for prototyping
- **Scaling:** MinMax normalization for neural network readiness
- **Discretization:** Quantile-based binning for continuous variables
- **Binarization:** Median-thresholding for categorical features
- **Adaptive Preprocessing:** Implemented logic-based transformations that only trigger when statistical thresholds are met (e.g., Skewness > 1.0), avoiding unnecessary data alteration.
### Skills Demonstrated

- Python programming
- Data wrangling with pandas
- Statistical analysis
- Feature engineering
- Machine learning preprocessing
- Big data handling
- Software engineering best practices
- Documentation and reporting

---

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

##  Contributors

**Group 13** - Master's Program in Computer and Software Engineering  
Faculty of Electrical and Computer Engineering  
University of Prishtina

- **Fisnik Hazrolli**
- **Altin Pajaziti**
- **Olta Pllana**
- **Rajmondë Shllaku**

**Course:** Data Preparation and Visualization  
**Profesor:** PhD Mërgim Hoti  
**Academic Year:** 2025-2026

---

##  Acknowledgments

- **City of Chicago** - For providing open access to crime data
- **PhD Mërgim Hoti** - For course instruction and guidance
- **University of Prishtina** - For academic support and resources
- **Python Community** - For excellent open-source libraries
- **scikit-learn Team** - For comprehensive ML preprocessing tools
- **pandas Development Team** - For powerful data manipulation capabilities

---
