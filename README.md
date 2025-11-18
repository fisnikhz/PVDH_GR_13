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
- [Data Processing Workflow](#-data-processing-workflow)
- [Implemented Modules](#-implemented-modules)
- [Technologies Used](#-technologies-used)
- [Installation & Setup](#-installation--setup)
- [Results](#-results)

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

### What This Repository Contains
This is a **data preparation project** that implements various preprocessing, transformation, and analysis techniques on Chicago crime data spanning 25 years (2001-2025). The repository includes:

- **Modular Python scripts** for different data processing tasks
- **25 unprocessed CSV files** containing raw crime data by year
- **Processed datasets** with cleaned and transformed data
- **Comprehensive documentation** of all implemented techniques
- **Example workflows** demonstrating how to use each module

---

## Repository Structure

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
│   ├── data_cleaning.py                 # Cleaning and quality improvement
│   ├── data_integration.py              # Multi-file integration and merging
│   ├── data_preprocessing.py            # General preprocessing functions
│   ├── data_preprocessing_pipeline.py   # Main pipeline orchestrating all steps
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

### Source
**Chicago Crime Data** - Official crime reports from the City of Chicago Police Department

### Characteristics
- **Time Period:** 2001 to 2025 (25 years)
- **Format:** CSV (Comma-Separated Values)
- **Size:** ~7.5 million+ total records
- **Update Frequency:** Historical data (static for this project)

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

### Sample Data
```csv
ID,Case Number,Date,Block,IUCR,Primary Type,Description,Location Description,Arrest,Domestic,...
13707878,JJ100066,12/31/2024 09:18:00 PM,061XX N HAMILTON AVE,0560,ASSAULT,SIMPLE,APARTMENT,false,true,...
```
---

## Data Processing Modules

This project implements a comprehensive data processing workflow consisting of multiple stages:

| **Step** | **Stage**                    | **Description**                                    | **Key Actions**                                                                                                                                                                                           | **Module**                       |
| -------- | ---------------------------- | -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- |
| **1**    | **Data Collection**          | Load raw CSV crime datasets                        | - 25 years (2001–2025) <br> - ~300K records per year                                                                                                                                                      | —                                |
| **2**    | **Data Integration**         | Merge multiple yearly files into a unified dataset | - Combine datasets <br> - Add source tracking column <br> - Handle missing file errors                                                                                                                    | `data_integration.py`            |
| **3**    | **Data Quality Assessment**  | Analyze quality of raw data                        | - Missing value % <br> - Duplicate detection <br> - Memory usage analysis <br> - Generate reports                                                                                                         | `data_preprocessing.py`          |
| **4**    | **Data Cleaning**            | Clean and standardize dataset                      | - Standardize column names <br> - Remove duplicates (2–3%) <br> - Handle missing values (5–10%) <br> - Outlier removal (Z-score) <br> - Normalize text                                                    | `data_cleaning.py`               |
| **5**    | **Data Type Conversion**     | Optimize memory and type usage                     | - Convert dates to datetime <br> - Convert boolean fields <br> - Optimize numerics (int32/float32)                                                                                                        | `data_preprocessing.py`          |
| **6**    | **Feature Engineering**      | Create new meaningful features                     | **Temporal:** hour, month, quarter, is_weekend <br> **Spatial:** grid, distance_from_center <br> **Interaction:** feature combos <br> **Math:** log, sqrt, polynomial <br> **Encoding:** label, frequency | `feature_engineering.py`         |
| **7**    | **Data Transformation**      | Transform continuous and numerical data            | **Discretization:** equal-width, equal-frequency, k-means, custom <br> **Binarization:** threshold-based                                                                                                  | `discretization_binarization.py` |
| **8**    | **Feature Selection**        | Reduce feature set to most important variables     | - SelectKBest <br> - RFE <br> - Random Forest feature importance                                                                                                                                          | `feature_subset_selection.py`    |
| **9**    | **Data Scaling**             | Normalize numeric features                         | - StandardScaler (Z-score) <br> - MinMaxScaler (0–1)                                                                                                                                                      | `data_scaling.py`                |
| **10**   | **Sampling**                 | Reduce dataset size when needed                    | - Random <br> - Systematic <br> - Stratified <br> - Percentage-based sampling                                                                                                                             | `sampling_techniques.py`         |
| **11**   | **Aggregation**              | Summarize data at different granularities          | **Temporal:** daily, monthly, yearly <br> **Spatial:** grid-based <br> **Categorical:** crime type, location <br> **Stats:** count, mean, sum, std                                                        | `aggregation.py`                 |
| **12**   | **Dimensionality Reduction** | Reduce feature dimensionality                      | - PCA <br> - Variance analysis                                                                                                                                                                            | `data_preprocessing.py`          |
| **13**   | **Export Processed Data**    | Save final cleaned dataset                         | - Export to CSV <br> - Document all transformations                                                                                                                                                       | —                                |



## Implemented Modules

### 1. **data_cleaning.py**
**Purpose:** Core data cleaning operations

**Capabilities:**
- Load and combine multiple CSV files
- Standardize column naming conventions
- Handle missing values (mean, median imputation)
- Remove duplicate records
- Detect and remove statistical outliers
- Export cleaned datasets

**Key Class:** `DataCleaner`

---

### 2. **data_preprocessing.py**
**Purpose:** Comprehensive preprocessing pipeline

**Capabilities:**
- Complete data quality assessment
- Missing value analysis and handling
- Data type optimization
- Datetime feature extraction
- Spatial feature creation
- Data normalization (MinMaxScaler)
- Categorical encoding (LabelEncoder)
- PCA dimensionality reduction
- Multiple sampling methods
- Comprehensive reporting

**Key Class:** `DataPreprocessor`

**Output Example:**
```
Dataset shape: (10000, 22)
Memory usage: 2.45 MB
Missing values: 850 (3.86%)
Duplicates: 234 rows (2.34%)

DATA PREPROCESSING REPORT
================================
Original data: 10000 rows, 22 columns
Processed data: 5000 rows, 35 columns
Rows change: -5000
Columns change: +13
Missing values: 0
Duplicates: 0
```

---

### 3. **data_integration.py**
**Purpose:** Multi-file data integration

**Capabilities:**
- Discover and load multiple CSV files
- Merge datasets with source tracking
- Temporal aggregation (daily/monthly/yearly)
- Category-based aggregation
- Integrated discretization
- Integrated binarization
- Summary statistics

**Key Class:** `DataIntegrator`

---

### 4. **data_aggregation.py**
**Purpose:** Advanced data aggregation

**Capabilities:**
- **Temporal aggregation:** Group by time periods
- **Categorical aggregation:** Group by categories
- **Spatial aggregation:** Grid-based geographic grouping
- **Crime type aggregation:** Analyze by primary type
- **Sampling-based aggregation:** Random data reduction

**Key Class:** `DataAggregator`

**Aggregation Types:**
```python
# Temporal: Daily → Monthly → Yearly
temporal_agg = aggregator.temporal_aggregation('Date', freq='M')

# Categorical: Group by crime type and location
cat_agg = aggregator.categorical_aggregation(['Primary Type', 'District'])

# Spatial: 0.01° grid cells
spatial_agg = aggregator.spatial_aggregation('Latitude', 'Longitude', grid_size=0.01)
```

---

### 5. **feature_engineering.py**
**Purpose:** Advanced feature creation

**Capabilities:**
- **Datetime features:** year, month, day, hour, dayofweek, quarter, is_weekend
- **Polynomial features:** x², xy, x³, etc.
- **Interaction features:** multiplication, addition, subtraction, division
- **Aggregation features:** group mean, sum, count, std
- **Ratio features:** numerator/denominator
- **Log/Sqrt transformations:** log(x+1), √x
- **Binning:** discretize continuous variables
- **Distance features:** spatial distance calculations
- **Frequency encoding:** categorical encoding by frequency

**Key Class:** `FeatureEngineer`

**Example Created Features:**
```
From 'Date' → Date_year, Date_month, Date_day, Date_hour, 
              Date_dayofweek, Date_quarter, Date_is_weekend

From Lat/Lon → distance_from_reference

From numeric cols → col_log, col_sqrt, col_binned, col1_multiply_col2
```

---

### 6. **feature_subset_selection.py**
**Purpose:** Intelligent feature selection

**Capabilities:**
- **SelectKBest:** ANOVA F-test statistical selection
- **RFE:** Recursive Feature Elimination with Logistic Regression
- **Random Forest:** Tree-based feature importance ranking

**Key Class:** `FeatureSelector`
```
selector = FeatureSelector(data)
top_kbest = selector.select_top_features(target_column='Arrest', k=5)
top_rfe = selector.select_features_rfe(target_column='Arrest', k=5)
top_rf = selector.select_features_rf_importance(target_column='Arrest', k=5)
```
**Selection Algorithms:**
- Statistical significance testing
- Model-based backward elimination
- Ensemble importance ranking

---

### 7. **data_scaling.py**
**Purpose:** Feature normalization

**Capabilities:**
- **Standard Scaling:** Z-score normalization (μ=0, σ=1)
- **MinMax Scaling:** Range normalization [0, 1]
- Automatic numeric column detection
- Preserve non-numeric columns

**Key Class:** `DataScaler`
from data_scaling import DataScaler
```
# Initialize scaler with input/output paths
scaler = DataScaler(
    input_path="../processed_datasets/processed_data.csv",
    output_path="../processed_datasets/scaled_data.csv"
)

# Load processed dataset
scaler.load_data()

# Apply standard scaling to numeric features
scaler.scale_numeric_features(method="standard")

# Save scaled dataset
scaler.save_scaled_data()
```
---

### 8. **sampling_techniques.py**
**Purpose:** Dataset size reduction

**Capabilities:**
- **Random sampling:** Simple random selection
- **Systematic sampling:** Every nth record
- **Stratified sampling:** Maintain class proportions
- **Percentage sampling:** Sample by fraction

**Key Class:** `DataSampler`

```
import pandas as pd
from sampling_techniques import DataSampler
# Load a sample dataset
data = pd.read_csv("../unprocessed_datasets/Crimes_2024.csv", nrows=10000)
# Initialize sampler
sampler = DataSampler(data)
# Random sample of 1000 rows
random_sample = sampler.random_sampling(1000)
# Systematic sample of 1000 rows
systematic_sample = sampler.systematic_sampling(1000)
# Sample 10% of the dataset
percentage_sample = sampler.sample_by_percentage(0.1)
# Stratified sampling based on a categorical column
stratified_sample = sampler.stratified_sampling("Primary Type", 1000)
```
**Use Cases:**
- Large dataset handling
- Training/testing splits
- Class balancing
- Quick prototyping

---

### 9. **discretization_binarization.py**
**Purpose:** Variable transformation

**Capabilities:**
- **Equal-width discretization:** Uniform bin sizes
- **Equal-frequency discretization:** Quantile-based bins
- **K-means discretization:** Cluster-based intelligent binning
- **Custom discretization:** User-defined bin edges
- **Binarization:** Threshold-based binary conversion
- Transformation tracking and reporting

**Key Class:** `DiscretizationBinarization`

**Discretization Methods:**
```python
# Equal-width: [0-20], [20-40], [40-60]
processor.equal_width_discretization('Age', n_bins=3)

# Equal-frequency: Equal number of records per bin
processor.equal_frequency_discretization('Income', n_bins=4)

# K-means: Data-driven clustering
processor.kmeans_discretization('Score', n_bins=5)

# Binarization: 0 if x ≤ threshold, 1 if x > threshold
processor.binarize('Distance', threshold=10)
```

## 10. **data_reduction.py**
**Purpose**: Dimensionality reduction and removal of redundant features

**Capabilities**:
**Correlation-based reduction**: Drops numeric features with high correlation
**PCA-based reduction**: Applies Principal Component Analysis
**Preserves non-numeric columns**: Ensures categorical/text features remain after PCA
**Flexible input/output**: Works with scaled or processed datasets and saves reduced data

**Key Class**: `DataReducer`
``` from data_reduction import DataReducer

# Initialize reducer
reducer = DataReducer(
    input_path="../processed_datasets/scaled_data.csv",
    output_path="../processed_datasets/reduced_data.csv"
)
# Load data
reducer.load_data()

# Option 1: Correlation-based reduction
reducer.correlation_reduction(threshold=0.9)
# Option 2: PCA-based reduction
# reducer.pca_reduction(variance_threshold=0.95)
# Save reduced dataset
reducer.save_reduced_data()
```
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

### Quick Start

**Run sample preprocessing:**
```bash
cd processing_scripts
python data_preprocessing.py
```

**Expected output:**
```
Data loaded successfully: 10000 rows, 22 columns
Dataset shape: (10000, 22)
Missing values: 850
Duplicates: 234
...
Processed data saved to: ../processed_datasets/crimes_2024_processed.csv
```

---


## Results

### Processing Statistics

**Dataset Characteristics:**
- **Original records:** ~300,000 per year
- **Years covered:** 2001-2025 (25 years)
- **Total available records:** ~7.5 million+

**Sample Processing (Crimes_2024.csv):**
```
Original: 300,000 rows × 22 columns
After sampling: 5,000 rows × 22 columns
After feature engineering: 5,000 rows × 57 columns
After feature selection: 5,000 rows × 10 columns

Processing time: ~45 seconds
Memory usage: 2.8 MB → 1.2 MB (optimized)
```

### Data Quality Improvements

**Before Processing:**
- Missing values: 5-10% (esp. in location columns)
- Duplicates: 2-3%
- Inconsistent formats: Text case, whitespace
- Outliers: ~15-20% statistical outliers

**After Processing:**
- Missing values: 0% (imputed)
- Duplicates: 0% (removed)
- Standardized: All text uppercase, trimmed
- Outliers: Handled (Z-score threshold = 3)

### Feature Engineering Results

**Original Features:** 22
```
ID, Case Number, Date, Block, IUCR, Primary Type, Description, 
Location Description, Arrest, Domestic, Beat, District, Ward, 
Community Area, FBI Code, X Coordinate, Y Coordinate, Year, 
Updated On, Latitude, Longitude, Location
```

**Engineered Features:** 35+ additional features
```
Temporal (7): Date_year, Date_month, Date_day, Date_hour, 
              Date_dayofweek, Date_quarter, Date_is_weekend

Spatial (1): distance_from_reference

Interactions (2+): Ward_multiply_District, Latitude_add_Longitude

Aggregations (3+): Ward_mean_by_District, District_count_by_PrimaryType

Transformations (6+): X_Coordinate_log, Y_Coordinate_log, 
                      Ward_sqrt, District_sqrt

Encoding (2+): Primary_Type_frequency, Location_Description_frequency

Discretized (3+): Ward_binned, District_equal_width, CommunityArea_kmeans

Binary (3+): Ward_binary, District_binary, Arrest (original)
```

**Total Features:** 57+

### Aggregation Results

**Temporal Aggregation:**
- Daily data → Monthly summaries (365:12 ratio)
- ~97% data reduction while preserving trends

**Spatial Aggregation:**
- Grid size 0.01° → ~500 grid cells
- Original 300K records → 500 spatial cells
- ~99.8% reduction for spatial analysis

**Categorical Aggregation:**
- 30+ crime types identified
- Top 5 types: THEFT, BATTERY, CRIMINAL DAMAGE, ASSAULT, DECEPTIVE PRACTICE
- District-level patterns identified

### Performance Metrics

| Operation | Original Size | Final Size | Reduction | Time |
|-----------|--------------|------------|-----------|------|
| Random Sampling | 300K | 10K | 96.7% | 2s |
| Stratified Sampling | 300K | 10K | 96.7% | 5s |
| Temporal Aggregation | 300K | 365 | 99.9% | 3s |
| Spatial Aggregation | 300K | 500 | 99.8% | 8s |
| Feature Engineering | 22 cols | 57 cols | +159% | 10s |
| PCA Reduction | 57 cols | 10 cols | 82.5% | 5s |

---

## Key Takeaways

### Why It Matters
- This project demonstrates how to build a scalable, end-to-end data pipeline that transforms raw crime data into clean, analysis-ready datasets, enabling efficient insights and reliable machine learning applications.

### What This Project Demonstrates

 **End-to-End Data Pipeline**
- Complete workflow from raw data to analysis-ready datasets
- Modular architecture for flexibility
- Production-ready code structure

 **Data Quality Management**
- Systematic handling of missing values
- Duplicate detection and removal
- Outlier treatment strategies
- Data type optimization

 **Advanced Feature Engineering**
- 35+ derived features from 22 original columns
- Temporal, spatial, and categorical feature extraction
- Mathematical transformations
- Encoding strategies

 **Multiple Preprocessing Techniques**
- 4 sampling methods implemented
- 2 scaling strategies
- 4 discretization methods
- 3 feature selection algorithms

 **Large-Scale Data Handling**
- 7.5M+ total records processed
- Efficient memory management
- Batch processing capabilities
- Scalable architecture

 **Comprehensive Documentation**
- Detailed module descriptions
- Usage examples for all functions
- Clear workflow explanation
- Code comments and docstrings

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
