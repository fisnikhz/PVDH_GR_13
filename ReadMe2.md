
-----

<table>
  <tr>
    <td width="150" align="center" valign="middle">
      <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/University_of_Prishtina_logo.svg/1200px-University_of_Prishtina_logo.svg.png" width="120" alt="University Logo" />
    </td>
    <td align="left" valign="top">
      <h3>Universiteti i Prishtinës</h3>
      <p>
        <strong>Fakulteti i Inxhinierisë Elektrike dhe Kompjuterike</strong><br>
        Inxhinieri Kompjuterike dhe Softuerike - Programi Master
      </p>
      <br>
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

-----

## Table of Contents

  - [Project Overview](https://www.google.com/search?q=%23-project-overview)
  - [Repository Structure](https://www.google.com/search?q=%23-repository-structure)
  - [Dataset Description](https://www.google.com/search?q=%23-dataset-description)
  - [Data Processing Workflow](https://www.google.com/search?q=%23-data-processing-workflow)
  - [Implemented Class Logic](https://www.google.com/search?q=%23-implemented-class-logic)
  - [Technologies Used](https://www.google.com/search?q=%23-technologies-used)
  - [Installation & Setup](https://www.google.com/search?q=%23-installation--setup)
  - [Results](https://www.google.com/search?q=%23-results)

-----

## Project Overview

This repository contains a robust, **Object-Oriented Data Preprocessing Pipeline** designed for the **Chicago Crime Dataset** (2001-present). 
The project implements a single, unified `DataPreprocessor` class that handles the entire lifecycle of data preparation—from raw CSV integration to advanced feature engineering and dimensionality reduction.

### Project Goals

  - **Automated Pipeline:** Transform raw, multi-file datasets into analysis-ready formats with a single execution.
  - **Advanced Engineering:** Derive spatial (distance from city center) and temporal (hour, day of week) features.
  - **Dimensionality Reduction:** Apply PCA and Feature Selection to reduce noise while retaining variance.
  - **Scalability:** Efficiently handle large datasets via random sampling mechanisms.

-----

## Repository Structure

The project is structured around a central preprocessing class to ensure modularity and ease of maintenance.

```
PVDH_GR_13/
│
├── unprocessed_datasets/                    # Raw crime data (CSV files)
│   ├── Crimes_2001.csv
│   ├── ...
│   └── Crimes_2025.csv
│
├── processed_datasets/                      # Output directory
│   └── CrimesChicagoDatasetPreprocessed.csv
│
├── processing_scripts/              
│   ├── data_preprocessing_pipeline.py       # Main class containing all logic
│   └── main.py                              # Execution entry point
│
├── LICENSE                                  # MIT License
├── README.md                                # Project documentation
└── .gitignore                               # Git ignore rules
```

-----
### Directory Breakdown

**`unprocessed_datasets/`**
- Contains 25 CSV files (Crimes_2001.csv through Crimes_2025.csv)
- Raw data from Chicago crime database
- Each file contains 100K-400K crime records
- Total dataset size: ~7.5 million+ crime records

**`processed_datasets/`**
- Output directory for cleaned and processed data
- Contains transformed datasets after applying various techniques
- Example: `crimes_2024_processed.csv` (preprocessed 2024 data)

**`processing_scripts/`**
- 9 Python modules implementing different data processing techniques
- Each module is self-contained and reusable
- Can be used independently or as part of a pipeline
- Includes example usage in `main()` functions


## Dataset Description

### Source

**Chicago Crime Data** - Official crime reports from the City of Chicago Police Department (Citizen Law Enforcement Analysis and Reporting system).

### Attributes (Selected)


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


-----

## Data Processing Workflow

The pipeline is designed to execute steps in a strict dependency order to prevent data leakage and ensure logical consistency.

### 1\. Integration & Sampling

  * **Integration:** Iteratively loads all `*.csv` files from the input directory and concatenates them into a single DataFrame.
  * **Sampling:** Offers a choice between processing the full dataset (millions of rows) or a random sample (default: 5,000 rows) for rapid prototyping.

### 2\. Quality Assessment & Cleaning

  * **Assessment:** Generates a report on missing values, memory usage, and duplicate rows.
  * **Cleaning:** \* Removes duplicate records.
      * Standardizes text columns (stripping whitespace, upper-casing).
      * Parses `Date` strings into Datetime objects.
      * Fills missing values in categorical columns (e.g., Ward, Community Area) with "UNKNOWN".

### 3\. Feature Engineering

  * **Temporal Extraction:** Extracts `Hour`, `Day`, `Month`, `Year`, and `DayOfWeek`.
  * **Spatial Engineering:** Calculates `DistanceFromCenter` (Euclidean distance from Chicago city center coordinates: 41.8781, -87.6298).
  * **Binary Flags:** Creates `IsViolent` flag based on specific crime types (HOMICIDE, ASSAULT, BATTERY, ROBBERY).

### 4\. Aggregation

  * **Granularity Adjustment:** Aggregates crime counts by `YearMonth` and `PrimaryType` to create statistical density features (`MonthlyCrimeCount`, `TypeMonthlyCount`).

### 5\. Encoding & Normalization

  * **Categorical Encoding:** Applies `LabelEncoder` to categorical columns with \<50 unique values.
  * **Normalization:** Applies `MinMaxScaler` to numeric columns to scale values between [0, 1].

### 6\. Transformation (Discretization & PCA)

  * **Discretization:** Transforms continuous variables into discrete bins using `KBinsDiscretizer` (Quantile strategy).
  * **Binarization:** Converts numeric metrics into binary flags (0/1) based on median thresholds.
  * **PCA:** Applies Principal Component Analysis (3 components) to dense numeric features to capture latent variance.

### 7\. Feature Selection

  * **Algorithm:** Uses `SelectKBest` with ANOVA F-value (`f_classif`) to select the top 20 most relevant features relative to the target variable (`Arrest`).

-----

## Implemented Class Logic

The entire logic is encapsulated in the `DataPreprocessor` class.

### Key Methods

| Method | Functionality |
| :--- | :--- |
| `integrate_unprocessed_csvs` | Merges fragmented CSVs into one dataframe. |
| `clean_data` | Handles duplicates, missing values, and type casting. |
| `create_features` | Generates temporal and spatial derived features. |
| `aggregate_monthly_and_type_counts` | Adds high-level statistical count features. |
| `normalize_numeric_minmax` | Scales data for machine learning compatibility. |
| `encode_categoricals` | Converts text labels to numeric IDs. |
| `discretize_numeric` | Bins continuous data (e.g., categorizing distance). |
| `apply_pca` | Reduces dimensionality for visualization/modeling. |
| `select_feature_subset` | Retains only the most statistically significant features. |

-----

## Technologies Used

  * **Python 3.x**
  * **Pandas:** For high-performance dataframe manipulation and aggregation.
  * **NumPy:** For vector calculations (Geo-distance).
  * **Scikit-Learn:**
      * `MinMaxScaler`, `StandardScaler`
      * `LabelEncoder`, `KBinsDiscretizer`, `Binarizer`
      * `PCA` (Decomposition)
      * `SelectKBest`, `f_classif` (Feature Selection)

-----

## Installation & Setup

**1. Clone the repository:**

```bash
git clone https://github.com/fisnikhz/PVDH_GR_13.git
cd PVDH_GR_13
```

**2. Install dependencies:**

```bash
pip install pandas numpy scikit-learn
```

**3. Run the pipeline:**

```bash
cd processing_scripts
python main.py
```

**4. Interactive Prompt:**
The script will ask if you want to run the full dataset or a sample:

```
Process full dataset or sample 5000 rows? [full/sample] (default sample): sample
```

-----

## Results

Upon execution, the script generates a comprehensive report on the transformations applied.

**Example Output:**

```text
============================================================
CHICAGO CRIMES DATA PREPROCESSING REPORT
============================================================
Processed data: 5,000 rows, 20 columns
Memory usage: 0.78 MB

Top columns with missing values:
No missing values found.

Numeric column summary (first 10 columns):
... [Statistical Summary] ...

PCA applied: 3 components, explained variance 45.2%
Selected features: ['Arrest', 'Domestic', 'DistanceFromCenter', 'IsViolent', 'Hour', ...]
============================================================
Processed data saved to: ../processed_datasets/CrimesChicagoDatasetPreprocessed.csv
```

-----

## License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

-----


##  Acknowledgments

- **City of Chicago** - For providing open access to crime data
- **PhD Mërgim Hoti** - For course instruction and guidance
- **University of Prishtina** - For academic support and resources
- **Python Community** - For excellent open-source libraries
- **scikit-learn Team** - For comprehensive ML preprocessing tools
- **pandas Development Team** - For powerful data manipulation capabilities

---
