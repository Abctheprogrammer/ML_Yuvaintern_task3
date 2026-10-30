"""
Week 1: Python for Machine Learning & Data Preprocessing
Run: python preprocessing_project.py

This self-contained beginner project demonstrates:
- Loading a sample CSV dataset
- Inspecting data and performing exploratory data analysis (EDA)
- Removing duplicate records
- Handling missing numeric and categorical values
- Encoding categorical variables
- Selecting useful features
- Standardizing numeric features
- Saving cleaned and transformed outputs
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = Path(__file__).resolve().parent
INPUT_FILE = OUTPUT_DIR / "sample_student_performance.csv"

def create_sample_dataset():
    """Create a small sample dataset so the project runs without external files."""
    data = {
        "Student_ID": [101, 102, 103, 104, 105, 106, 107, 108, 108, 109],
        "Study_Hours": [2.0, 4.5, np.nan, 6.0, 3.0, 7.5, 5.0, 4.0, 4.0, 8.0],
        "Attendance_Percent": [65, 78, 72, np.nan, 80, 95, 88, 76, 76, 98],
        "Internet_Access": ["No", "Yes", "Yes", "Yes", np.nan, "Yes", "No", "Yes", "Yes", "Yes"],
        "Parental_Support": ["Low", "Medium", "High", "Medium", "Low", "High", "Medium", "Low", "Low", "High"],
        "Final_Score": [48, 67, 61, 75, np.nan, 92, 79, 64, 64, 96],
    }
    df = pd.DataFrame(data)
    df.to_csv(INPUT_FILE, index=False)
    return df

def main():
    print("=" * 60)
    print("WEEK 1: PYTHON FOR MACHINE LEARNING & DATA PREPROCESSING")
    print("=" * 60)

    # 1. Load data
    if INPUT_FILE.exists():
        df = pd.read_csv(INPUT_FILE)
        print(f"\nLoaded dataset: {INPUT_FILE.name}")
    else:
        df = create_sample_dataset()
        print(f"\nCreated sample dataset: {INPUT_FILE.name}")

    # If the sample file already exists, use it. If not, create it.
    print("\n1. ORIGINAL DATA PREVIEW")
    print(df.head().to_string(index=False))
    print(f"\nOriginal shape: {df.shape}")
    print("\nData types:\n", df.dtypes)
    print("\nMissing values by column:\n", df.isnull().sum())
    print(f"\nDuplicate rows: {df.duplicated().sum()}")

    # 2. EDA: numeric summary and correlations
    numeric_cols = ["Study_Hours", "Attendance_Percent", "Final_Score"]
    print("\n2. EXPLORATORY DATA ANALYSIS (EDA)")
    print(df[numeric_cols].describe().round(2))
    print("\nNumeric correlation matrix:\n", df[numeric_cols].corr().round(2))

    # 3. Remove duplicate rows
    cleaned = df.drop_duplicates().copy()
    rows_removed = len(df) - len(cleaned)

    # 4. Handle missing values: median for numeric, mode for categorical
    for col in numeric_cols:
        cleaned[col] = cleaned[col].fillna(cleaned[col].median())
    categorical_cols = ["Internet_Access", "Parental_Support"]
    for col in categorical_cols:
        cleaned[col] = cleaned[col].fillna(cleaned[col].mode()[0])

    # 5. Feature selection: ID is an identifier, not a useful predictive feature
    selected_features = [
        "Study_Hours", "Attendance_Percent", "Internet_Access",
        "Parental_Support", "Final_Score"
    ]
    selected = cleaned[selected_features].copy()

    # 6. Encode categorical variables using one-hot encoding
    encoded = pd.get_dummies(
        selected,
        columns=["Internet_Access", "Parental_Support"],
        drop_first=False,
        dtype=int
    )

    # 7. Normalize/standardize continuous variables
    # StandardScaler gives each numeric feature mean ~0 and standard deviation ~1.
    scale_cols = ["Study_Hours", "Attendance_Percent", "Final_Score"]
    scaler = StandardScaler()
    encoded[scale_cols] = scaler.fit_transform(encoded[scale_cols])

    # 8. Save outputs
    cleaned.to_csv(OUTPUT_DIR / "cleaned_dataset.csv", index=False)
    encoded.to_csv(OUTPUT_DIR / "preprocessed_dataset.csv", index=False)

    summary = [
        "WEEK 1 PREPROCESSING SUMMARY",
        f"Original rows: {len(df)}",
        f"Rows removed as duplicates: {rows_removed}",
        f"Rows after duplicate removal: {len(cleaned)}",
        f"Missing values remaining in cleaned data: {int(cleaned.isnull().sum().sum())}",
        f"Selected features: {', '.join(selected_features)}",
        "Numeric missing values: filled with median.",
        "Categorical missing values: filled with mode.",
        "Categorical encoding: one-hot encoding.",
        "Numeric scaling: StandardScaler (mean 0, standard deviation 1).",
        "Files created: cleaned_dataset.csv, preprocessed_dataset.csv",
    ]
    (OUTPUT_DIR / "preprocessing_summary.txt").write_text("\\n".join(summary), encoding="utf-8")

    print("\n3. PREPROCESSING RESULTS")
    print(f"Duplicate rows removed: {rows_removed}")
    print("Missing values remaining:", int(cleaned.isnull().sum().sum()))
    print("\nCleaned dataset preview:\n", cleaned.head().to_string(index=False))
    print("\nPreprocessed dataset preview:\n", encoded.head().round(3).to_string(index=False))
    print("\nFiles saved:")
    print("- cleaned_dataset.csv")
    print("- preprocessed_dataset.csv")
    print("- preprocessing_summary.txt")
    print("\nTask completed successfully.")

if __name__ == "__main__":
    main()
