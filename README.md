# Week 1: Python for Machine Learning & Data Preprocessing

## Project overview
This beginner-friendly project uses a sample student performance dataset to demonstrate data loading, EDA, duplicate removal, missing-value handling, feature selection, categorical encoding, and numerical standardization.

## Requirements
- Python 3.9 or newer
- pandas
- numpy
- scikit-learn

## Run the project
Open a terminal in this folder and run:

```bash
python -m pip install -r requirements.txt
python preprocessing_project.py
```

The script creates `sample_student_performance.csv` when it is not already present, then saves:
- `cleaned_dataset.csv` — cleaned data before encoding/scaling
- `preprocessed_dataset.csv` — model-ready data with one-hot encoding and standardized numeric columns
- `preprocessing_summary.txt` — summary of preprocessing steps

## Notes
The dataset is synthetic and is for learning purposes only. The `Student_ID` column is excluded from selected features because it is an identifier rather than a meaningful student-performance measurement.
