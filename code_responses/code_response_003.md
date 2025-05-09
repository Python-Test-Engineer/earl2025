# Data Cleaning Analysis for Medical Appointments Dataset

After examining the CSV data, here are the necessary cleaning actions and the Python code to implement them:

## Issues to Address:
1. Convert date columns to datetime format
2. Fix column names (e.g., "Handcap" should be "Handicap")
3. Rename "No-show" to a more code-friendly format
4. Check for missing values
5. Check for inconsistent data types (esp. in Age, Hipertension, etc.)
6. Ensure all categorical variables are properly encoded
7. Check for outliers in numerical columns

## Python Code for Data Cleaning:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("DATA_FILE")

# Initial exploration
print("Original data shape:", df.shape)
print("\nColumn names:", df.columns.tolist())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())

# 1. Fix column names
df = df.rename(columns={
    'Handcap': 'Handicap',
    'Hipertension': 'Hypertension',
    'No-show': 'NoShow'
})

# 2. Convert date columns to datetime
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

# 3. Create wait time column (days between scheduling and appointment)
df['WaitDays'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days

# 4. Convert binary text to integers in NoShow column
df['NoShow'] = df['NoShow'].map({'No': 0, 'Yes': 1})

# 5. Check for outliers in Age
print("\nAge statistics:")
print(df['Age'].describe())
# Remove any negative ages or unreasonably high ages
df = df[df['Age'] >= 0]
df = df[df['Age'] < 120]  # Assuming max human age around 120

# 6. Check for consistency in binary columns
binary_cols = ['Scholarship', 'Hypertension', 'Diabetes', 'Alcoholism', 'SMS_received']
for col in binary_cols:
    unique_vals = df[col].unique()
    print(f"\nUnique values in {col}: {unique_vals}")
    # Ensure only 0 and 1 values exist
    if not all(val in [0, 1] for val in unique_vals):
        print(f"Warning: Non-binary values in {col}")

# 7. Check Handicap values
print("\nHandicap value counts:")
print(df['Handicap'].value_counts())
# If Handicap has values other than 0/1, consider converting to binary
# (0 for no handicap, 1 for any level of handicap)
if df['Handicap'].max() > 1:
    df['Handicap'] = df['Handicap'].apply(lambda x: 1 if x > 0 else 0)

# Final cleaned dataset summary
print("\nCleaned data shape:", df.shape)
print("\nCleaned data types:")
print(df.dtypes)
print("\nSample of cleaned data:")
print(df.head())

# Save cleaned dataset
df.to_csv("cleaned_appointments.csv", index=False)
```

This code handles the main cleaning tasks while keeping the implementation simple. It:
1. Renames problematic columns
2. Converts date columns to proper datetime format
3. Creates a useful wait time feature
4. Converts the target variable to numeric format
5. Checks for and handles age outliers
6. Ensures binary columns contain only proper values
7. Standardizes the Handicap column

You can further customize this based on additional insights gained during the cleaning process.