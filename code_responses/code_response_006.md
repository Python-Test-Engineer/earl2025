# Data Cleaning Analysis for Medical Appointments Dataset

After reviewing the provided CSV content, here are the actions needed to clean the data and the Python code to implement them:

## Issues to Address:
1. Check for missing values
2. Format date columns properly
3. Rename the "No-show" column to remove the hyphen (for easier use in Python)
4. Convert categorical variables to appropriate types
5. Check and handle any inconsistencies in the "Handcap" column (looks misspelled)
6. Check for and handle outliers in the Age column
7. PatientId has unusual length/formatting

## Python Code for Data Cleaning:

```python
import pandas as pd
import numpy as np
import random
from datetime import datetime

# Load the data
df = pd.read_csv('DATA_FILE')

# 1. Check basic information
print("Initial data shape:", df.shape)
print("Missing values before cleaning:\n", df.isnull().sum())

# 2. Rename columns for consistency (fix No-show and Handcap)
df = df.rename(columns={
    'No-show': 'NoShow',
    'Handcap': 'Handicap'
})

# 3. Convert date columns to datetime format
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

# 4. Create a new feature: days between scheduling and appointment
df['DaysDiff'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days

# 5. Convert categorical variables
df['NoShow'] = df['NoShow'].map({'No': 0, 'Yes': 1})

# 6. Check for age outliers
print(f"Age range: {df['Age'].min()} to {df['Age'].max()}")

# Remove extreme age outliers if any (e.g., ages over 120 or negative)
df = df[(df['Age'] >= 0) & (df['Age'] <= 120)]

# 7. Standardize neighbourhood names (uppercase)
df['Neighbourhood'] = df['Neighbourhood'].str.upper()

# 8. Convert binary variables to proper int format
binary_cols = ['Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'SMS_received']
for col in binary_cols:
    df[col] = df[col].astype(int)

# 9. Check for any other missing values after transformations
print("Missing values after cleaning:\n", df.isnull().sum())

# 10. Save the cleaned dataset
random_num = random.randint(1000, 9999)
output_file = f'25_medical_appointments_CLEANED_{random_num}.csv'
df.to_csv(output_file, index=False)

print(f"Cleaned data saved to {output_file}")
print(f"Final data shape: {df.shape}")
```

This code addresses the main cleaning tasks needed for this dataset:
- Corrects column names for easier access
- Properly formats date columns
- Creates a useful feature (days between scheduling and appointment)
- Converts categorical variables to numeric format
- Handles potential age outliers
- Standardizes text data (neighborhood names)
- Ensures binary variables are properly formatted
- Provides summary information about the cleaning process

The code is kept simple and focuses on essential cleaning operations without unnecessary complexity.