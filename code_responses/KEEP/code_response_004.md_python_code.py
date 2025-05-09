import pandas as pd
import numpy as np
from datetime import datetime

# Load the data
df = pd.read_csv("25_medical_appointments.csv")

# 1. Check for duplicates
print("Before removing duplicates:", df.shape)
df = df.drop_duplicates()
print("After removing duplicates:", df.shape)

# 2. Fix date columns - convert to datetime and remove timezone info
for col in ['ScheduledDay', 'AppointmentDay']:
    df[col] = pd.to_datetime(df[col]).dt.date

# 3. Convert binary fields to boolean for clarity
binary_cols = ['Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received']
for col in binary_cols:
    df[col] = df[col].astype(bool)

# 4. Convert 'No-show' to boolean (Yes = True, No = False)
df['No-show'] = df['No-show'].map({'Yes': True, 'No': False})

# 5. Check for missing values
missing_values = df.isnull().sum()
print("Missing values per column:\n", missing_values)

# 6. Analyze age distribution to identify potential outliers
print("Age statistics:")
print(df['Age'].describe())

# 7. Rename columns for consistency (optional)
df = df.rename(columns={
    'Hipertension': 'Hypertension',
    'Handcap': 'Handicap',
    'No-show': 'NoShow'
})

# 8. Save the cleaned data
df.to_csv("cleaned_25_medical_appointments.csv", index=False)

# Display sample of cleaned data
print(df.head())