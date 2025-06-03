import pandas as pd
import numpy as np
from datetime import datetime

# Read the CSV file
df = pd.read_csv('25_medical_appointments.csv')

# 1. Check for missing values
print(f"Missing values before cleaning:\n{df.isnull().sum()}")

# 2. Convert date columns to datetime format
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

# 3. Fix column names
df = df.rename(columns={
    'Handcap': 'Handicap',
    'No-show': 'NoShow',
    'Hipertension': 'Hypertension'
})

# 4. Standardize boolean columns (convert to 0 and 1)
boolean_columns = ['Scholarship', 'Hypertension', 'Diabetes', 'Alcoholism', 'SMS_received', 'Handicap']
for col in boolean_columns:
    df[col] = df[col].astype(int)

# 5. Standardize NoShow column to boolean
df['NoShow'] = df['NoShow'].map({'No': 0, 'Yes': 1})

# 6. Check and filter out invalid ages (negative or unreasonably high)
print(f"Age range before cleaning: {df['Age'].min()} to {df['Age'].max()}")
df = df[(df['Age'] >= 0) & (df['Age'] < 120)]

# 7. Check for duplicates
duplicates = df.duplicated().sum()
print(f"Number of duplicates: {duplicates}")
if duplicates > 0:
    df = df.drop_duplicates()

# 8. Calculate appointment wait time (days between scheduled and appointment)
df['WaitTime'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days

# 9. Final check for any remaining missing values
print(f"Missing values after cleaning:\n{df.isnull().sum()}")

# Save the cleaned dataset
df.to_csv('25_medical_appointments_CLEANED.csv', index=False)

print("Data cleaning completed successfully!")