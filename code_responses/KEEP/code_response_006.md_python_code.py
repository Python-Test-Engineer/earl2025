import pandas as pd
import numpy as np
from datetime import datetime

# Load the data
df = pd.read_csv('25_medical_appointments.csv')

# 1. Convert date columns to datetime
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

# 2. Create a feature for days between scheduling and appointment
df['DaysBetween'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days

# 3. Check and handle age outliers
print(f"Age statistics: Min={df['Age'].min()}, Max={df['Age'].max()}, Mean={df['Age'].mean():.2f}")
# Optionally filter out unreasonable ages
# df = df[(df['Age'] >= 0) & (df['Age'] <= 120)]

# 4. Convert binary columns to proper boolean
binary_cols = ['Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received']
for col in binary_cols:
    df[col] = df[col].astype(bool)

# 5. Convert No-show to boolean
df['No-show'] = df['No-show'].map({'Yes': True, 'No': False})

# 6. Check for duplicate appointments
duplicates = df.duplicated(subset=['PatientId', 'AppointmentDay'], keep=False)
print(f"Number of duplicate appointments: {duplicates.sum()}")

# 7. Check for missing values
missing_values = df.isnull().sum()
print("Missing values by column:")
print(missing_values[missing_values > 0])

# 8. Create a cleaned dataset
df_clean = df.copy()

# Display the first few rows of cleaned data
print(df_clean.head())

# Save the cleaned dataset
df_clean.to_csv('cleaned_appointments.csv', index=False)