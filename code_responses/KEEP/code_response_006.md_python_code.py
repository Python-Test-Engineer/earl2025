import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('25_medical_appointments.csv')

# Basic info and checking for missing values
print("Initial data shape:", df.shape)
print("Missing values:")
print(df.isna().sum())

# Rename columns for consistency
df = df.rename(columns={
    'No-show': 'no_show',
    'Handcap': 'handicap'
})

# Convert date columns to datetime
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

# Create a new feature: days between scheduling and appointment
df['days_difference'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.total_seconds() / (60*60*24)

# Convert binary columns to proper boolean type
binary_columns = ['Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'handicap', 'SMS_received']
for col in binary_columns:
    df[col] = df[col].astype(bool)

# Convert no_show to boolean (Yes = missed appointment / No = attended)
df['no_show'] = df['no_show'].map({'Yes': True, 'No': False})

# Check for and handle potential age outliers
print("Age statistics:")
print(df['Age'].describe())

# Remove any negative ages or extremely high ages (e.g., > 120)
df = df[(df['Age'] >= 0) & (df['Age'] <= 120)]

# Check for duplicates
duplicate_count = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_count}")
if duplicate_count > 0:
    df = df.drop_duplicates()

# Final cleaned data
print("Final data shape:", df.shape)
print(df.head())

# Save cleaned data
df.to_csv('cleaned_appointment_data.csv', index=False)