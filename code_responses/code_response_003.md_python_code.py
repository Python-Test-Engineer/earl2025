import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('25_medical_appointments.csv')

# 1. Check for missing values and drop if necessary
print("Missing values before cleaning:")
print(df.isnull().sum())
df = df.dropna()

# 2. Convert date columns to datetime format
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

# 3. Fix column name 'Handcap' to 'Handicap'
df = df.rename(columns={'Handcap': 'Handicap'})

# 4. Convert binary columns to proper boolean type
binary_columns = ['Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handicap', 'SMS_received']
for col in binary_columns:
    df[col] = df[col].astype(bool)

# 5. Rename 'No-show' to 'no_show' and convert to boolean
df = df.rename(columns={'No-show': 'no_show'})
df['no_show'] = df['no_show'].map({'No': False, 'Yes': True})

# 6. Ensure Age values are reasonable
print(f"Age range before cleaning: {df['Age'].min()} to {df['Age'].max()}")
df = df[df['Age'] >= 0]  # Remove negative ages if any

# 7. Check for duplicate appointments
duplicates = df.duplicated(subset=['PatientId', 'AppointmentDay']).sum()
print(f"Number of duplicate appointments: {duplicates}")

# Display cleaned data info
print("\nDataset after cleaning:")
print(df.info())
print(df.head())

# Save the cleaned data
df.to_csv('cleaned_appointment_data.csv', index=False)