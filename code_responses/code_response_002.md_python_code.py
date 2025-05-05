import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('25_medical_appointments.csv')

# 1. Convert date columns to datetime format
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

# 2. Create a feature for days between scheduling and appointment
df['Days_Diff'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days

# 3. Fix column names for consistency
df = df.rename(columns={
    'Hipertension': 'Hypertension',
    'Handcap': 'Handicap',
    'No-show': 'NoShow'
})

# 4. Convert No-show to boolean
df['NoShow'] = df['NoShow'].map({'Yes': True, 'No': False})

# 5. Check for age outliers (optional removal)
print(f"Age range: {df['Age'].min()} to {df['Age'].max()}")
# Filter out unreasonable ages if needed (e.g., negative or extremely high)
df = df[df['Age'] >= 0]

# 6. Convert binary health condition columns to consistent boolean
for col in ['Scholarship', 'Hypertension', 'Diabetes', 'Alcoholism', 'Handicap', 'SMS_received']:
    df[col] = df[col].astype(bool)

# 7. Check for duplicate appointments
duplicates = df.duplicated(subset=['PatientId', 'AppointmentDay'], keep=False)
print(f"Number of duplicate appointments: {duplicates.sum()}")

# 8. Final data info
print(df.info())
print(df.describe())

# Save cleaned data
df.to_csv('cleaned_appointments.csv', index=False)