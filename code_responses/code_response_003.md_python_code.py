import pandas as pd
import numpy as np
from datetime import datetime
import random

# Load the data
df = pd.read_csv("25_medical_appointments.csv")

# 1. Check and handle missing values
print(f"Missing values before cleaning:\n{df.isnull().sum()}")
df = df.dropna()  # Remove rows with missing values

# 2. Convert date columns to proper datetime format
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

# 3. Fix column naming
df = df.rename(columns={
    'No-show': 'NoShow',
    'Handcap': 'Handicap'
})

# 4. Ensure data type consistency
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
# Remove records with invalid ages (negative or extremely high)
df = df[df['Age'].between(0, 120)]

# 5. Normalize categorical values
df['Gender'] = df['Gender'].str.upper()
df['NoShow'] = df['NoShow'].map({'No': 0, 'Yes': 1})

# 6. Check and handle binary fields - ensure they're 0 or 1
binary_columns = ['Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handicap', 'SMS_received']
for col in binary_columns:
    df[col] = df[col].astype(int)

# 7. Create a more manageable PatientId (optional)
# This creates a mapping from original IDs to new sequential IDs
unique_patients = df['PatientId'].unique()
id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_patients, 1)}
df['PatientId'] = df['PatientId'].map(id_mapping)

# 8. Create days_diff: days between scheduled and appointment
df['days_diff'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days

# Final check for any remaining issues
print(f"Missing values after cleaning:\n{df.isnull().sum()}")

# Save the cleaned dataset
output_filename = f"25_medical_appointments_CLEANED_{random.randint(1000, 9999)}.csv"
df.to_csv(output_filename, index=False)
print(f"Cleaned data saved to {output_filename}")