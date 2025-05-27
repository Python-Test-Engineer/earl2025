import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('25_medical_appointments.csv')

# 1. Fix column names
df = df.rename(columns={
    'Handcap': 'Handicap',
    'No-show': 'NoShow'
})

# 2. Convert datetime columns
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

# 3. Create a feature for waiting days
df['WaitingDays'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days

# 4. Convert binary columns to int (0/1)
binary_cols = ['Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'SMS_received']
for col in binary_cols:
    df[col] = df[col].astype(int)

# 5. Convert NoShow to binary (0 = showed up, 1 = no-show)
df['NoShow'] = df['NoShow'].map({'No': 0, 'Yes': 1})

# 6. Check for invalid age values
# Filter out patients with unreasonable ages (e.g., negative or very high)
df = df[(df['Age'] >= 0) & (df['Age'] <= 115)]

# 7. Standardize neighborhood names (capitalize)
df['Neighbourhood'] = df['Neighbourhood'].str.title()

# 8. Handle any missing values
df = df.dropna()

# Display cleaned data information
print(f"Cleaned data shape: {df.shape}")
print(df.dtypes)
print(df.head())