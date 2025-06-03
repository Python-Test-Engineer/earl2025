import pandas as pd
import numpy as np
from random import randint

# Load the dataset
df = pd.read_csv('25_medical_appointments.csv')

# 1. Examine the data structure
print("Original data info:")
print(df.info())
print("\nSample data:")
print(df.head())

# 2. Fix the PatientId - convert to string to prevent numerical issues with large numbers
df['PatientId'] = df['PatientId'].astype(str)

# 3. Convert date columns to datetime
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

# 4. Fix column names
df = df.rename(columns={'No-show': 'NoShow', 'Handcap': 'Handicap'})

# 5. Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# 6. Convert binary columns to boolean type (0/1)
binary_columns = ['Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handicap', 'SMS_received']
for col in binary_columns:
    df[col] = df[col].astype(bool)

# 7. Convert NoShow to boolean (Yes/No to True/False)
df['NoShow'] = df['NoShow'] == 'Yes'

# 8. Create basic stats for numerical features
print("\nAge statistics:")
print(df['Age'].describe())

# 9. Check for any invalid values in Age column
invalid_ages = df[df['Age'] < 0]
if not invalid_ages.empty:
    print(f"Found {len(invalid_ages)} records with invalid ages")
    df = df[df['Age'] >= 0]

# 10. Generate a random number for the filename
file_num = randint(1000, 9999)
output_file = f'25_medical_appointments_CLEANED_{file_num}.csv'

# 11. Save the cleaned dataset
df.to_csv(output_file, index=False)
print(f"\nCleaned data saved to {output_file}")