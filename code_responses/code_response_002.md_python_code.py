import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('25_medical_appointments.csv')

# 1. Basic Data Inspection
print(f"Shape of data: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Missing values: {df.isnull().sum().sum()}")

# 2. Fix column names
df = df.rename(columns={
    'No-show': 'NoShow',  # Remove hyphen for better accessibility
    'Handcap': 'Handicap'  # Fix typo
})

# 3. Convert date columns to datetime
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

# 4. Create feature for days between scheduling and appointment
df['DaysBeforeAppointment'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days

# 5. Check for unreasonable values in Age
print(f"Age range: {df['Age'].min()} to {df['Age'].max()}")
# Filter out any records with negative or unreasonably high ages
df = df[(df['Age'] >= 0) & (df['Age'] <= 120)]

# 6. Convert binary features to consistent format
binary_columns = ['Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'SMS_received']
for col in binary_columns:
    df[col] = df[col].astype(int)

# 7. Convert target variable to binary (1 for no-show, 0 for showed up)
df['NoShow'] = df['NoShow'].map({'Yes': 1, 'No': 0})

# 8. Check for duplicates
duplicate_count = df.duplicated().sum()
print(f"Number of duplicates: {duplicate_count}")
if duplicate_count > 0:
    df = df.drop_duplicates()

# 9. Create a final clean dataset
df_clean = df.copy()

# 10. Display the cleaned dataset info
print("\nCleaned Dataset Information:")
print(df_clean.info())
print("\nSample of cleaned data:")
print(df_clean.head())

# Save the cleaned dataset
df_clean.to_csv('cleaned_appointment_data.csv', index=False)