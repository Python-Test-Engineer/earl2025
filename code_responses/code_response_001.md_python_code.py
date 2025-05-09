import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('25_medical_appointments.csv')

# 1. Check basic information and missing values
print("Data information:")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())

# 2. Rename problematic columns
df = df.rename(columns={
    'Handcap': 'Handicap',
    'No-show': 'NoShow',
    'Hipertension': 'Hypertension'  # Fixing spelling if needed
})

# 3. Convert datetime columns to proper datetime format
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

# 4. Create a feature for days between scheduling and appointment
df['DaysDiff'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days

# 5. Convert binary columns to proper int type
binary_cols = ['Scholarship', 'Hypertension', 'Diabetes', 'Alcoholism', 'SMS_received']
for col in binary_cols:
    df[col] = df[col].astype(int)

# 6. Convert NoShow to binary (1 = Yes, 0 = No)
df['NoShow'] = df['NoShow'].map({'Yes': 1, 'No': 0})

# 7. Check for age outliers
print("\nAge statistics:")
print(df['Age'].describe())

# 8. Handle age outliers if necessary (e.g., if there are negative ages)
if df['Age'].min() < 0:
    df = df[df['Age'] >= 0]

# 9. Check for duplicate appointments
duplicates = df.duplicated(subset=['PatientId', 'AppointmentDay'], keep=False)
print(f"\nNumber of duplicate appointments: {duplicates.sum()}")

# 10. Save the cleaned data
df.to_csv('cleaned_appointment_data.csv', index=False)

# 11. Preview the cleaned data
print("\nCleaned data preview:")
print(df.head())

# Basic visualization of appointments by neighborhood
plt.figure(figsize=(10, 6))
df['Neighbourhood'].value_counts().head(10).plot(kind='bar')
plt.title('Top 10 Neighborhoods by Number of Appointments')
plt.tight_layout()
plt.savefig('appointments_by_neighborhood.png')

# NoShow distribution
plt.figure(figsize=(8, 5))
df['NoShow'].value_counts().plot(kind='pie', autopct='%1.1f%%', labels=['Attended', 'No-show'])
plt.title('Appointment Attendance Distribution')
plt.ylabel('')
plt.savefig('noshow_distribution.png')