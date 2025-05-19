import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the data
df = pd.read_csv("25_medical_appointments.csv")

# 2. Initial inspection
print("Initial shape:", df.shape)
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())

# 3. Fix column naming inconsistency and typo
df = df.rename(columns={"No-show": "NoShow", "Handcap": "Handicap"})

# 4. Convert date columns to datetime
df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"])
df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"])

# 5. Create a wait time feature (days between scheduling and appointment)
df["WaitDays"] = (df["AppointmentDay"] - df["ScheduledDay"]).dt.days

# 6. Convert binary indicators to boolean
binary_columns = [
    "Scholarship",
    "Hipertension",
    "Diabetes",
    "Alcoholism",
    "Handicap",
    "SMS_received",
]
for col in binary_columns:
    df[col] = df[col].astype(bool)

# 7. Convert NoShow to boolean (Yes/No to True/False)
df["NoShow"] = df["NoShow"].map({"Yes": True, "No": False})

# 8. Check for age outliers
print("\nAge statistics:")
print(df["Age"].describe())

# Remove any negative ages or extreme outliers
df = df[df["Age"] >= 0]
df = df[df["Age"] < 120]  # Assuming 120 as reasonable upper limit

# 9. Standardize neighborhood names (keeping them uppercase but trimming spaces)
df["Neighbourhood"] = df["Neighbourhood"].str.strip().str.upper()

# 10. Final data summary
print("\nFinal cleaned data shape:", df.shape)
print("\nData types after cleaning:\n", df.dtypes)
print("\nSample of cleaned data:\n", df.head())

# Save the cleaned data
df.to_csv("25_medical_appointments_CLEANED.csv", index=False)
