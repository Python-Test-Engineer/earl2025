# Data Cleaning Analysis for dirty_sales_data.csv

After examining the dataset, here are the issues I've identified and the actions needed to clean the data:

## Issues Found:
1. Missing values in multiple columns
2. Inconsistent product names (capitalization: "TABLET", "Tablet", "tablet")
3. Non-numeric values in numeric columns (Quantity and Price)
4. Inconsistent date formats (YYYY-MM-DD vs MM/DD/YYYY)
5. Non-standard status values (capitalization: "pending", "PROCESSING")
6. Duplicate date information (PurchaseDate and original_PurchaseDate)
7. Product specifications stored as string representation of JSON
8. ZipCode stored as float with decimal points instead of string

## Python Code to Clean the Data:

```python
import pandas as pd
import json
import numpy as np

# Load the data
df = pd.read_csv('dirty_sales_data.csv')

# 1. Convert date columns to consistent datetime format
df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'], errors='coerce')
df['original_PurchaseDate'] = pd.to_datetime(df['original_PurchaseDate'], format='%m/%d/%Y', errors='coerce')

# 2. Compare date columns and fill missing values
df['PurchaseDate'] = df['PurchaseDate'].fillna(df['original_PurchaseDate'])
df = df.drop(columns=['original_PurchaseDate'])  # Remove redundant column

# 3. Fix product names - standardize to lowercase
df['Product'] = df['Product'].str.lower()

# 4. Convert numeric columns to appropriate types
df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# 5. Standardize Status column
df['Status'] = df['Status'].str.lower()

# 6. Convert ZipCode to string without decimal
df['ZipCode'] = df['ZipCode'].fillna(0).astype(int).astype(str)
df['ZipCode'] = df['ZipCode'].replace('0', np.nan)

# 7. Parse ProductSpecifications from string to dictionary
def parse_specs(spec_str):
    if pd.isna(spec_str):
        return np.nan
    try:
        return json.loads(spec_str.replace("'", '"'))
    except:
        return np.nan

df['ProductSpecifications'] = df['ProductSpecifications'].apply(parse_specs)

# 8. Drop rows with critical missing information
df_cleaned = df.dropna(subset=['Product', 'Price'])

# Display basic info about the cleaned dataset
print(f"Original rows: {len(df)}, Cleaned rows: {len(df_cleaned)}")
print(f"Missing values in cleaned dataset:\n{df_cleaned.isnull().sum()}")

# Save cleaned dataset
df_cleaned.to_csv('cleaned_sales_data.csv', index=False)
```

This code addresses all the key issues in the dataset while keeping the implementation simple and straightforward. It converts dates to a standard format, fixes capitalization issues, handles missing values, converts numeric data to the correct types, and properly parses the JSON-like product specifications.