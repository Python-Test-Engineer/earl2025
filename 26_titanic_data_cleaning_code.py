import pandas as pd
import numpy as np

# Load the dataset
# In a real scenario, you would use: df = pd.read_csv('titanic.csv')
# For this example, we'll create the dataframe from the provided sample
data = {
    "PassengerId": [1, 2, 3, 4, 5, 6, 7, 8],
    "Survived": [0, 1, 1, 1, 0, 0, 0, 0],
    "Pclass": [3, 1, 3, 1, 3, 3, 1, 3],
    "Name": [
        "Braund, Mr. Owen Harris",
        "Cumings, Mrs. John Bradley (Florence Briggs Thayer)",
        "Heikkinen, Miss. Laina",
        "Futrelle, Mrs. Jacques Heath (Lily May Peel)",
        "Allen, Mr. William Henry",
        "Moran, Mr. James",
        "McCarthy, Mr. Timothy J",
        "Palsson, Master. Gosta Leonard",
    ],
    "Sex": ["male", "female", "female", "female", "male", "male", "male", "male"],
    "Age": [22.0, 38.0, 26.0, 35.0, 35.0, None, 54.0, 2.0],
    "SibSp": [1, 1, 0, 1, 0, 0, 0, 3],
    "Parch": [0, 0, 0, 0, 0, 0, 0, 1],
    "Ticket": [
        "A/5 21171",
        "PC 17599",
        "STON/O2. 3101282",
        "113803",
        "373450",
        "330877",
        "17463",
        "349909",
    ],
    "Fare": [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583, 51.8625, 21.075],
    "Cabin": ["", "C85", "", "C123", "", "", "E46", ""],
    "Embarked": ["S", "C", "S", "S", "S", "Q", "S", "S"],
}

df = pd.read_csv("26_titanic_original.csv")

# 1. Initial Data Exploration
print("Dataset Shape:", df.shape)
print("\nData Types:")
print(df.dtypes)

print("\nMissing Values Count:")
print(df.isnull().sum())

print("\nBasic Statistics for Numerical Columns:")
print(df.describe())

# 2. Handling Missing Values

# For Age: Impute with median (could also use mean or mode based on your needs)
age_median = df["Age"].median()
df["Age_filled"] = df["Age"].fillna(age_median)
print(f"\nImputing missing Age values with median: {age_median}")

# Alternative: Impute using a more sophisticated strategy (e.g., based on Pclass and Sex)
df["Age_by_class"] = df["Age"]  # Initialize with original values
# Example showing how to impute age based on passenger class:
for pclass in [1, 2, 3]:
    pclass_data = df[df["Pclass"] == pclass]
    if not pclass_data["Age"].empty:
        median_age = pclass_data["Age"].median()
        if not pd.isna(median_age):  # Ensure we have a valid median
            df.loc[(df["Age"].isnull()) & (df["Pclass"] == pclass), "Age_by_class"] = (
                median_age
            )
        else:
            df.loc[(df["Age"].isnull()) & (df["Pclass"] == pclass), "Age_by_class"] = (
                age_median
            )
    else:
        # If no data for a particular class, use global median
        df.loc[(df["Age"].isnull()) & (df["Pclass"] == pclass), "Age_by_class"] = (
            age_median
        )

# For Cabin: This column is highly sparse. Options include:
# Option 1: Create binary feature - has cabin or not
df["Has_Cabin"] = df["Cabin"].apply(
    lambda x: 0 if x is None or pd.isna(x) or x == "" else 1
)

# Option 2: Extract cabin letter as deck information
df["Deck"] = df["Cabin"].apply(
    lambda x: str(x)[0] if x is not None and pd.notna(x) and x != "" else "U"
)

# 3. Feature Engineering

# Extract title from name
df["Title"] = df["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
print("\nUnique Titles:", df["Title"].unique())

# Family Size feature
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1  # +1 for the passenger themselves
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

# 4. Categorical Encoding

# Option 1: One-hot encoding for categorical variables
try:
    df_encoded = pd.get_dummies(df, columns=["Sex", "Embarked", "Deck"])
except Exception as e:
    print(f"Error during one-hot encoding: {e}")
    # Fallback: Create manual encoding for the critical columns
    df["Sex_male"] = (df["Sex"] == "male").astype(int)
    df["Sex_female"] = (df["Sex"] == "female").astype(int)
    for port in df["Embarked"].unique():
        if isinstance(port, str):  # Ensure it's a valid string
            df[f"Embarked_{port}"] = (df["Embarked"] == port).astype(int)
    df_encoded = df.copy()

# 5. Data Type Conversions
df_encoded["Survived"] = df_encoded["Survived"].astype(bool)
df_encoded["Pclass"] = df_encoded["Pclass"].astype("category")

# 6. Final Dataset Preparation

# Get all the one-hot encoded columns (this approach is more robust)
sex_cols = [col for col in df_encoded.columns if col.startswith("Sex_")]
embarked_cols = [col for col in df_encoded.columns if col.startswith("Embarked_")]
deck_cols = [col for col in df_encoded.columns if col.startswith("Deck_")]

# Select relevant columns for your model (example)
features = (
    [
        "Pclass",
        "Age_filled",
        "SibSp",
        "Parch",
        "Fare",
        "Has_Cabin",
        "FamilySize",
        "IsAlone",
    ]
    + sex_cols
    + embarked_cols
    + deck_cols
)

# Make sure we're only selecting columns that actually exist
features = [f for f in features if f in df_encoded.columns]
print("\nActual Features Used:")
print(features)

# Create the final dataset for modeling
X = df_encoded[features]
y = df_encoded["Survived"]

# Sample code to check data quality after transformations
print("\nMissing Values After Transformation:")
print(X.isnull().sum())

# Display the transformed dataset
print("\nTransformed Data Sample:")
print(X.head())


df_encoded.to_csv("titanic_cleaned.csv", index=False)
print("\nCleaned dataset saved to 'titanic_cleaned.csv'")
