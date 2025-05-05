# Required packages:
# pip install pandas numpy matplotlib seaborn scikit-learn joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import warnings

warnings.filterwarnings("ignore")


def load_and_clean_data(filepath):
    """
    Load and preprocess the medical appointments dataset
    """
    print("Loading and cleaning data...")

    # Load the dataset
    df = pd.read_csv(filepath)

    # Display basic information
    print("Dataset Shape:", df.shape)
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nFirst 5 rows:")
    print(df.head())

    # Clean column names
    df.columns = df.columns.str.strip()

    # Convert 'No-show' to binary
    df["No_Show"] = df["No-show"].map({"Yes": 1, "No": 0})

    # Convert date columns to datetime and make them timezone-naive
    df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"]).dt.tz_localize(None)
    df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"]).dt.tz_localize(None)

    # Create new features
    df["DayDifference"] = (
        df["AppointmentDay"] - df["ScheduledDay"]
    ).dt.total_seconds() / (24 * 60 * 60)
    df["ScheduledDayOfWeek"] = df["ScheduledDay"].dt.dayofweek
    df["AppointmentDayOfWeek"] = df["AppointmentDay"].dt.dayofweek
    df["ScheduledHour"] = df["ScheduledDay"].dt.hour

    # Fix data types
    df["Age"] = df["Age"].astype(int)
    df["Scholarship"] = df["Scholarship"].astype(int)
    df["Hipertension"] = df["Hipertension"].astype(int)
    df["Diabetes"] = df["Diabetes"].astype(int)
    df["Alcoholism"] = df["Alcoholism"].astype(int)
    df["Handcap"] = df["Handcap"].astype(int)
    df["SMS_received"] = df["SMS_received"].astype(int)

    # Print the cleaned dataset
    print("\nCleaned Dataset Preview:")
    print(df.head())

    return df


def basic_statistics(df):
    """
    Generate basic statistics and plots for the dataset
    """
    print("\nGenerating basic statistics and visualizations...")

    # Set style for visualizations
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("viridis")

    # Create output directory if it doesn't exist
    import os

    if not os.path.exists("output"):
        os.makedirs("output")

    # Descriptive statistics
    print("\nDescriptive Statistics:")
    print(df.describe())

    # No-show rate
    no_show_rate = df["No_Show"].mean() * 100
    print(f"\nNo-show rate: {no_show_rate:.2f}%")

    # Distribution of categorical variables
    cat_columns = [
        "Gender",
        "Neighbourhood",
        "Scholarship",
        "Hipertension",
        "Diabetes",
        "Alcoholism",
        "Handcap",
        "SMS_received",
    ]

    fig, axs = plt.subplots(4, 2, figsize=(18, 24))
    axs = axs.flatten()

    for i, col in enumerate(cat_columns):
        if col == "Neighbourhood":
            top_neighborhoods = df[col].value_counts().head(10)
            top_neighborhoods.plot(kind="bar", ax=axs[i])
            axs[i].set_title(f"Top 10 {col}")
        else:
            df[col].value_counts().plot(kind="bar", ax=axs[i])
            axs[i].set_title(f"Distribution of {col}")
        axs[i].set_ylabel("Count")
        axs[i].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig("output/categorical_distributions.png")
    plt.close()

    # Age distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(df["Age"], bins=30, kde=True)
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.savefig("output/age_distribution.png")
    plt.close()

    # Day difference distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(df["DayDifference"], bins=30, kde=True)
    plt.title("Days Between Scheduling and Appointment")
    plt.xlabel("Days")
    plt.ylabel("Count")
    plt.savefig("output/day_difference_distribution.png")
    plt.close()


def analyze_no_show_patterns(df):
    """
    Analyze patterns in no-show appointments
    """
    print("\nAnalyzing no-show patterns...")

    # No-show rate by gender
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Gender", y="No_Show", data=df)
    plt.title("No-Show Rate by Gender")
    plt.xlabel("Gender")
    plt.ylabel("No-Show Rate")
    plt.savefig("output/no_show_by_gender.png")
    plt.close()

    # No-show rate by age groups
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 18, 35, 50, 65, 100],
        labels=["0-18", "19-35", "36-50", "51-65", "65+"],
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(x="AgeGroup", y="No_Show", data=df)
    plt.title("No-Show Rate by Age Group")
    plt.xlabel("Age Group")
    plt.ylabel("No-Show Rate")
    plt.savefig("output/no_show_by_age_group.png")
    plt.close()

    # No-show rate by day of week
    plt.figure(figsize=(10, 6))
    sns.barplot(x="AppointmentDayOfWeek", y="No_Show", data=df)
    plt.title("No-Show Rate by Day of Week")
    plt.xlabel("Day of Week (0=Monday, 6=Sunday)")
    plt.ylabel("No-Show Rate")
    plt.savefig("output/no_show_by_day.png")
    plt.close()

    # No-show rate by scheduling hour
    plt.figure(figsize=(12, 6))
    sns.barplot(x="ScheduledHour", y="No_Show", data=df)
    plt.title("No-Show Rate by Scheduling Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("No-Show Rate")
    plt.savefig("output/no_show_by_hour.png")
    plt.close()

    # No-show rate by SMS received
    plt.figure(figsize=(10, 6))
    sns.barplot(x="SMS_received", y="No_Show", data=df)
    plt.title("No-Show Rate by SMS Received")
    plt.xlabel("SMS Received (0=No, 1=Yes)")
    plt.ylabel("No-Show Rate")
    plt.savefig("output/no_show_by_sms.png")
    plt.close()

    # No-show rate by wait time
    plt.figure(figsize=(12, 6))
    day_diff_groups = [-1, 0, 1, 7, 15, 30, 60, df["DayDifference"].max()]
    df["WaitTimeGroup"] = pd.cut(
        df["DayDifference"],
        bins=day_diff_groups,
        labels=[
            "Same day",
            "1 day",
            "2-7 days",
            "8-15 days",
            "16-30 days",
            "31-60 days",
            "60+ days",
        ],
    )
    sns.barplot(x="WaitTimeGroup", y="No_Show", data=df)
    plt.title("No-Show Rate by Wait Time")
    plt.xlabel("Wait Time")
    plt.ylabel("No-Show Rate")
    plt.xticks(rotation=45)
    plt.savefig("output/no_show_by_wait_time.png")
    plt.close()

    return df


def analyze_demographic_factors(df):
    """
    Analyze demographic factors related to no-shows
    """
    print("\nAnalyzing demographic factors...")

    # No-show rate by medical conditions
    conditions = ["Hipertension", "Diabetes", "Alcoholism"]
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    for i, condition in enumerate(conditions):
        sns.barplot(x=condition, y="No_Show", data=df, ax=axs[i])
        axs[i].set_title(f"No-Show Rate by {condition}")
        axs[i].set_xlabel(f"{condition} (0=No, 1=Yes)")
        axs[i].set_ylabel("No-Show Rate")

    plt.tight_layout()
    plt.savefig("output/no_show_by_conditions.png")
    plt.close()

    # No-show rate by scholarship status
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Scholarship", y="No_Show", data=df)
    plt.title("No-Show Rate by Scholarship Status")
    plt.xlabel("Scholarship (0=No, 1=Yes)")
    plt.ylabel("No-Show Rate")
    plt.savefig("output/no_show_by_scholarship.png")
    plt.close()

    # No-show rate by neighborhood (top 10)
    top_neighborhoods = df["Neighbourhood"].value_counts().head(10).index
    neighborhood_no_show = (
        df[df["Neighbourhood"].isin(top_neighborhoods)]
        .groupby("Neighbourhood")["No_Show"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(12, 6))
    sns.barplot(x=neighborhood_no_show.index, y=neighborhood_no_show.values)
    plt.title("No-Show Rate by Top 10 Neighborhoods")
    plt.xlabel("Neighborhood")
    plt.ylabel("No-Show Rate")
    plt.xticks(rotation=90)
    plt.savefig("output/no_show_by_neighborhood.png")
    plt.close()

    # Correlation heatmap of numeric features
    numeric_cols = [
        "Age",
        "Scholarship",
        "Hipertension",
        "Diabetes",
        "Alcoholism",
        "Handcap",
        "SMS_received",
        "DayDifference",
        "No_Show",
    ]
    plt.figure(figsize=(12, 10))
    correlation = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f", mask=mask)
    plt.title("Correlation Matrix")
    plt.savefig("output/correlation_heatmap.png")
    plt.close()


def build_predictive_model(df):
    """
    Build and evaluate a predictive model for no-shows
    """
    print("\nBuilding predictive model...")

    # Prepare data for modeling
    features = [
        "Gender",
        "Age",
        "Scholarship",
        "Hipertension",
        "Diabetes",
        "Alcoholism",
        "Handcap",
        "SMS_received",
        "DayDifference",
        "ScheduledDayOfWeek",
        "AppointmentDayOfWeek",
        "ScheduledHour",
    ]

    # Create dummies for categorical variables
    df_model = pd.get_dummies(
        df[features + ["No_Show"]], columns=["Gender"], drop_first=True
    )

    # Split into train and test sets
    X = df_model.drop("No_Show", axis=1)
    y = df_model["No_Show"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train a Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)

    # Feature importance
    feature_importance = pd.DataFrame(
        {"Feature": X.columns, "Importance": model.feature_importances_}
    ).sort_values("Importance", ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x="Importance", y="Feature", data=feature_importance)
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig("output/feature_importance.png")
    plt.close()

    # Threshold optimization
    thresholds = np.arange(0, 1, 0.01)
    accuracy_scores = []

    for threshold in thresholds:
        y_pred_threshold = (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
        accuracy_scores.append(accuracy_score(y_test, y_pred_threshold))

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, accuracy_scores)
    plt.title("Accuracy vs Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    plt.axvline(
        x=thresholds[np.argmax(accuracy_scores)],
        color="r",
        linestyle="--",
        label=f"Optimal threshold: {thresholds[np.argmax(accuracy_scores)]:.2f}",
    )
    plt.legend()
    plt.savefig("output/threshold_optimization.png")
    plt.close()

    # Save the model
    joblib.dump(model, "output/no_show_prediction_model.pkl")
    with open("output/model_features.txt", "w") as f:
        for col in X.columns:
            f.write(f"{col}\n")

    print("Model saved as 'output/no_show_prediction_model.pkl'")

    return model, X.columns.tolist()


def schedule_sms_reminders(patient_id, appointment_date, phone_number):
    """
    Schedule SMS reminders at strategic intervals
    """
    today = datetime.now()
    # Make sure both datetime objects are timezone-naive
    appointment_date = pd.to_datetime(appointment_date).tz_localize(None)
    days_until_appointment = (appointment_date - today).days

    reminder_days = [7, 3, 1]  # Send reminders 7 days, 3 days, and 1 day before

    for days in reminder_days:
        if days_until_appointment >= days:
            reminder_date = appointment_date - pd.Timedelta(days=days)
            # Schedule SMS for this date
            message = (
                f"Reminder: Your appointment is in {days} day{'s' if days > 1 else ''}."
            )
            print(
                f"Scheduled reminder for {patient_id} on {reminder_date.strftime('%Y-%m-%d')}: {message}"
            )
            # In production: integrate with SMS API


def optimize_appointment_schedule(appointments_df, max_wait_days=15):
    """
    Reorganize appointment schedule to minimize wait times
    """
    # Ensure we have the necessary columns
    if "no_show_probability" not in appointments_df.columns:
        print("Error: 'no_show_probability' column not found. Run prediction first.")
        return

    # Identify high-risk appointments (predicted no-shows)
    high_risk_appointments = appointments_df[
        appointments_df["no_show_probability"] > 0.5
    ]

    # For high-risk appointments, try to reduce wait time
    for idx, appointment in high_risk_appointments.iterrows():
        current_wait = (
            appointment["AppointmentDay"] - appointment["ScheduledDay"]
        ).days

        if current_wait > max_wait_days:
            # In production: search for earlier slots and reschedule
            print(
                f"High-risk appointment {appointment['AppointmentID']} should be rescheduled before {appointment['ScheduledDay'] + pd.Timedelta(days=max_wait_days)}"
            )


def identify_high_risk_patients(df, model, X_columns):
    """
    Use the predictive model to identify patients at high risk of no-shows
    """
    # Prepare features
    features = [
        "Gender",
        "Age",
        "Scholarship",
        "Hipertension",
        "Diabetes",
        "Alcoholism",
        "Handcap",
        "SMS_received",
        "DayDifference",
        "ScheduledDayOfWeek",
        "AppointmentDayOfWeek",
        "ScheduledHour",
    ]

    # Create dummies for categorical variables if needed
    if "Gender_M" in X_columns and "Gender" in df.columns:
        df_features = pd.get_dummies(df[features], columns=["Gender"], drop_first=True)
    else:
        df_features = df[features].copy()

    # Ensure all columns from the training data are present
    for col in X_columns:
        if col not in df_features.columns:
            df_features[col] = 0

    # Select only the columns used during training
    X = df_features[X_columns]

    # Predict no-show probability
    df["no_show_probability"] = model.predict_proba(X)[:, 1]

    # Identify high-risk patients
    high_risk = df[df["no_show_probability"] > 0.7]

    print(f"\nIdentified {len(high_risk)} high-risk patients out of {len(df)} total")

    # Generate intervention strategies
    interventions = []
    for idx, patient in high_risk.iterrows():
        intervention = {
            "patient_id": patient["PatientId"],
            "appointment_id": patient["AppointmentID"],
            "no_show_probability": patient["no_show_probability"],
            "strategies": [],
        }

        # Add personalized strategies based on patient attributes
        if patient["Age"] > 65:
            intervention["strategies"].append("Offer transportation assistance")

        if patient["SMS_received"] == 0:
            intervention["strategies"].append("Add SMS reminder")

        if patient["DayDifference"] > 15:
            intervention["strategies"].append("Try to reschedule for earlier date")

        interventions.append(intervention)

    # Print sample interventions
    print("\nSample intervention strategies:")
    for i in range(min(5, len(interventions))):
        print(
            f"Patient {interventions[i]['patient_id']}: {interventions[i]['strategies']}"
        )

    return interventions


def optimize_scheduling_calendar(historical_data):
    """
    Create an optimized scheduling template based on historical attendance patterns
    """
    # Analyze no-show rates by day and hour
    day_hour_analysis = historical_data.groupby(
        ["AppointmentDayOfWeek", "ScheduledHour"]
    )["No_Show"].mean()

    # Create scheduling preference matrix (lower values = better slots)
    scheduling_preference = day_hour_analysis.unstack()

    # Identify optimal slots for different patient risk categories
    high_risk_slots = scheduling_preference.stack().sort_values().head(10)
    medium_risk_slots = scheduling_preference.stack().sort_values().iloc[10:30]

    print("\nRecommended slots for high-risk patients:")
    for (day, hour), rate in high_risk_slots.items():
        day_name = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ][day]
        print(f"  {day_name} at {hour}:00 (no-show rate: {rate:.2%})")

    # Save the scheduling preference matrix
    scheduling_preference.to_csv("output/scheduling_preference.csv")
    print("Scheduling preference matrix saved to 'output/scheduling_preference.csv'")

    return scheduling_preference


def generate_recommendations(df, model, X_columns):
    """
    Generate actionable recommendations based on the analysis
    """
    print("\nGenerating actionable recommendations...")

    # 1. Generate intervention strategies for high-risk patients
    interventions = identify_high_risk_patients(df, model, X_columns)

    # 2. Optimize scheduling calendar
    scheduling_preference = optimize_scheduling_calendar(df)

    # 3. Demonstrate SMS reminder system
    print("\nSMS Reminder System Demo:")
    sample_patient = df.iloc[0]
    schedule_sms_reminders(
        sample_patient["PatientId"],
        sample_patient["AppointmentDay"],
        "555-1234",  # Placeholder
    )

    # 4. Optimize wait times
    print("\nWait Time Optimization Demo:")
    df_sample = df.head(100).copy()  # Use a small sample for demonstration
    df_sample["no_show_probability"] = np.random.random(
        len(df_sample)
    )  # Random probabilities for demo
    optimize_appointment_schedule(df_sample)

    # 5. Generate final recommendations summary
    print("\n=== FINAL RECOMMENDATIONS ===")
    print("1. Implement Enhanced SMS Reminder System")
    print("   - Send multiple reminders (7 days, 3 days, 1 day before appointment)")
    print("   - Personalize message content based on patient risk factors")

    print("\n2. Optimize Appointment Wait Times")
    print(
        "   - Keep wait times under 15 days when possible, especially for high-risk patients"
    )
    print("   - Monitor correlation between wait time and no-show rates")

    print("\n3. Implement Targeted Interventions")
    print("   - Use predictive model to identify high-risk patients")
    print("   - Apply personalized strategies based on patient characteristics")

    print("\n4. Optimize Scheduling Practices")
    print("   - Schedule high-risk patients during optimal day/time slots")
    print("   - Reserve certain slots for patients with specific risk factors")

    print("\n5. Deploy Predictive Model")
    print("   - Integrate the model into the scheduling system")
    print("   - Continuously monitor and improve model performance")

    # Save recommendations to a text file
    with open("output/recommendations.txt", "w") as f:
        f.write("=== MEDICAL APPOINTMENT NO-SHOW RECOMMENDATIONS ===\n\n")
        f.write("1. Implement Enhanced SMS Reminder System\n")
        f.write(
            "   - Send multiple reminders (7 days, 3 days, 1 day before appointment)\n"
        )
        f.write("   - Personalize message content based on patient risk factors\n\n")

        f.write("2. Optimize Appointment Wait Times\n")
        f.write(
            "   - Keep wait times under 15 days when possible, especially for high-risk patients\n"
        )
        f.write("   - Monitor correlation between wait time and no-show rates\n\n")

        f.write("3. Implement Targeted Interventions\n")
        f.write("   - Use predictive model to identify high-risk patients\n")
        f.write(
            "   - Apply personalized strategies based on patient characteristics\n\n"
        )

        f.write("4. Optimize Scheduling Practices\n")
        f.write("   - Schedule high-risk patients during optimal day/time slots\n")
        f.write(
            "   - Reserve certain slots for patients with specific risk factors\n\n"
        )

        f.write("5. Deploy Predictive Model\n")
        f.write("   - Integrate the model into the scheduling system\n")
        f.write("   - Continuously monitor and improve model performance\n")

    print("\nRecommendations saved to 'output/recommendations.txt'")


def main():
    """
    Main function to run the complete analysis pipeline
    """
    print("=== MEDICAL APPOINTMENT NO-SHOW ANALYSIS ===")

    # File path - adjust as needed
    filepath = "25_medical_appointments.csv"

    # Load and clean data
    df = load_and_clean_data(filepath)

    # Generate basic statistics and visualizations
    basic_statistics(df)

    # Analyze no-show patterns
    df = analyze_no_show_patterns(df)

    # Analyze demographic factors
    analyze_demographic_factors(df)

    # Build predictive model
    model, X_columns = build_predictive_model(df)

    # Generate recommendations
    generate_recommendations(df, model, X_columns)

    print("\nAnalysis complete! All outputs saved to the 'output' directory.")


if __name__ == "__main__":
    main()
