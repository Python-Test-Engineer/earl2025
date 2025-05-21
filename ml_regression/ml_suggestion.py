#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

PROMPT USED: Suggest best ML approach and write me the code for it as ml_suggestion.py
Film Revenue Prediction Model
-----------------------------
This script implements a comprehensive approach to predicting movie revenue
using gradient boosting with feature importance analysis, hyperparameter tuning,
and model evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import shap
import warnings

warnings.filterwarnings("ignore")


class FilmRevenuePredictor:
    """
    A class to build, train, evaluate and analyze models for predicting film revenue.
    """

    def __init__(self, data_path):
        """Initialize with the path to the film dataset."""
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.feature_importance = None

    def load_and_prepare_data(self):
        """Load the dataset and prepare it for modeling."""
        print("Loading and preparing data...")

        # Load the dataset
        self.df = pd.read_csv(self.data_path)

        # Display basic information about the dataset
        print(f"Dataset shape: {self.df.shape}")
        print("\nDataset sample:")
        print(self.df.head())
        print("\nData types:")
        print(self.df.dtypes)
        print("\nMissing values per column:")
        print(self.df.isnull().sum())

        # Handle missing values
        # For this example, we'll use simple imputation strategies
        # In a real scenario, more sophisticated imputation might be needed
        numeric_cols = self.df.select_dtypes(include=["int64", "float64"]).columns
        for col in numeric_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col].fillna(self.df[col].median(), inplace=True)

        # Create feature matrix X and target vector y
        # Assuming 'revenue' is our target variable
        if "revenue" in self.df.columns:
            self.y = self.df["revenue"]
            self.X = self.df.drop("revenue", axis=1)
        else:
            raise ValueError("Dataset does not contain 'revenue' column")

        # Log transform the target variable (as revenue is typically right-skewed)
        self.y = np.log1p(self.y)

        # List of potential features based on the histograms
        self.potential_features = [
            "budget",
            "runtime",
            "popularity",
            "vote_average",
            "cast_size",
            "crew_size",
            "production_companies",
            "cast_tr",
            "director_tr",
            "production_company_tr",
            "title_length",
            "rel_year",
        ]

        # Filter X to only include potential features that are available in the dataset
        self.available_features = [
            f for f in self.potential_features if f in self.X.columns
        ]
        self.X = self.X[self.available_features]

        print(f"\nFeatures selected for modeling: {self.available_features}")

        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Testing set size: {self.X_test.shape[0]}")

        return self

    def perform_feature_engineering(self):
        """Perform feature engineering to improve model performance."""
        print("\nPerforming feature engineering...")

        # Create interaction features (e.g., budget per minute)
        if "budget" in self.X.columns and "runtime" in self.X.columns:
            self.X_train["budget_per_minute"] = self.X_train["budget"] / (
                self.X_train["runtime"] + 1
            )
            self.X_test["budget_per_minute"] = self.X_test["budget"] / (
                self.X_test["runtime"] + 1
            )

        # Create ratio features
        if "cast_size" in self.X.columns and "crew_size" in self.X.columns:
            self.X_train["cast_crew_ratio"] = self.X_train["cast_size"] / (
                self.X_train["crew_size"] + 1
            )
            self.X_test["cast_crew_ratio"] = self.X_test["cast_size"] / (
                self.X_test["crew_size"] + 1
            )

        # Create "star power" feature
        if "cast_tr" in self.X.columns and "cast_size" in self.X.columns:
            self.X_train["star_power"] = self.X_train["cast_tr"] / (
                self.X_train["cast_size"] + 1
            )
            self.X_test["star_power"] = self.X_test["cast_tr"] / (
                self.X_test["cast_size"] + 1
            )

        # Add year-based features (recency)
        if "rel_year" in self.X.columns:
            current_year = self.X_train["rel_year"].max()
            self.X_train["recency"] = current_year - self.X_train["rel_year"]
            self.X_test["recency"] = current_year - self.X_test["rel_year"]

        # Power transformations for skewed features
        skewed_features = [
            "budget",
            "popularity",
            "cast_tr",
            "director_tr",
            "production_company_tr",
        ]
        skewed_features = [f for f in skewed_features if f in self.X.columns]

        if skewed_features:
            pt = PowerTransformer(method="yeo-johnson")

            for feature in skewed_features:
                self.X_train[f"{feature}_transformed"] = pt.fit_transform(
                    self.X_train[[feature]].replace([np.inf, -np.inf], np.nan).fillna(0)
                )
                self.X_test[f"{feature}_transformed"] = pt.transform(
                    self.X_test[[feature]].replace([np.inf, -np.inf], np.nan).fillna(0)
                )

        print(f"Features after engineering: {self.X_train.columns.tolist()}")
        return self

    def train_model(self, model_type="xgboost"):
        """Train the selected model type with hyperparameter tuning."""
        print(f"\nTraining {model_type} model...")

        if model_type == "xgboost":
            # XGBoost with grid search for hyperparameter tuning
            param_grid = {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 7],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0],
            }

            base_model = xgb.XGBRegressor(
                objective="reg:squarederror", random_state=42, n_jobs=-1
            )

        elif model_type == "lightgbm":
            # LightGBM with grid search
            param_grid = {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1],
                "num_leaves": [31, 50, 100],
                "subsample": [0.8, 0.9, 1.0],
            }

            base_model = lgb.LGBMRegressor(
                objective="regression", random_state=42, n_jobs=-1
            )

        elif model_type == "gradient_boosting":
            # Gradient Boosting Regressor
            param_grid = {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5],
                "subsample": [0.8, 0.9],
            }

            base_model = GradientBoostingRegressor(random_state=42)

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Use smaller grid for demonstration (in practice use larger grid)
        # Simplified grid for faster execution
        simplified_param_grid = {
            "n_estimators": [100],
            "learning_rate": [0.1],
            "max_depth": [5] if model_type != "lightgbm" else None,
            "num_leaves": [31] if model_type == "lightgbm" else None,
            "subsample": [0.8],
        }
        # Remove None values
        simplified_param_grid = {
            k: v for k, v in simplified_param_grid.items() if v is not None
        }

        # Create the grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=simplified_param_grid,  # Use param_grid for full search
            cv=5,
            scoring="neg_mean_squared_error",
            verbose=1,
            n_jobs=-1,
        )

        # Fit the grid search to the training data
        grid_search.fit(self.X_train, self.y_train)

        # Get the best model
        self.model = grid_search.best_estimator_

        # Display best parameters
        print(f"Best parameters: {grid_search.best_params_}")

        return self

    def evaluate_model(self):
        """Evaluate the trained model on test data."""
        print("\nEvaluating model performance...")

        # Make predictions on the test set
        y_pred = self.model.predict(self.X_test)

        # Calculate various performance metrics
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Root Mean Squared Error: {rmse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")

        # Calculate metrics on original scale (exponentiating the log-transformed values)
        y_test_original = np.expm1(self.y_test)
        y_pred_original = np.expm1(y_pred)

        mse_original = mean_squared_error(y_test_original, y_pred_original)
        rmse_original = np.sqrt(mse_original)
        mae_original = mean_absolute_error(y_test_original, y_pred_original)

        print(f"\nMetrics on original scale:")
        print(f"RMSE: {rmse_original:.2f}")
        print(f"MAE: {mae_original:.2f}")

        # Cross-validation score
        cv_scores = cross_val_score(
            self.model, self.X, self.y, cv=5, scoring="neg_mean_squared_error"
        )
        cv_rmse = np.sqrt(-cv_scores.mean())
        print(f"\nCross-validation RMSE: {cv_rmse:.4f}")

        return self

    def analyze_feature_importance(self):
        """Analyze feature importance to understand what drives revenue."""
        print("\nAnalyzing feature importance...")

        # Get feature importances (method depends on model type)
        if hasattr(self.model, "feature_importances_"):
            self.feature_importance = pd.DataFrame(
                {
                    "Feature": self.X_train.columns,
                    "Importance": self.model.feature_importances_,
                }
            ).sort_values(by="Importance", ascending=False)
        else:
            # For models that don't have feature_importances_ attribute
            print("Feature importance not available for this model.")
            return self

        # Display the top features
        print("\nTop features by importance:")
        print(self.feature_importance.head(10))

        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x="Importance", y="Feature", data=self.feature_importance.head(15))
        plt.title("Feature Importance for Revenue Prediction")
        plt.tight_layout()
        plt.savefig("feature_importance.png")

        # Try to generate SHAP values for more detailed feature importance
        try:
            explainer = shap.Explainer(self.model, self.X_train)
            shap_values = explainer(self.X_train)

            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, self.X_train, plot_type="bar", show=False)
            plt.title("SHAP Feature Importance")
            plt.tight_layout()
            plt.savefig("shap_importance.png")
            print("\nSHAP importance analysis saved to 'shap_importance.png'")
        except Exception as e:
            print(f"SHAP analysis not available: {e}")

        return self

    def provide_revenue_optimization_insights(self):
        """Provide insights on how to optimize revenue based on the model."""
        print("\nRevenue Optimization Insights:")

        if self.feature_importance is None:
            print("Feature importance analysis required for optimization insights.")
            return self

        # Get top 5 important features
        top_features = self.feature_importance.head(5)["Feature"].tolist()

        print("\nBased on our model, the top revenue drivers are:")

        for i, feature in enumerate(top_features, 1):
            print(f"{i}. {feature}")

            # Provide specific recommendations based on feature name
            if "budget" in feature.lower():
                print(
                    "   - Optimal budget allocation is crucial. Consider increasing budget for projects"
                )
                print("     with strong indicators in other top features.")

            elif "popularity" in feature.lower():
                print(
                    "   - Focus on marketing strategies that boost popularity metrics."
                )
                print(
                    "     This might include social media campaigns, strategic partnerships, etc."
                )

            elif any(x in feature.lower() for x in ["cast", "star"]):
                print(
                    "   - Invest in casting decisions. Select actors with high 'star power'"
                )
                print("     (high cast_tr to cast_size ratio).")

            elif "director" in feature.lower():
                print(
                    "   - Director selection has significant impact. Prioritize directors"
                )
                print("     with proven track records (high director_tr values).")

            elif "production_company" in feature.lower():
                print(
                    "   - Partner with production companies that have extensive reach and distribution capability."
                )
                print("     Look for high production_company_tr values.")

            elif "runtime" in feature.lower():
                print(
                    "   - Optimize film runtime to match successful patterns in similar genres."
                )

            elif "vote" in feature.lower():
                print(
                    "   - Quality matters. Focus on production elements that historically lead to higher ratings."
                )

        print("\nRecommended next steps:")
        print(
            "1. Develop a portfolio approach for film investments (low, medium, high risk)"
        )
        print("2. Create a pre-production scoring system using these key features")
        print(
            "3. Use 'what-if' analysis with this model to simulate revenue for specific projects"
        )
        print(
            "4. Refine the model with additional data on genre, seasonality, and competition"
        )

        return self


def main():
    """Main function to execute the film revenue prediction workflow."""

    # Path to the film dataset
    # In a real scenario, replace this with the actual path
    data_path = "./ml_regression/movies.csv"

    try:
        # Create an instance of the FilmRevenuePredictor
        predictor = FilmRevenuePredictor(data_path)

        print("\n" + "=" * 80)
        print(" FILM REVENUE PREDICTION AND OPTIMIZATION MODEL ".center(80, "="))
        print("=" * 80 + "\n")

        # Simulate the workflow (comment out load_and_prepare_data in real application)
        print("NOTE: This script requires a real dataset to run properly.")
        print("The following is a demonstration of the workflow:")
        print("\n" + "-" * 80)

        # Display the expected workflow
        print("1. Load and prepare data")
        print("2. Perform feature engineering")
        print("3. Train XGBoost model with hyperparameter tuning")
        print("4. Evaluate model performance")
        print("5. Analyze feature importance")
        print("6. Provide revenue optimization insights")

        print("\n" + "-" * 80)
        print("EXPECTED OUTPUT FROM MODEL ANALYSIS:")

        print("HOW TO USE THIS SCRIPT:")
        print("1. Replace 'film_dataset.csv' with your actual dataset path")
        print("2. Uncomment the workflow execution code below")
        print("3. Run this script to perform the complete analysis")
        print("-" * 80 + "\n")

        # Uncomment these lines when using a real dataset

        predictor.load_and_prepare_data()
        predictor.perform_feature_engineering()
        predictor.train_model(
            model_type="xgboost"
        )  # Options: 'xgboost', 'lightgbm', 'gradient_boosting'
        predictor.evaluate_model()
        predictor.analyze_feature_importance()
        predictor.provide_revenue_optimization_insights()

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure the dataset exists and contains the required features.")


if __name__ == "__main__":
    main()
