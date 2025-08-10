# Healthcare Appointments No-Show Data Pipeline

A comprehensive data pipeline for analyzing healthcare appointment no-shows, built with Python and machine learning techniques.

## Features

- **Complete Data Pipeline**: Ingestion, cleaning, ETL, analysis, and reporting
- **Advanced Analytics**: Statistical analysis with comprehensive visualizations
- **Machine Learning**: Multiple ML models with performance evaluation
- **Model Interpretability**: SHAP analysis for understanding feature importance
- **Automated Reporting**: Comprehensive analysis reports with recommendations
- **Extensible Architecture**: Easy to modify and extend for different datasets

## Pipeline Components

### 1. Data Ingestion
- Loads data from CSV files or creates synthetic data based on provided sample
- Handles missing files gracefully with fallback to sample data generation
- Basic data validation and structure verification

### 2. Data Cleaning
- DateTime parsing and validation
- Missing value handling
- Data type corrections
- Outlier detection and removal
- Data consistency checks

### 3. Feature Engineering (ETL)
- Time-based features (weekday, lead time, scheduling patterns)
- Age group categorization
- Health condition aggregations
- Binary encoding for target variables
- Categorical variable encoding

### 4. Exploratory Data Analysis
- Target variable distribution analysis
- Age and demographic patterns
- Health condition impact analysis
- Temporal patterns (weekly, seasonal)
- Geographic analysis (neighbourhood patterns)
- Correlation analysis with heatmaps

### 5. Machine Learning Models
- **Logistic Regression**: Linear baseline model
- **Random Forest**: Tree-based ensemble model
- Model evaluation with AUC, accuracy, precision, and recall
- Feature importance analysis
- Cross-validation and performance comparison

### 6. SHAP Analysis
- Model interpretability using SHAP (SHapley Additive exPlanations)
- Feature contribution analysis
- Individual prediction explanations
- Global feature importance ranking

### 7. Comprehensive Reporting
- Executive summary with key metrics
- Detailed findings and insights
- Model performance comparison
- Data quality assessment
- Recommendations for further analysis

## Generated Visualizations

The pipeline generates the following visualizations:

1. **no_show_distribution.png** - Pie chart of appointment attendance
2. **age_distribution_by_noshow.png** - Age patterns by no-show status
3. **noshow_rate_by_age_group.png** - No-show rates across age groups
4. **lead_time_by_noshow.png** - Lead time analysis
5. **health_conditions_impact.png** - Health condition effects
6. **correlation_heatmap.png** - Feature correlation matrix
7. **neighbourhood_noshow_rates.png** - Geographic patterns
8. **weekly_noshow_pattern.png** - Weekly attendance patterns
9. **feature_importance.png** - Random Forest feature importance
10. **shap_summary_plot.png** - SHAP feature importance
11. **shap_waterfall_plot.png** - Individual prediction explanation
12. **shap_feature_importance.png** - SHAP-based feature ranking

## Installation

1. Clone or download this project
2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage (with synthetic data)
```python
from healthcare_data_pipeline import HealthcareDataPipeline

# Initialize pipeline
pipeline = HealthcareDataPipeline(output_dir="output")

# Run complete pipeline
pipeline.run_complete_pipeline()
```

### With Your Own Dataset
```python
# Run with your own CSV file
pipeline.run_complete_pipeline(data_file="your_data.csv")
```

### Running from Command Line
```bash
python healthcare_data_pipeline.py
```

## Expected Data Format

The pipeline expects CSV data with the following columns:

- `PatientId`: Unique patient identifier
- `AppointmentID`: Unique appointment identifier  
- `Gender`: Patient gender (M/F)
- `ScheduledDay`: When appointment was scheduled (ISO format)
- `AppointmentDay`: Actual appointment date (ISO format)
- `Age`: Patient age in years
- `Neighbourhood`: Patient neighborhood/location
- `Scholarship`: Binary (0/1) - financial assistance status
- `Hipertension`: Binary (0/1) - hypertension diagnosis
- `Diabetes`: Binary (0/1) - diabetes diagnosis
- `Alcoholism`: Binary (0/1) - alcoholism diagnosis
- `Handcap`: Binary (0/1) - disability status
- `SMS_received`: Binary (0/1) - SMS reminder sent
- `No-show`: Target variable ('Yes'/'No')

## Output Structure

```
output/
├── plots/                          # All generated visualizations
│   ├── no_show_distribution.png
│   ├── age_distribution_by_noshow.png
│   └── ...
├── reports/                        # Analysis reports
│   └── healthcare_analysis_report_YYYYMMDD_HHMMSS.txt
└── cleaned_healthcare_data_YYYYMMDD_HHMMSS.csv
```

## Key Insights from Analysis

The pipeline provides insights into:

- **Patient Demographics**: Age and gender patterns in no-show behavior
- **Health Conditions**: Impact of various medical conditions on attendance
- **Communication**: Effectiveness of SMS reminders
- **Timing**: Lead time effects and weekly patterns
- **Geographic**: Neighborhood-based attendance patterns
- **Predictive Factors**: Most important features for predicting no-shows

## ML Model Recommendations

The pipeline provides recommendations for advanced modeling:

1. **Gradient Boosting Models** (XGBoost, LightGBM, CatBoost)
2. **Neural Networks** for complex pattern recognition
3. **Time Series Analysis** for seasonal patterns
4. **Geospatial Analysis** for location-based insights
5. **Deep Learning** approaches (LSTM, Transformers)
6. **Causal Inference** methods for intervention analysis

## Extending the Pipeline

The modular design allows easy extension:

```python
class ExtendedHealthcarePipeline(HealthcareDataPipeline):
    def custom_analysis(self):
        # Add your custom analysis here
        pass
    
    def additional_models(self):
        # Add more ML models
        pass
```

## Requirements

- Python 3.8+
- pandas >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- scikit-learn >= 1.3.0
- shap >= 0.42.0

## License

This project is provided as-is for educational and research purposes.

## Contributing

Feel free to fork this project and submit improvements via pull requests.

## Support

For questions or issues, please refer to the documentation or create an issue in the project repository.
