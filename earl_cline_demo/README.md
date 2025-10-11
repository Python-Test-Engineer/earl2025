# Sales Data ETL and Analysis Program 🚀

A comprehensive ETL (Extract, Transform, Load) pipeline and data analysis tool for sales data with automated visualization and reporting.

## Overview

This program processes sales data through a complete ETL pipeline and generates insightful visualizations and reports. It demonstrates modern data engineering practices with clean, modular Python code.

## Features

### 🔄 ETL Pipeline
- **Extract**: Load data from CSV files
- **Transform**: Clean data, add derived columns, categorize products
- **Load**: Save processed data with timestamps

### 📊 Data Analysis
- Summary statistics and key metrics
- Product performance analysis
- Category-based insights
- Time-series analysis

### 🎨 Visualizations
- Sales by Product (Bar Chart)
- Quantity vs Unit Price (Scatter Plot)
- Sales Distribution by Category (Pie Chart)
- Daily Sales Trend (Line Chart)
- Revenue per Unit Analysis
- Sales Heatmap by Day/Category
- Product Quantity Distribution
- Cumulative Sales Over Time
- Sales Amount Distribution

### 📝 Reporting
- Detailed analysis reports
- Executive summary
- Product performance breakdown
- Key insights and recommendations

## Project Structure

```
earl_cline_demo/
├── sales_data.csv              # Input sales data
├── sales_etl_analyzer.py       # Main ETL and analysis classes
├── demo.py                     # Simple demo script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── output/                     # Generated outputs
    ├── cleaned_sales_data_*.csv
    ├── plots/
    │   └── sales_analysis_dashboard.png
    └── reports/
        └── sales_analysis_report_*.txt
```

## Installation

1. Ensure you have Python 3.7+ installed
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start (Recommended)
Run the demo script for a guided experience:

```bash
python demo.py
```

### Direct Execution
Run the main analysis program:

```bash
python sales_etl_analyzer.py
```

### As a Module
Import and use the classes in your own code:

```python
from sales_etl_analyzer import SalesETL, SalesAnalyzer, SalesReporter

# Initialize ETL
etl = SalesETL("your_data.csv")

# Run ETL Process
if etl.extract() and etl.transform() and etl.load():
    # Run Analysis
    analyzer = SalesAnalyzer(etl.cleaned_data)
    analyzer.generate_summary_stats()
    analyzer.create_visualizations()
    
    # Generate Report
    reporter = SalesReporter(etl.cleaned_data)
    reporter.generate_report()
```

## Input Data Format

The program expects CSV data with the following columns:
- `Date`: Transaction date (YYYY-MM-DD format)
- `Product`: Product name
- `Quantity`: Number of units sold
- `Unit_Price`: Price per unit
- `Total`: Total transaction value

### Sample Data
```csv
Date,Product,Quantity,Unit_Price,Total
2024-01-15,Laptop,2,999.99,1999.98
2024-01-16,Mouse,5,29.99,149.95
2024-01-17,Keyboard,3,79.99,239.97
```

## Output Files

The program generates several output files in the `output/` directory:

### Processed Data
- `cleaned_sales_data_YYYYMMDD_HHMMSS.csv`: Cleaned data with derived columns

### Visualizations
- `plots/sales_analysis_dashboard.png`: Comprehensive 9-panel dashboard

### Reports
- `reports/sales_analysis_report_YYYYMMDD_HHMMSS.txt`: Detailed analysis report

## Class Structure

### `SalesETL`
Handles the Extract, Transform, Load operations:
- `extract()`: Load data from CSV
- `transform()`: Clean and enhance data
- `load()`: Save processed data

### `SalesAnalyzer`
Performs data analysis and visualization:
- `generate_summary_stats()`: Calculate key metrics
- `create_visualizations()`: Generate comprehensive plots

### `SalesReporter`
Creates detailed reports:
- `generate_report()`: Generate analysis report with insights

## Key Metrics Analyzed

- **Total Revenue**: Sum of all sales
- **Total Units Sold**: Sum of all quantities
- **Average Order Value**: Mean transaction value
- **Product Performance**: Sales and units by product
- **Category Analysis**: Performance by product category
- **Daily Trends**: Sales patterns over time

## Requirements

- Python 3.7+
- pandas >= 1.5.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- numpy >= 1.21.0

## Sample Output

```
🚀 SALES DATA ETL & ANALYSIS PIPELINE
==================================================
📥 Extracting data from CSV...
✅ Successfully loaded 5 records
🔄 Transforming data...
✅ Data transformed successfully
💾 Loading processed data...
✅ Cleaned data saved to: output/cleaned_sales_data_20241011_130739.csv

📊 SALES DATA SUMMARY
==================================================
📈 Total Records: 5
💰 Total Revenue: $3,289.85
📦 Total Units Sold: 15
💵 Average Order Value: $657.97
📅 Date Range: 2024-01-15 to 2024-01-19

🏆 TOP PERFORMING PRODUCTS:
  1. Laptop: $1,999.98 (2 units)
  2. Headphones: $599.96 (4 units)
  3. Monitor: $299.99 (1 units)
```

## Extensibility

The program is designed to be easily extended:

1. **Add new data sources**: Modify the `extract()` method
2. **Enhance transformations**: Add logic to the `transform()` method
3. **Create new visualizations**: Add plots to `create_visualizations()`
4. **Custom reports**: Extend the `SalesReporter` class

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to fork this project and submit pull requests for improvements!

---

*Created as part of the Earl 2025 AI Engineering demonstration series.*
