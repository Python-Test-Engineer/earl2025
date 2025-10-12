# Sales Data ETL and Analysis Program

A comprehensive ETL (Extract, Transform, Load) and data analysis program for sales data with advanced visualization capabilities.

## 📋 Overview

This program demonstrates a complete data pipeline that:
- **Extracts** sales data from CSV files
- **Transforms** the data by adding calculated fields and categories
- **Loads** cleaned data into output files
- **Analyzes** the data with comprehensive statistics
- **Visualizes** results with 8 different chart types
- **Reports** findings in detailed text reports

## 🚀 Features

### ETL Pipeline
- ✅ **Data Extraction**: Load CSV data with error handling
- ✅ **Data Transformation**: Clean, categorize, and enhance data
- ✅ **Data Loading**: Save processed data with timestamps
- ✅ **Data Validation**: Comprehensive error checking throughout

### Analytics
- 📊 **Sales Performance Analysis**
- 📈 **Revenue Trend Analysis**
- 🏷️ **Product Categorization**
- 💰 **Profit Margin Calculations**
- 📅 **Time-based Analysis**

### Visualizations (8 Individual Charts)
1. **Daily Sales Trend** - Line chart showing sales over time
2. **Product Revenue Analysis** - Horizontal bar chart of product performance
3. **Category Sales Distribution** - Pie chart showing category breakdown
4. **Quantity vs Price Analysis** - Scatter plot with trend line
5. **Sales by Weekday** - Bar chart of weekday performance
6. **Product Performance Heatmap** - Normalized heatmap visualization
7. **Sale Size Distribution** - Bar chart of transaction sizes
8. **Revenue vs Profit Analysis** - Comparative bar chart

## 📁 Project Structure

```
earl_cline_demo_2/
│
├── sales_data.csv                 # Input data file
├── sales_etl_analyzer.py          # Main analysis program
├── requirements.txt               # Python dependencies
├── README.md                      # This documentation
│
└── output/                        # Generated outputs
    ├── plots/                     # Individual chart images (PNG)
    │   ├── daily_sales_trend.png
    │   ├── product_revenue.png
    │   ├── category_pie_chart.png
    │   ├── quantity_vs_price_scatter.png
    │   ├── sales_by_weekday.png
    │   ├── product_heatmap.png
    │   ├── sale_size_distribution.png
    │   └── revenue_profit_analysis.png
    │
    ├── reports/                   # Analysis reports
    │   └── sales_analysis_report_[timestamp].txt
    │
    └── cleaned_sales_data_[timestamp].csv
```

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. **Clone or download** this project to your local machine

2. **Navigate** to the project directory:
   ```bash
   cd earl_cline_demo_2
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🏃‍♂️ Usage

### Quick Start
Run the complete analysis pipeline:

```bash
python sales_etl_analyzer.py
```

### What Happens When You Run the Program

1. **📥 Data Extraction**
   - Loads sales data from `sales_data.csv`
   - Displays data shape and record count

2. **🔄 Data Transformation**
   - Converts dates to datetime format
   - Adds calculated fields (Revenue per Unit, Estimated Profit)
   - Categorizes products automatically
   - Creates sale size categories

3. **💾 Data Loading**
   - Saves cleaned data with timestamp
   - Creates output directory structure

4. **📊 Analysis**
   - Calculates comprehensive statistics
   - Shows top products and categories
   - Displays key performance metrics

5. **📈 Visualization**
   - Generates 8 individual chart images
   - Saves all plots as high-quality PNG files
   - Each chart saved separately in `output/plots/`

6. **📝 Reporting**
   - Creates detailed analysis report
   - Includes executive summary and insights
   - Saves timestamped report file

## 📊 Sample Data

The program includes sample sales data with the following structure:

| Date | Product | Quantity | Unit_Price | Total |
|------|---------|----------|------------|-------|
| 2024-01-15 | Laptop | 2 | 999.99 | 1999.98 |
| 2024-01-16 | Mouse | 5 | 29.99 | 149.95 |
| 2024-01-17 | Keyboard | 3 | 79.99 | 239.97 |
| 2024-01-18 | Monitor | 1 | 299.99 | 299.99 |
| 2024-01-19 | Headphones | 4 | 149.99 | 599.96 |

## 🎯 Key Insights Generated

The program automatically generates insights including:
- **Total Revenue**: Sum of all sales
- **Best Performing Products**: Ranked by revenue
- **Category Breakdown**: Sales distribution by product type
- **Profit Analysis**: Estimated profit margins
- **Temporal Patterns**: Sales trends over time
- **Price-Quantity Relationships**: Correlation analysis

## 🔧 Customization

### Using Your Own Data
1. Replace `sales_data.csv` with your data file
2. Ensure your CSV has columns: `Date`, `Product`, `Quantity`, `Unit_Price`, `Total`
3. Run the program normally

### Modifying Categories
Edit the `_categorize_product()` method in `sales_etl_analyzer.py` to customize product categorization logic.

### Adding New Visualizations
Add new plotting methods following the pattern of existing `_plot_*()` methods in the `SalesETLAnalyzer` class.

## 📈 Output Examples

### Console Output
```
🚀 Starting Sales Data ETL and Analysis Pipeline
============================================================
📥 Extracting data from CSV...
✅ Successfully loaded 5 records
📊 Data shape: (5, 5)

🔄 Transforming data...
✅ Data transformation completed successfully
🆕 Added columns: Year, Month, Day, Weekday, Revenue_per_Unit, Category, Estimated_Profit, Sale_Size

💾 Loading cleaned data...
✅ Cleaned data saved to: output/cleaned_sales_data_20251212_092855.csv

📊 Performing data analysis...

📈 BASIC STATISTICS
==================================================
Total Records: 5
Total Revenue: $3,289.86
Average Sale Value: $657.97
Total Units Sold: 15
Number of Unique Products: 5
Date Range: 2024-01-15 to 2024-01-19

📊 Creating visualizations...
📈 Saved: daily_sales_trend.png
📊 Saved: product_revenue.png
🥧 Saved: category_pie_chart.png
📈 Saved: quantity_vs_price_scatter.png
📅 Saved: sales_by_weekday.png
🔥 Saved: product_heatmap.png
📏 Saved: sale_size_distribution.png
💰 Saved: revenue_profit_analysis.png
✅ All visualizations created successfully!

📝 Generating analysis report...
✅ Analysis report saved to: output/reports/sales_analysis_report_20251212_092855.txt

🎉 Analysis completed successfully!
```

## 🛠️ Technical Details

### Dependencies
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Core plotting library
- **seaborn**: Statistical visualization

### Key Features
- **Error Handling**: Comprehensive error checking throughout the pipeline
- **Flexible Design**: Easy to extend with new analysis methods
- **Professional Output**: High-quality visualizations with proper formatting
- **Timestamped Files**: All outputs include timestamps to prevent overwrites
- **Modular Architecture**: Clean separation of ETL and analysis concerns

## 🎨 Visualization Features

All charts include:
- Professional styling with seaborn themes
- High-resolution output (300 DPI)
- Value annotations and labels
- Color-coded categories
- Grid lines and formatting
- Tight layout optimization

## 📝 License

This project is provided as-is for educational and demonstration purposes.

## 🙋‍♂️ Support

For questions or issues, please refer to the code comments and documentation within the Python files.

---
**Created by**: Cline AI Assistant  
**Date**: 2025  
**Purpose**: Sales Data ETL and Analysis Demonstration
