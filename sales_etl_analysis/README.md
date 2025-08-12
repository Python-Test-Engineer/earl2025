# Sales ETL and Data Analysis Program

A comprehensive ETL (Extract, Transform, Load) and data analysis program for sales data with interactive visualizations and detailed reporting.

## Features

- **Extract**: Load data from CSV files
- **Transform**: Clean and process data with calculated fields
- **Load**: Save cleaned data to new CSV files
- **Analyze**: Comprehensive statistical analysis
- **Visualize**: Generate multiple types of charts and graphs
- **Report**: Create detailed analysis reports

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the program with the sample data:
```bash
python sales_etl_analyzer.py
```

### Using Your Own Data

```python
from sales_etl_analyzer import SalesETLAnalyzer

# Initialize analyzer
analyzer = SalesETLAnalyzer()

# Run full ETL pipeline
analyzer.run_full_etl_analysis('your_sales_data.csv')

# Display summary
analyzer.display_summary()
```

### Step-by-Step Usage

```python
from sales_etl_analyzer import SalesETLAnalyzer

# Initialize
analyzer = SalesETLAnalyzer()

# Extract data
analyzer.extract_data('your_sales_data.csv')

# Transform data
analyzer.transform_data()

# Load cleaned data
analyzer.load_data()

# Analyze data
analyzer.analyze_data()

# Create visualizations
analyzer.create_visualizations()

# Generate report
analyzer.generate_report()
```

## Expected CSV Format

Your CSV file should have the following columns:
- `date`: Transaction date (YYYY-MM-DD format)
- `product_id`: Unique product identifier
- `product_name`: Product name
- `category`: Product category
- `quantity`: Quantity sold
- `unit_price`: Price per unit
- `customer_id`: Customer identifier
- `customer_name`: Customer name
- `region`: Sales region
- `sales_rep`: Sales representative name

## Output

The program generates:

### 📊 Visualizations (saved to `output/plots/`)
- Sales by Category (Bar Chart)
- Sales by Region (Pie Chart)
- Daily Sales Trend (Line Chart)
- Top 10 Products (Bar Chart)
- Sales Rep Performance (Bar Chart)
- Quantity vs Price Scatter Plot
- Sales Heatmap (Category vs Region)

### 📋 Reports (saved to `output/reports/`)
- Comprehensive analysis report with statistics
- Sales breakdowns by category, region, and sales rep
- Top products analysis

### 💾 Data (saved to `output/`)
- Cleaned and transformed CSV data

## Analysis Features

- **Basic Statistics**: Total revenue, average order value, quantity sold
- **Sales by Category**: Revenue breakdown by product categories
- **Regional Analysis**: Sales performance by geographic regions
- **Sales Rep Performance**: Individual sales representative metrics
- **Product Analysis**: Top-performing products
- **Time Series Analysis**: Daily and monthly sales trends
- **Correlation Analysis**: Relationship between price and quantity

## Sample Data

The program includes sample sales data (`sample_sales_data.csv`) with:
- 40 sales transactions
- 3 product categories (Electronics, Furniture, Stationery)
- 4 sales regions (North, South, East, West)
- 4 sales representatives
- Date range: January-February 2024

## Class Methods

### SalesETLAnalyzer

- `extract_data(csv_file_path)`: Load data from CSV
- `transform_data()`: Clean and transform data
- `load_data(output_file)`: Save cleaned data
- `analyze_data()`: Perform statistical analysis
- `create_visualizations()`: Generate charts and graphs
- `generate_report()`: Create analysis report
- `run_full_etl_analysis(csv_file_path)`: Execute complete pipeline
- `display_summary()`: Show analysis summary

## Dependencies

- pandas: Data manipulation and analysis
- numpy: Numerical computing
- matplotlib: Plotting and visualization
- seaborn: Statistical data visualization
- pathlib: File system paths

## Error Handling

The program includes comprehensive error handling for:
- Missing or invalid CSV files
- Data format issues
- Missing required columns
- File I/O operations

## Customization

You can easily customize the program by:
- Modifying visualization styles and colors
- Adding new analysis metrics
- Changing output formats
- Adding new chart types
- Customizing report templates

## Example Output

```
🚀 Starting Full ETL and Analysis Pipeline
==================================================
📊 Extracting data from: sample_sales_data.csv
✅ Successfully extracted 40 records
📋 Columns: ['date', 'product_id', 'product_name', 'category', 'quantity', 'unit_price', 'customer_id', 'customer_name', 'region', 'sales_rep']

🔄 Transforming data...
✅ Converted date column to datetime
✅ Calculated total sales amount
✅ Extracted date components
✅ Data transformation complete. Final dataset: 40 records

💾 Cleaned data saved to: output/cleaned_sales_data_20250107_070000.csv

📈 Performing data analysis...
✅ Data analysis complete

📊 Creating visualizations...
✅ Visualizations saved to: output/plots

📋 Generating analysis report...
✅ Report saved to: output/reports/sales_analysis_report_20250107_070000.txt

🎉 ETL and Analysis Pipeline Complete!
📁 Check the 'output' directory for results:
   📊 Plots: output/plots
   📋 Reports: output/reports

============================================================
SALES ANALYSIS SUMMARY
============================================================
📊 Total Records: 40
💰 Total Revenue: $25,847.60
📈 Average Order Value: $646.19
📦 Total Quantity Sold: 267
📅 Date Range: 2024-01-15 to 2024-02-03

🏆 Top Product: Laptop Pro 15 ($7,799.94)
============================================================
```

## License

This program is provided as-is for educational and commercial use.
