#!/usr/bin/env python3
"""
Sales ETL and Data Analysis Program
==================================

A comprehensive ETL (Extract, Transform, Load) and data analysis program for sales data.
Features:
- Data extraction from CSV files
- Data cleaning and transformation
- Statistical analysis
- Interactive visualizations and plots
- Export capabilities

Author: Sales Analytics Team
Date: 2025-01-07
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SalesETLAnalyzer:
    """
    A comprehensive ETL and analysis class for sales data.
    """
    
    def __init__(self, csv_file_path=None):
        """
        Initialize the Sales ETL Analyzer.
        
        Args:
            csv_file_path (str): Path to the CSV file containing sales data
        """
        self.csv_file_path = csv_file_path
        self.raw_data = None
        self.clean_data = None
        self.analysis_results = {}
        
        # Create output directories
        self.output_dir = Path("output")
        self.plots_dir = self.output_dir / "plots"
        self.reports_dir = self.output_dir / "reports"
        
        for directory in [self.output_dir, self.plots_dir, self.reports_dir]:
            directory.mkdir(exist_ok=True)
    
    def extract_data(self, csv_file_path=None):
        """
        Extract data from CSV file.
        
        Args:
            csv_file_path (str): Path to CSV file (optional if set in __init__)
        
        Returns:
            pd.DataFrame: Raw extracted data
        """
        if csv_file_path:
            self.csv_file_path = csv_file_path
        
        if not self.csv_file_path:
            raise ValueError("No CSV file path provided")
        
        try:
            print(f"📊 Extracting data from: {self.csv_file_path}")
            self.raw_data = pd.read_csv(self.csv_file_path)
            print(f"✅ Successfully extracted {len(self.raw_data)} records")
            print(f"📋 Columns: {list(self.raw_data.columns)}")
            return self.raw_data
        except Exception as e:
            print(f"❌ Error extracting data: {str(e)}")
            raise
    
    def transform_data(self):
        """
        Transform and clean the extracted data.
        
        Returns:
            pd.DataFrame: Cleaned and transformed data
        """
        if self.raw_data is None:
            raise ValueError("No data to transform. Run extract_data() first.")
        
        print("\n🔄 Transforming data...")
        
        # Create a copy for transformation
        self.clean_data = self.raw_data.copy()
        
        # Convert date column to datetime
        if 'date' in self.clean_data.columns:
            self.clean_data['date'] = pd.to_datetime(self.clean_data['date'])
            print("✅ Converted date column to datetime")
        
        # Calculate total sales amount
        if 'quantity' in self.clean_data.columns and 'unit_price' in self.clean_data.columns:
            self.clean_data['total_sales'] = self.clean_data['quantity'] * self.clean_data['unit_price']
            print("✅ Calculated total sales amount")
        
        # Extract date components
        if 'date' in self.clean_data.columns:
            self.clean_data['year'] = self.clean_data['date'].dt.year
            self.clean_data['month'] = self.clean_data['date'].dt.month
            self.clean_data['day'] = self.clean_data['date'].dt.day
            self.clean_data['weekday'] = self.clean_data['date'].dt.day_name()
            print("✅ Extracted date components")
        
        # Clean text columns
        text_columns = self.clean_data.select_dtypes(include=['object']).columns
        for col in text_columns:
            if col != 'date':  # Skip date column
                self.clean_data[col] = self.clean_data[col].astype(str).str.strip()
        
        # Handle missing values
        missing_before = self.clean_data.isnull().sum().sum()
        self.clean_data = self.clean_data.dropna()
        missing_after = len(self.raw_data) - len(self.clean_data)
        
        if missing_after > 0:
            print(f"✅ Removed {missing_after} rows with missing values")
        
        print(f"✅ Data transformation complete. Final dataset: {len(self.clean_data)} records")
        return self.clean_data
    
    def load_data(self, output_file=None):
        """
        Load (save) the cleaned data to a new CSV file.
        
        Args:
            output_file (str): Output file path
        """
        if self.clean_data is None:
            raise ValueError("No cleaned data to load. Run transform_data() first.")
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"cleaned_sales_data_{timestamp}.csv"
        
        try:
            self.clean_data.to_csv(output_file, index=False)
            print(f"💾 Cleaned data saved to: {output_file}")
        except Exception as e:
            print(f"❌ Error saving data: {str(e)}")
            raise
    
    def analyze_data(self):
        """
        Perform comprehensive data analysis.
        
        Returns:
            dict: Analysis results
        """
        if self.clean_data is None:
            raise ValueError("No data to analyze. Run transform_data() first.")
        
        print("\n📈 Performing data analysis...")
        
        # Basic statistics
        self.analysis_results['basic_stats'] = {
            'total_records': len(self.clean_data),
            'date_range': {
                'start': self.clean_data['date'].min(),
                'end': self.clean_data['date'].max()
            },
            'total_revenue': self.clean_data['total_sales'].sum(),
            'average_order_value': self.clean_data['total_sales'].mean(),
            'total_quantity_sold': self.clean_data['quantity'].sum()
        }
        
        # Sales by category
        if 'category' in self.clean_data.columns:
            self.analysis_results['sales_by_category'] = (
                self.clean_data.groupby('category')['total_sales']
                .agg(['sum', 'mean', 'count'])
                .round(2)
            )
        
        # Sales by region
        if 'region' in self.clean_data.columns:
            self.analysis_results['sales_by_region'] = (
                self.clean_data.groupby('region')['total_sales']
                .agg(['sum', 'mean', 'count'])
                .round(2)
            )
        
        # Sales by sales rep
        if 'sales_rep' in self.clean_data.columns:
            self.analysis_results['sales_by_rep'] = (
                self.clean_data.groupby('sales_rep')['total_sales']
                .agg(['sum', 'mean', 'count'])
                .round(2)
            )
        
        # Top products
        if 'product_name' in self.clean_data.columns:
            self.analysis_results['top_products'] = (
                self.clean_data.groupby('product_name')['total_sales']
                .sum()
                .sort_values(ascending=False)
                .head(10)
                .round(2)
            )
        
        # Daily sales trend
        if 'date' in self.clean_data.columns:
            self.analysis_results['daily_sales'] = (
                self.clean_data.groupby('date')['total_sales']
                .sum()
                .round(2)
            )
        
        # Monthly sales trend
        if 'year' in self.clean_data.columns and 'month' in self.clean_data.columns:
            self.analysis_results['monthly_sales'] = (
                self.clean_data.groupby(['year', 'month'])['total_sales']
                .sum()
                .round(2)
            )
        
        print("✅ Data analysis complete")
        return self.analysis_results
    
    def create_visualizations(self):
        """
        Create comprehensive visualizations and save them as plots.
        """
        if self.clean_data is None:
            raise ValueError("No data to visualize. Run transform_data() first.")
        
        print("\n📊 Creating visualizations...")
        
        # Set up the plotting style
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
        # 1. Sales by Category (Bar Chart)
        if 'category' in self.clean_data.columns:
            plt.figure(figsize=(10, 6))
            category_sales = self.clean_data.groupby('category')['total_sales'].sum().sort_values(ascending=True)
            category_sales.plot(kind='barh', color='skyblue', edgecolor='black')
            plt.title('Total Sales by Category', fontsize=16, fontweight='bold')
            plt.xlabel('Total Sales ($)', fontsize=12)
            plt.ylabel('Category', fontsize=12)
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'sales_by_category.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Sales by Region (Pie Chart)
        if 'region' in self.clean_data.columns:
            plt.figure(figsize=(10, 8))
            region_sales = self.clean_data.groupby('region')['total_sales'].sum()
            plt.pie(region_sales.values, labels=region_sales.index, autopct='%1.1f%%', startangle=90)
            plt.title('Sales Distribution by Region', fontsize=16, fontweight='bold')
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'sales_by_region_pie.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Daily Sales Trend (Line Chart)
        if 'date' in self.clean_data.columns:
            plt.figure(figsize=(14, 6))
            daily_sales = self.clean_data.groupby('date')['total_sales'].sum()
            plt.plot(daily_sales.index, daily_sales.values, marker='o', linewidth=2, markersize=4)
            plt.title('Daily Sales Trend', fontsize=16, fontweight='bold')
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Total Sales ($)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'daily_sales_trend.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Top 10 Products (Horizontal Bar Chart)
        if 'product_name' in self.clean_data.columns:
            plt.figure(figsize=(12, 8))
            top_products = self.clean_data.groupby('product_name')['total_sales'].sum().sort_values(ascending=True).tail(10)
            top_products.plot(kind='barh', color='lightcoral', edgecolor='black')
            plt.title('Top 10 Products by Sales', fontsize=16, fontweight='bold')
            plt.xlabel('Total Sales ($)', fontsize=12)
            plt.ylabel('Product', fontsize=12)
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'top_products.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 5. Sales Rep Performance (Bar Chart)
        if 'sales_rep' in self.clean_data.columns:
            plt.figure(figsize=(10, 6))
            rep_sales = self.clean_data.groupby('sales_rep')['total_sales'].sum().sort_values(ascending=True)
            rep_sales.plot(kind='barh', color='lightgreen', edgecolor='black')
            plt.title('Sales Representative Performance', fontsize=16, fontweight='bold')
            plt.xlabel('Total Sales ($)', fontsize=12)
            plt.ylabel('Sales Representative', fontsize=12)
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'sales_rep_performance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 6. Quantity vs Price Scatter Plot
        if 'quantity' in self.clean_data.columns and 'unit_price' in self.clean_data.columns:
            plt.figure(figsize=(10, 6))
            plt.scatter(self.clean_data['unit_price'], self.clean_data['quantity'], 
                       alpha=0.6, c='purple', s=50)
            plt.title('Quantity vs Unit Price Relationship', fontsize=16, fontweight='bold')
            plt.xlabel('Unit Price ($)', fontsize=12)
            plt.ylabel('Quantity Sold', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'quantity_vs_price_scatter.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 7. Sales Heatmap by Category and Region
        if 'category' in self.clean_data.columns and 'region' in self.clean_data.columns:
            plt.figure(figsize=(10, 6))
            heatmap_data = self.clean_data.pivot_table(
                values='total_sales', 
                index='category', 
                columns='region', 
                aggfunc='sum',
                fill_value=0
            )
            sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlOrRd', cbar_kws={'label': 'Total Sales ($)'})
            plt.title('Sales Heatmap: Category vs Region', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'sales_heatmap_category_region.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✅ Visualizations saved to: {self.plots_dir}")
    
    def generate_report(self):
        """
        Generate a comprehensive analysis report.
        """
        if not self.analysis_results:
            self.analyze_data()
        
        print("\n📋 Generating analysis report...")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_file = self.reports_dir / f"sales_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("SALES DATA ANALYSIS REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated on: {timestamp}\n")
            f.write(f"Data source: {self.csv_file_path}\n")
            f.write("="*80 + "\n\n")
            
            # Basic Statistics
            f.write("BASIC STATISTICS\n")
            f.write("-"*40 + "\n")
            stats = self.analysis_results['basic_stats']
            f.write(f"Total Records: {stats['total_records']:,}\n")
            f.write(f"Date Range: {stats['date_range']['start'].strftime('%Y-%m-%d')} to {stats['date_range']['end'].strftime('%Y-%m-%d')}\n")
            f.write(f"Total Revenue: ${stats['total_revenue']:,.2f}\n")
            f.write(f"Average Order Value: ${stats['average_order_value']:,.2f}\n")
            f.write(f"Total Quantity Sold: {stats['total_quantity_sold']:,}\n\n")
            
            # Sales by Category
            if 'sales_by_category' in self.analysis_results:
                f.write("SALES BY CATEGORY\n")
                f.write("-"*40 + "\n")
                f.write(self.analysis_results['sales_by_category'].to_string())
                f.write("\n\n")
            
            # Sales by Region
            if 'sales_by_region' in self.analysis_results:
                f.write("SALES BY REGION\n")
                f.write("-"*40 + "\n")
                f.write(self.analysis_results['sales_by_region'].to_string())
                f.write("\n\n")
            
            # Sales by Rep
            if 'sales_by_rep' in self.analysis_results:
                f.write("SALES BY REPRESENTATIVE\n")
                f.write("-"*40 + "\n")
                f.write(self.analysis_results['sales_by_rep'].to_string())
                f.write("\n\n")
            
            # Top Products
            if 'top_products' in self.analysis_results:
                f.write("TOP 10 PRODUCTS\n")
                f.write("-"*40 + "\n")
                f.write(self.analysis_results['top_products'].to_string())
                f.write("\n\n")
        
        print(f"✅ Report saved to: {report_file}")
    
    def run_full_etl_analysis(self, csv_file_path=None):
        """
        Run the complete ETL and analysis pipeline.
        
        Args:
            csv_file_path (str): Path to CSV file
        """
        print("🚀 Starting Full ETL and Analysis Pipeline")
        print("="*50)
        
        try:
            # Extract
            self.extract_data(csv_file_path)
            
            # Transform
            self.transform_data()
            
            # Load
            self.load_data()
            
            # Analyze
            self.analyze_data()
            
            # Visualize
            self.create_visualizations()
            
            # Report
            self.generate_report()
            
            print("\n🎉 ETL and Analysis Pipeline Complete!")
            print(f"📁 Check the '{self.output_dir}' directory for results:")
            print(f"   📊 Plots: {self.plots_dir}")
            print(f"   📋 Reports: {self.reports_dir}")
            
        except Exception as e:
            print(f"❌ Pipeline failed: {str(e)}")
            raise
    
    def display_summary(self):
        """
        Display a summary of the analysis results.
        """
        if not self.analysis_results:
            print("No analysis results available. Run analyze_data() first.")
            return
        
        print("\n" + "="*60)
        print("SALES ANALYSIS SUMMARY")
        print("="*60)
        
        stats = self.analysis_results['basic_stats']
        print(f"📊 Total Records: {stats['total_records']:,}")
        print(f"💰 Total Revenue: ${stats['total_revenue']:,.2f}")
        print(f"📈 Average Order Value: ${stats['average_order_value']:,.2f}")
        print(f"📦 Total Quantity Sold: {stats['total_quantity_sold']:,}")
        print(f"📅 Date Range: {stats['date_range']['start'].strftime('%Y-%m-%d')} to {stats['date_range']['end'].strftime('%Y-%m-%d')}")
        
        if 'top_products' in self.analysis_results:
            print(f"\n🏆 Top Product: {self.analysis_results['top_products'].index[0]} (${self.analysis_results['top_products'].iloc[0]:,.2f})")
        
        print("="*60)


def main():
    """
    Main function to demonstrate the ETL and analysis capabilities.
    """
    print("🎯 Sales ETL and Data Analysis Program")
    print("="*50)
    
    # Initialize the analyzer
    analyzer = SalesETLAnalyzer()
    
    # Check if sample data exists
    sample_file = "sample_sales_data.csv"
    if os.path.exists(sample_file):
        print(f"📁 Found sample data file: {sample_file}")
        
        # Run the full pipeline
        analyzer.run_full_etl_analysis(sample_file)
        
        # Display summary
        analyzer.display_summary()
        
    else:
        print(f"❌ Sample data file '{sample_file}' not found.")
        print("Please ensure you have a CSV file with sales data.")
        print("\nExpected CSV format:")
        print("date,product_id,product_name,category,quantity,unit_price,customer_id,customer_name,region,sales_rep")
        
        # You can still create an analyzer and use it with your own data
        print("\nTo use with your own data:")
        print("analyzer = SalesETLAnalyzer()")
        print("analyzer.run_full_etl_analysis('your_sales_data.csv')")


if __name__ == "__main__":
    main()
