#!/usr/bin/env python3
"""
Sales Data ETL and Analysis Program
==================================

A comprehensive ETL (Extract, Transform, Load) and data analysis program
for sales data with visualization capabilities.

Author: Cline AI Assistant
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

class SalesETLAnalyzer:
    """
    A comprehensive ETL and analysis class for sales data.
    """
    
    def __init__(self, data_file='sales_data.csv', output_dir='output'):
        """Initialize the analyzer with data file and output directory."""
        self.data_file = data_file
        self.output_dir = output_dir
        self.plots_dir = os.path.join(output_dir, 'plots')
        self.reports_dir = os.path.join(output_dir, 'reports')
        self.raw_data = None
        self.cleaned_data = None
        
        # Ensure output directories exist
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def extract_data(self):
        """Extract: Load data from CSV file."""
        print("📥 Extracting data from CSV...")
        try:
            self.raw_data = pd.read_csv(self.data_file)
            print(f"✅ Successfully loaded {len(self.raw_data)} records")
            print(f"📊 Data shape: {self.raw_data.shape}")
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def transform_data(self):
        """Transform: Clean and prepare data for analysis."""
        print("\n🔄 Transforming data...")
        
        if self.raw_data is None:
            print("❌ No raw data available for transformation")
            return False
        
        # Create a copy for transformation
        self.cleaned_data = self.raw_data.copy()
        
        try:
            # Convert date column to datetime
            self.cleaned_data['Date'] = pd.to_datetime(self.cleaned_data['Date'])
            
            # Extract date components
            self.cleaned_data['Year'] = self.cleaned_data['Date'].dt.year
            self.cleaned_data['Month'] = self.cleaned_data['Date'].dt.month
            self.cleaned_data['Day'] = self.cleaned_data['Date'].dt.day
            self.cleaned_data['Weekday'] = self.cleaned_data['Date'].dt.day_name()
            
            # Calculate derived metrics
            self.cleaned_data['Revenue_per_Unit'] = self.cleaned_data['Total'] / self.cleaned_data['Quantity']
            
            # Add product categories (simple categorization based on product name)
            self.cleaned_data['Category'] = self.cleaned_data['Product'].apply(self._categorize_product)
            
            # Calculate profit margin (assumed 30% for demo purposes)
            self.cleaned_data['Estimated_Profit'] = self.cleaned_data['Total'] * 0.30
            
            # Add size categories based on total sales
            self.cleaned_data['Sale_Size'] = pd.cut(
                self.cleaned_data['Total'], 
                bins=[0, 200, 500, 1000, float('inf')], 
                labels=['Small', 'Medium', 'Large', 'Extra Large']
            )
            
            print("✅ Data transformation completed successfully")
            print(f"🆕 Added columns: Year, Month, Day, Weekday, Revenue_per_Unit, Category, Estimated_Profit, Sale_Size")
            return True
            
        except Exception as e:
            print(f"❌ Error during transformation: {e}")
            return False
    
    def _categorize_product(self, product_name):
        """Helper method to categorize products."""
        product_lower = product_name.lower()
        if any(word in product_lower for word in ['laptop', 'computer', 'pc']):
            return 'Computers'
        elif any(word in product_lower for word in ['mouse', 'keyboard']):
            return 'Peripherals'
        elif any(word in product_lower for word in ['monitor', 'display', 'screen']):
            return 'Displays'
        elif any(word in product_lower for word in ['headphone', 'speaker', 'audio']):
            return 'Audio'
        else:
            return 'Other'
    
    def load_data(self):
        """Load: Save cleaned data to CSV file."""
        print("\n💾 Loading cleaned data...")
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cleaned_file = os.path.join(self.output_dir, f'cleaned_sales_data_{timestamp}.csv')
            self.cleaned_data.to_csv(cleaned_file, index=False)
            print(f"✅ Cleaned data saved to: {cleaned_file}")
            return True
        except Exception as e:
            print(f"❌ Error saving cleaned data: {e}")
            return False
    
    def analyze_data(self):
        """Perform comprehensive data analysis."""
        print("\n📊 Performing data analysis...")
        
        if self.cleaned_data is None:
            print("❌ No cleaned data available for analysis")
            return False
        
        # Basic statistics
        print("\n📈 BASIC STATISTICS")
        print("=" * 50)
        print(f"Total Records: {len(self.cleaned_data):,}")
        print(f"Total Revenue: ${self.cleaned_data['Total'].sum():,.2f}")
        print(f"Average Sale Value: ${self.cleaned_data['Total'].mean():.2f}")
        print(f"Total Units Sold: {self.cleaned_data['Quantity'].sum():,}")
        print(f"Number of Unique Products: {self.cleaned_data['Product'].nunique()}")
        print(f"Date Range: {self.cleaned_data['Date'].min().date()} to {self.cleaned_data['Date'].max().date()}")
        
        # Top products by revenue
        print("\n🏆 TOP PRODUCTS BY REVENUE")
        print("=" * 50)
        top_products = self.cleaned_data.groupby('Product')['Total'].sum().sort_values(ascending=False)
        for product, revenue in top_products.items():
            print(f"{product}: ${revenue:,.2f}")
        
        # Sales by category
        print("\n🏷️ SALES BY CATEGORY")
        print("=" * 50)
        category_sales = self.cleaned_data.groupby('Category')['Total'].sum().sort_values(ascending=False)
        for category, revenue in category_sales.items():
            print(f"{category}: ${revenue:,.2f}")
        
        return True
    
    def create_visualizations(self):
        """Create individual visualization plots."""
        print("\n📊 Creating visualizations...")
        
        if self.cleaned_data is None:
            print("❌ No data available for visualization")
            return False
        
        # 1. Daily Sales Trend
        self._plot_daily_sales_trend()
        
        # 2. Product Revenue Distribution
        self._plot_product_revenue()
        
        # 3. Category Sales Pie Chart
        self._plot_category_pie_chart()
        
        # 4. Quantity vs Price Scatter Plot
        self._plot_quantity_vs_price()
        
        # 5. Sales by Weekday
        self._plot_sales_by_weekday()
        
        # 6. Product Performance Heatmap
        self._plot_product_heatmap()
        
        # 7. Sale Size Distribution
        self._plot_sale_size_distribution()
        
        # 8. Revenue vs Profit Analysis
        self._plot_revenue_profit_analysis()
        
        print("✅ All visualizations created successfully!")
        return True
    
    def _plot_daily_sales_trend(self):
        """Create daily sales trend line chart."""
        plt.figure(figsize=(12, 6))
        daily_sales = self.cleaned_data.groupby('Date')['Total'].sum()
        
        plt.plot(daily_sales.index, daily_sales.values, marker='o', linewidth=2, markersize=8)
        plt.title('Daily Sales Trend', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Total Sales ($)', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value annotations
        for date, value in daily_sales.items():
            plt.annotate(f'${value:,.0f}', 
                        (date, value), 
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center',
                        fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'daily_sales_trend.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("📈 Saved: daily_sales_trend.png")
    
    def _plot_product_revenue(self):
        """Create product revenue bar chart."""
        plt.figure(figsize=(10, 6))
        product_revenue = self.cleaned_data.groupby('Product')['Total'].sum().sort_values(ascending=True)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(product_revenue)))
        bars = plt.barh(product_revenue.index, product_revenue.values, color=colors)
        
        plt.title('Product Revenue Analysis', fontsize=16, fontweight='bold')
        plt.xlabel('Total Revenue ($)', fontsize=12)
        plt.ylabel('Product', fontsize=12)
        
        # Add value labels on bars
        for bar in bars:
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2, 
                    f'${width:,.0f}', 
                    ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'product_revenue.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Saved: product_revenue.png")
    
    def _plot_category_pie_chart(self):
        """Create category sales pie chart."""
        plt.figure(figsize=(10, 8))
        category_sales = self.cleaned_data.groupby('Category')['Total'].sum()
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(category_sales)))
        wedges, texts, autotexts = plt.pie(category_sales.values, 
                                          labels=category_sales.index,
                                          autopct='%1.1f%%',
                                          colors=colors,
                                          startangle=90,
                                          explode=[0.05] * len(category_sales))
        
        plt.title('Sales Distribution by Category', fontsize=16, fontweight='bold')
        
        # Enhance text formatting
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'category_pie_chart.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("🥧 Saved: category_pie_chart.png")
    
    def _plot_quantity_vs_price(self):
        """Create quantity vs unit price scatter plot."""
        plt.figure(figsize=(10, 6))
        
        # Create scatter plot with different colors for each category
        categories = self.cleaned_data['Category'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
        
        for i, category in enumerate(categories):
            cat_data = self.cleaned_data[self.cleaned_data['Category'] == category]
            plt.scatter(cat_data['Quantity'], cat_data['Unit_Price'], 
                       c=[colors[i]], label=category, s=100, alpha=0.7)
        
        plt.title('Quantity vs Unit Price Analysis', fontsize=16, fontweight='bold')
        plt.xlabel('Quantity Sold', fontsize=12)
        plt.ylabel('Unit Price ($)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(self.cleaned_data['Quantity'], self.cleaned_data['Unit_Price'], 1)
        p = np.poly1d(z)
        plt.plot(self.cleaned_data['Quantity'], p(self.cleaned_data['Quantity']), 
                "r--", alpha=0.8, label='Trend Line')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'quantity_vs_price_scatter.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("📈 Saved: quantity_vs_price_scatter.png")
    
    def _plot_sales_by_weekday(self):
        """Create sales by weekday bar chart."""
        plt.figure(figsize=(10, 6))
        weekday_sales = self.cleaned_data.groupby('Weekday')['Total'].sum()
        
        # Reorder by weekday
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_sales = weekday_sales.reindex([day for day in weekday_order if day in weekday_sales.index])
        
        bars = plt.bar(weekday_sales.index, weekday_sales.values, 
                      color=plt.cm.plasma(np.linspace(0, 1, len(weekday_sales))))
        
        plt.title('Sales Performance by Weekday', fontsize=16, fontweight='bold')
        plt.xlabel('Day of Week', fontsize=12)
        plt.ylabel('Total Sales ($)', fontsize=12)
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:,.0f}',
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'sales_by_weekday.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("📅 Saved: sales_by_weekday.png")
    
    def _plot_product_heatmap(self):
        """Create product performance heatmap."""
        plt.figure(figsize=(8, 6))
        
        # Create a matrix for heatmap
        heatmap_data = self.cleaned_data.pivot_table(
            values=['Quantity', 'Total'], 
            index='Product', 
            aggfunc='sum'
        )
        
        # Normalize the data for better visualization
        heatmap_normalized = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())
        
        sns.heatmap(heatmap_normalized, annot=True, cmap='YlOrRd', 
                   fmt='.2f', cbar_kws={'label': 'Normalized Value'})
        
        plt.title('Product Performance Heatmap\n(Normalized Values)', fontsize=16, fontweight='bold')
        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Products', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'product_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("🔥 Saved: product_heatmap.png")
    
    def _plot_sale_size_distribution(self):
        """Create sale size distribution chart."""
        plt.figure(figsize=(10, 6))
        size_counts = self.cleaned_data['Sale_Size'].value_counts()
        
        # Create horizontal bar chart
        bars = plt.barh(size_counts.index, size_counts.values, 
                       color=plt.cm.coolwarm(np.linspace(0, 1, len(size_counts))))
        
        plt.title('Distribution of Sale Sizes', fontsize=16, fontweight='bold')
        plt.xlabel('Number of Sales', fontsize=12)
        plt.ylabel('Sale Size Category', fontsize=12)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{int(width)}', 
                    ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'sale_size_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("📏 Saved: sale_size_distribution.png")
    
    def _plot_revenue_profit_analysis(self):
        """Create revenue vs profit analysis chart."""
        plt.figure(figsize=(12, 6))
        
        # Create subplot with two y-axes
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        products = self.cleaned_data.groupby('Product').agg({
            'Total': 'sum',
            'Estimated_Profit': 'sum'
        }).reset_index()
        
        x_pos = np.arange(len(products))
        
        # Revenue bars
        bars1 = ax1.bar(x_pos - 0.2, products['Total'], 0.4, 
                       label='Revenue', color='skyblue', alpha=0.8)
        
        # Profit bars
        bars2 = ax1.bar(x_pos + 0.2, products['Estimated_Profit'], 0.4, 
                       label='Estimated Profit', color='lightcoral', alpha=0.8)
        
        ax1.set_xlabel('Products', fontsize=12)
        ax1.set_ylabel('Amount ($)', fontsize=12)
        ax1.set_title('Revenue vs Estimated Profit by Product', fontsize=16, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(products['Product'], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:,.0f}',
                    ha='center', va='bottom', fontsize=9)
        
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:,.0f}',
                    ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'revenue_profit_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("💰 Saved: revenue_profit_analysis.png")
    
    def generate_report(self):
        """Generate comprehensive analysis report."""
        print("\n📝 Generating analysis report...")
        
        if self.cleaned_data is None:
            print("❌ No data available for report generation")
            return False
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.reports_dir, f'sales_analysis_report_{timestamp}.txt')
        
        try:
            with open(report_file, 'w') as f:
                f.write("="*60 + "\n")
                f.write("SALES DATA ANALYSIS REPORT\n")
                f.write("="*60 + "\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Executive Summary
                f.write("EXECUTIVE SUMMARY\n")
                f.write("-"*20 + "\n")
                f.write(f"Total Revenue: ${self.cleaned_data['Total'].sum():,.2f}\n")
                f.write(f"Total Units Sold: {self.cleaned_data['Quantity'].sum():,}\n")
                f.write(f"Average Sale Value: ${self.cleaned_data['Total'].mean():.2f}\n")
                f.write(f"Number of Products: {self.cleaned_data['Product'].nunique()}\n")
                f.write(f"Analysis Period: {self.cleaned_data['Date'].min().date()} to {self.cleaned_data['Date'].max().date()}\n\n")
                
                # Product Performance
                f.write("PRODUCT PERFORMANCE\n")
                f.write("-"*20 + "\n")
                top_products = self.cleaned_data.groupby('Product')['Total'].sum().sort_values(ascending=False)
                for i, (product, revenue) in enumerate(top_products.items(), 1):
                    f.write(f"{i}. {product}: ${revenue:,.2f}\n")
                f.write("\n")
                
                # Category Analysis
                f.write("CATEGORY ANALYSIS\n")
                f.write("-"*20 + "\n")
                category_sales = self.cleaned_data.groupby('Category')['Total'].sum().sort_values(ascending=False)
                for category, revenue in category_sales.items():
                    percentage = (revenue / self.cleaned_data['Total'].sum()) * 100
                    f.write(f"{category}: ${revenue:,.2f} ({percentage:.1f}%)\n")
                f.write("\n")
                
                # Key Insights
                f.write("KEY INSIGHTS\n")
                f.write("-"*20 + "\n")
                best_seller = self.cleaned_data.loc[self.cleaned_data['Total'].idxmax()]
                highest_margin = self.cleaned_data.loc[self.cleaned_data['Unit_Price'].idxmax()]
                f.write(f"• Highest Single Sale: {best_seller['Product']} (${best_seller['Total']:,.2f})\n")
                f.write(f"• Highest Unit Price: {highest_margin['Product']} (${highest_margin['Unit_Price']:,.2f})\n")
                f.write(f"• Most Popular Category: {category_sales.index[0]}\n")
                f.write(f"• Average Profit Margin: {(self.cleaned_data['Estimated_Profit'].sum() / self.cleaned_data['Total'].sum() * 100):.1f}%\n")
                
            print(f"✅ Analysis report saved to: {report_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error generating report: {e}")
            return False
    
    def run_complete_analysis(self):
        """Run the complete ETL and analysis pipeline."""
        print("🚀 Starting Sales Data ETL and Analysis Pipeline")
        print("="*60)
        
        # ETL Process
        if not self.extract_data():
            return False
        
        if not self.transform_data():
            return False
            
        if not self.load_data():
            return False
        
        # Analysis Process
        if not self.analyze_data():
            return False
        
        if not self.create_visualizations():
            return False
            
        if not self.generate_report():
            return False
        
        print("\n🎉 Analysis completed successfully!")
        print(f"📁 Check the '{self.output_dir}' folder for results:")
        print(f"   📊 Plots: {self.plots_dir}")
        print(f"   📝 Reports: {self.reports_dir}")
        
        return True


def main():
    """Main function to run the ETL and analysis."""
    # Initialize the analyzer
    analyzer = SalesETLAnalyzer()
    
    # Run complete analysis
    success = analyzer.run_complete_analysis()
    
    if success:
        print("\n✨ All analysis completed successfully!")
        print("🔍 Review the generated plots and reports in the output directory.")
    else:
        print("\n❌ Analysis failed. Please check the error messages above.")


if __name__ == "__main__":
    main()
