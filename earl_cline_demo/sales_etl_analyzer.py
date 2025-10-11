"""
Sales Data ETL and Analysis Program
==================================
A comprehensive ETL pipeline and data analysis tool for sales data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

class SalesETL:
    """Extract, Transform, Load operations for sales data"""
    
    def __init__(self, input_file):
        self.input_file = input_file
        self.raw_data = None
        self.cleaned_data = None
        self.output_dir = "output"
        
    def extract(self):
        """Extract data from CSV file"""
        print("📥 Extracting data from CSV...")
        try:
            self.raw_data = pd.read_csv(self.input_file)
            print(f"✅ Successfully loaded {len(self.raw_data)} records")
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def transform(self):
        """Transform and clean the data"""
        print("🔄 Transforming data...")
        if self.raw_data is None:
            print("❌ No data to transform. Run extract() first.")
            return False
        
        # Make a copy for transformation
        self.cleaned_data = self.raw_data.copy()
        
        # Convert Date column to datetime
        self.cleaned_data['Date'] = pd.to_datetime(self.cleaned_data['Date'])
        
        # Add derived columns
        self.cleaned_data['Year'] = self.cleaned_data['Date'].dt.year
        self.cleaned_data['Month'] = self.cleaned_data['Date'].dt.month
        self.cleaned_data['Day_of_Week'] = self.cleaned_data['Date'].dt.day_name()
        self.cleaned_data['Revenue_per_Unit'] = self.cleaned_data['Total'] / self.cleaned_data['Quantity']
        
        # Add product categories based on product type
        def categorize_product(product):
            if product in ['Laptop', 'Monitor']:
                return 'Hardware_Large'
            elif product in ['Mouse', 'Keyboard']:
                return 'Accessories'
            else:
                return 'Electronics'
        
        self.cleaned_data['Product_Category'] = self.cleaned_data['Product'].apply(categorize_product)
        
        # Data quality checks
        print(f"✅ Data transformed successfully")
        print(f"   - Added derived columns: Year, Month, Day_of_Week, Revenue_per_Unit, Product_Category")
        print(f"   - Date range: {self.cleaned_data['Date'].min()} to {self.cleaned_data['Date'].max()}")
        
        return True
    
    def load(self):
        """Load processed data to output files"""
        print("💾 Loading processed data...")
        if self.cleaned_data is None:
            print("❌ No processed data to load. Run transform() first.")
            return False
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/plots", exist_ok=True)
        
        # Save cleaned data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{self.output_dir}/cleaned_sales_data_{timestamp}.csv"
        self.cleaned_data.to_csv(output_file, index=False)
        
        print(f"✅ Cleaned data saved to: {output_file}")
        return True

class SalesAnalyzer:
    """Data analysis and visualization for sales data"""
    
    def __init__(self, data):
        self.data = data
        self.output_dir = "output"
        
    def generate_summary_stats(self):
        """Generate summary statistics"""
        print("\n📊 SALES DATA SUMMARY")
        print("=" * 50)
        
        print(f"📈 Total Records: {len(self.data)}")
        print(f"💰 Total Revenue: ${self.data['Total'].sum():,.2f}")
        print(f"📦 Total Units Sold: {self.data['Quantity'].sum()}")
        print(f"💵 Average Order Value: ${self.data['Total'].mean():.2f}")
        print(f"📅 Date Range: {self.data['Date'].min().strftime('%Y-%m-%d')} to {self.data['Date'].max().strftime('%Y-%m-%d')}")
        
        print("\n🏆 TOP PERFORMING PRODUCTS:")
        top_products = self.data.groupby('Product').agg({
            'Total': 'sum',
            'Quantity': 'sum'
        }).sort_values('Total', ascending=False)
        
        for idx, (product, row) in enumerate(top_products.iterrows(), 1):
            print(f"  {idx}. {product}: ${row['Total']:,.2f} ({row['Quantity']} units)")
            
        print("\n📈 BY CATEGORY:")
        category_stats = self.data.groupby('Product_Category').agg({
            'Total': 'sum',
            'Quantity': 'sum'
        }).sort_values('Total', ascending=False)
        
        for category, row in category_stats.iterrows():
            print(f"  {category}: ${row['Total']:,.2f} ({row['Quantity']} units)")
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n🎨 Creating visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Sales by Product (Bar Chart)
        ax1 = plt.subplot(3, 3, 1)
        product_sales = self.data.groupby('Product')['Total'].sum().sort_values(ascending=False)
        bars = ax1.bar(product_sales.index, product_sales.values, color=plt.cm.Set3(np.arange(len(product_sales))))
        ax1.set_title('Total Sales by Product', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Sales ($)')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:,.0f}', ha='center', va='bottom')
        
        # 2. Quantity vs Unit Price (Scatter Plot)
        ax2 = plt.subplot(3, 3, 2)
        scatter = ax2.scatter(self.data['Unit_Price'], self.data['Quantity'], 
                            c=self.data['Total'], cmap='viridis', s=100, alpha=0.7)
        ax2.set_title('Quantity vs Unit Price', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Unit Price ($)')
        ax2.set_ylabel('Quantity')
        plt.colorbar(scatter, label='Total Sales ($)')
        
        # 3. Sales by Category (Pie Chart)
        ax3 = plt.subplot(3, 3, 3)
        category_sales = self.data.groupby('Product_Category')['Total'].sum()
        wedges, texts, autotexts = ax3.pie(category_sales.values, labels=category_sales.index, 
                                          autopct='%1.1f%%', startangle=90)
        ax3.set_title('Sales Distribution by Category', fontsize=14, fontweight='bold')
        
        # 4. Daily Sales Trend (Line Chart)
        ax4 = plt.subplot(3, 3, 4)
        daily_sales = self.data.groupby('Date')['Total'].sum()
        ax4.plot(daily_sales.index, daily_sales.values, marker='o', linewidth=2, markersize=8)
        ax4.set_title('Daily Sales Trend', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Sales ($)')
        plt.xticks(rotation=45)
        
        # 5. Revenue per Unit Analysis
        ax5 = plt.subplot(3, 3, 5)
        revenue_per_unit = self.data.groupby('Product')['Revenue_per_Unit'].first()
        bars = ax5.bar(revenue_per_unit.index, revenue_per_unit.values, color=plt.cm.Pastel1(np.arange(len(revenue_per_unit))))
        ax5.set_title('Revenue per Unit by Product', fontsize=14, fontweight='bold')
        ax5.set_ylabel('Revenue per Unit ($)')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:.2f}', ha='center', va='bottom')
        
        # 6. Sales Heatmap by Day
        ax6 = plt.subplot(3, 3, 6)
        pivot_data = self.data.pivot_table(values='Total', index='Day_of_Week', 
                                          columns='Product_Category', fill_value=0)
        sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax6)
        ax6.set_title('Sales Heatmap: Day vs Category', fontsize=14, fontweight='bold')
        
        # 7. Product Quantity Distribution
        ax7 = plt.subplot(3, 3, 7)
        self.data.boxplot(column='Quantity', by='Product', ax=ax7)
        ax7.set_title('Quantity Distribution by Product', fontsize=14, fontweight='bold')
        ax7.set_xlabel('Product')
        ax7.set_ylabel('Quantity')
        plt.xticks(rotation=45, ha='right')
        
        # 8. Cumulative Sales
        ax8 = plt.subplot(3, 3, 8)
        cumulative_sales = self.data.sort_values('Date')['Total'].cumsum()
        ax8.plot(self.data.sort_values('Date')['Date'], cumulative_sales, 
                linewidth=3, color='green', marker='o')
        ax8.set_title('Cumulative Sales Over Time', fontsize=14, fontweight='bold')
        ax8.set_xlabel('Date')
        ax8.set_ylabel('Cumulative Sales ($)')
        plt.xticks(rotation=45)
        
        # 9. Sales Distribution
        ax9 = plt.subplot(3, 3, 9)
        ax9.hist(self.data['Total'], bins=10, edgecolor='black', alpha=0.7, color='skyblue')
        ax9.set_title('Sales Amount Distribution', fontsize=14, fontweight='bold')
        ax9.set_xlabel('Sales Amount ($)')
        ax9.set_ylabel('Frequency')
        
        plt.tight_layout()
        
        # Save the plot
        plot_file = f"{self.output_dir}/plots/sales_analysis_dashboard.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✅ Dashboard saved to: {plot_file}")
        
        # Show the plot
        plt.show()
        
        return plot_file

class SalesReporter:
    """Generate comprehensive sales reports"""
    
    def __init__(self, data):
        self.data = data
        self.output_dir = "output"
        
    def generate_report(self):
        """Generate detailed analysis report"""
        print("\n📝 Generating analysis report...")
        
        os.makedirs(f"{self.output_dir}/reports", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"{self.output_dir}/reports/sales_analysis_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("SALES DATA ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary Statistics
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Records Analyzed: {len(self.data)}\n")
            f.write(f"Total Revenue: ${self.data['Total'].sum():,.2f}\n")
            f.write(f"Total Units Sold: {self.data['Quantity'].sum()}\n")
            f.write(f"Average Order Value: ${self.data['Total'].mean():.2f}\n")
            f.write(f"Date Range: {self.data['Date'].min().strftime('%Y-%m-%d')} to {self.data['Date'].max().strftime('%Y-%m-%d')}\n\n")
            
            # Product Performance
            f.write("PRODUCT PERFORMANCE ANALYSIS\n")
            f.write("-" * 30 + "\n")
            product_stats = self.data.groupby('Product').agg({
                'Total': ['sum', 'mean', 'count'],
                'Quantity': 'sum',
                'Unit_Price': 'first'
            }).round(2)
            
            for product in product_stats.index:
                f.write(f"\n{product}:\n")
                f.write(f"  Total Sales: ${product_stats.loc[product, ('Total', 'sum')]:,.2f}\n")
                f.write(f"  Average Sale: ${product_stats.loc[product, ('Total', 'mean')]:,.2f}\n")
                f.write(f"  Units Sold: {product_stats.loc[product, ('Quantity', 'sum')]}\n")
                f.write(f"  Unit Price: ${product_stats.loc[product, ('Unit_Price', 'first')]:,.2f}\n")
            
            # Category Analysis
            f.write("\n\nCATEGORY ANALYSIS\n")
            f.write("-" * 20 + "\n")
            category_stats = self.data.groupby('Product_Category').agg({
                'Total': 'sum',
                'Quantity': 'sum'
            }).round(2)
            
            for category in category_stats.index:
                f.write(f"{category}: ${category_stats.loc[category, 'Total']:,.2f} ({category_stats.loc[category, 'Quantity']} units)\n")
            
            # Insights and Recommendations
            f.write("\n\nKEY INSIGHTS & RECOMMENDATIONS\n")
            f.write("-" * 35 + "\n")
            
            # Find best performing product
            best_product = self.data.groupby('Product')['Total'].sum().idxmax()
            best_revenue = self.data.groupby('Product')['Total'].sum().max()
            
            f.write(f"1. Top Performer: {best_product} generated ${best_revenue:,.2f} in revenue\n")
            
            # Find highest margin product
            highest_price = self.data.loc[self.data['Unit_Price'].idxmax(), 'Product']
            f.write(f"2. Premium Product: {highest_price} has the highest unit price\n")
            
            # Volume leader
            volume_leader = self.data.groupby('Product')['Quantity'].sum().idxmax()
            volume_count = self.data.groupby('Product')['Quantity'].sum().max()
            f.write(f"3. Volume Leader: {volume_leader} sold {volume_count} units\n")
            
            f.write(f"\n4. Sales are concentrated in {len(self.data['Product'].unique())} products over {len(self.data['Date'].unique())} days\n")
            f.write(f"5. Average daily revenue: ${self.data.groupby('Date')['Total'].sum().mean():.2f}\n")
            
        print(f"✅ Report saved to: {report_file}")
        return report_file

def main():
    """Main execution function"""
    print("🚀 SALES DATA ETL & ANALYSIS PIPELINE")
    print("=" * 50)
    
    # Initialize ETL
    etl = SalesETL("sales_data.csv")
    
    # Run ETL Process
    if etl.extract() and etl.transform() and etl.load():
        print("✅ ETL Process completed successfully!")
        
        # Run Analysis
        analyzer = SalesAnalyzer(etl.cleaned_data)
        analyzer.generate_summary_stats()
        analyzer.create_visualizations()
        
        # Generate Report
        reporter = SalesReporter(etl.cleaned_data)
        reporter.generate_report()
        
        print("\n🎉 Analysis completed! Check the 'output' folder for results.")
        print("📁 Generated files:")
        print("   - Cleaned data: output/cleaned_sales_data_*.csv")
        print("   - Visualizations: output/plots/sales_analysis_dashboard.png")
        print("   - Report: output/reports/sales_analysis_report_*.txt")
        
    else:
        print("❌ ETL Process failed!")

if __name__ == "__main__":
    main()
