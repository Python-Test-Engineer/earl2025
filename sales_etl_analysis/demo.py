#!/usr/bin/env python3
"""
Demo script for Sales ETL and Data Analysis Program
==================================================

This script demonstrates how to use the SalesETLAnalyzer class
with different usage patterns.
"""

from sales_etl_analyzer import SalesETLAnalyzer
import os

def demo_full_pipeline():
    """Demonstrate the full ETL pipeline."""
    print("🎯 Demo: Full ETL Pipeline")
    print("="*50)
    
    analyzer = SalesETLAnalyzer()
    
    # Check if sample data exists
    if os.path.exists("sample_sales_data.csv"):
        analyzer.run_full_etl_analysis("sample_sales_data.csv")
        analyzer.display_summary()
    else:
        print("❌ Sample data not found!")

def demo_step_by_step():
    """Demonstrate step-by-step ETL process."""
    print("\n🎯 Demo: Step-by-Step ETL Process")
    print("="*50)
    
    analyzer = SalesETLAnalyzer()
    
    if os.path.exists("sample_sales_data.csv"):
        # Step 1: Extract
        print("\n1️⃣ EXTRACT")
        analyzer.extract_data("sample_sales_data.csv")
        
        # Step 2: Transform
        print("\n2️⃣ TRANSFORM")
        analyzer.transform_data()
        
        # Step 3: Load
        print("\n3️⃣ LOAD")
        analyzer.load_data()
        
        # Step 4: Analyze
        print("\n4️⃣ ANALYZE")
        results = analyzer.analyze_data()
        
        # Display some results
        print("\n📊 Sample Analysis Results:")
        if 'basic_stats' in results:
            stats = results['basic_stats']
            print(f"   💰 Total Revenue: ${stats['total_revenue']:,.2f}")
            print(f"   📦 Total Quantity: {stats['total_quantity_sold']:,}")
        
        # Step 5: Visualize
        print("\n5️⃣ VISUALIZE")
        analyzer.create_visualizations()
        
        # Step 6: Report
        print("\n6️⃣ REPORT")
        analyzer.generate_report()
        
    else:
        print("❌ Sample data not found!")

def demo_custom_analysis():
    """Demonstrate custom analysis capabilities."""
    print("\n🎯 Demo: Custom Analysis")
    print("="*50)
    
    analyzer = SalesETLAnalyzer()
    
    if os.path.exists("sample_sales_data.csv"):
        # Load and transform data
        analyzer.extract_data("sample_sales_data.csv")
        analyzer.transform_data()
        
        # Access the cleaned data directly
        data = analyzer.clean_data
        
        print(f"\n📋 Data Overview:")
        print(f"   📊 Shape: {data.shape}")
        print(f"   📅 Date Range: {data['date'].min()} to {data['date'].max()}")
        print(f"   🏷️ Categories: {data['category'].unique().tolist()}")
        print(f"   🌍 Regions: {data['region'].unique().tolist()}")
        
        # Custom analysis examples
        print(f"\n🔍 Custom Analysis:")
        
        # Top 3 products by revenue
        top_products = data.groupby('product_name')['total_sales'].sum().sort_values(ascending=False).head(3)
        print(f"   🏆 Top 3 Products:")
        for i, (product, sales) in enumerate(top_products.items(), 1):
            print(f"      {i}. {product}: ${sales:,.2f}")
        
        # Average order value by region
        avg_by_region = data.groupby('region')['total_sales'].mean().sort_values(ascending=False)
        print(f"   🌍 Average Order Value by Region:")
        for region, avg in avg_by_region.items():
            print(f"      {region}: ${avg:,.2f}")
        
        # Sales rep performance
        rep_performance = data.groupby('sales_rep')['total_sales'].agg(['sum', 'count']).round(2)
        rep_performance.columns = ['Total Sales', 'Number of Sales']
        print(f"   👥 Sales Rep Performance:")
        print(rep_performance.to_string())
        
    else:
        print("❌ Sample data not found!")

def main():
    """Run all demos."""
    print("🚀 Sales ETL and Data Analysis - Demo Suite")
    print("="*60)
    
    # Demo 1: Full pipeline
    demo_full_pipeline()
    
    # Demo 2: Step-by-step
    demo_step_by_step()
    
    # Demo 3: Custom analysis
    demo_custom_analysis()
    
    print("\n✅ All demos completed!")
    print("\n📁 Check the 'output' directory for generated files:")
    print("   📊 Plots: output/plots/")
    print("   📋 Reports: output/reports/")
    print("   💾 Data: output/")

if __name__ == "__main__":
    main()
