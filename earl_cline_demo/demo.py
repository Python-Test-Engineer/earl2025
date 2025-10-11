#!/usr/bin/env python3
"""
Simple Demo Script for Sales ETL and Analysis
============================================
This script demonstrates how to use the Sales ETL Analyzer
"""

import os
from sales_etl_analyzer import main

def run_demo():
    """Run the sales analysis demo"""
    print("🎯 SALES DATA ANALYSIS DEMO")
    print("=" * 40)
    print("This demo will:")
    print("1. 📥 Extract data from sales_data.csv")
    print("2. 🔄 Transform and clean the data")
    print("3. 💾 Load processed data to output files")
    print("4. 📊 Generate summary statistics")
    print("5. 🎨 Create visualizations")
    print("6. 📝 Generate analysis report")
    print()
    
    input("Press Enter to start the demo...")
    print()
    
    # Change to the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Run the main analysis
    main()
    
    print("\n🎉 Demo completed!")
    print("Check the 'output' folder for generated files.")

if __name__ == "__main__":
    run_demo()
