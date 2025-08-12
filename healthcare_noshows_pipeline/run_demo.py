"""
Healthcare Data Pipeline Demo Script
==================================

Simple demonstration of the healthcare appointments no-show data pipeline.
This script runs the complete pipeline with synthetic data and generates
all visualizations, models, and reports.
"""

import sys
import os
from datetime import datetime

# Add current directory to path to import the pipeline module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from healthcare_data_pipeline import HealthcareDataPipeline
print("Healthcare Data Pipeline Demo Script")
def run_demo():
    """
    Run the healthcare data pipeline demo
    """
    print("="*80)
    print("HEALTHCARE APPOINTMENTS NO-SHOW DATA PIPELINE DEMO")
    print("="*80)
    print(f"Demo started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Initialize the pipeline with output directory
        output_dir = "demo_output"
        pipeline = HealthcareDataPipeline(output_dir=output_dir)
        
        print(f"Pipeline initialized with output directory: {output_dir}")
        print()
        
        # Run the complete pipeline
        # This will use synthetic data based on the provided sample structure
        pipeline.run_complete_pipeline()
        
        print("\n" + "="*80)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("Generated files:")
        print(f"• Output directory: {output_dir}")
        print(f"• Plots: {output_dir}/plots/")
        print(f"• Reports: {output_dir}/reports/")
        print("• Cleaned data: CSV file in output directory")
        print()
        print("You can now explore the generated visualizations and reports!")
        print("="*80)
        
    except Exception as e:
        print(f"ERROR: Demo failed with error: {str(e)}")
        print("Please check the error message and ensure all dependencies are installed.")
        sys.exit(1)

if __name__ == "__main__":
    print("🚀 Starting Healthcare Data Pipeline Demo")
    run_demo()
