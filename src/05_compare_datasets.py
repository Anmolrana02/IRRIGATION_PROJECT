import pandas as pd
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

# --- File Paths ---
SIMULATED_DATA_PATH = "data/FINAL_DATA_for_LSTM_Training.csv"
REAL_DATA_PATH = "data/FINAL_DATA_with_REAL_Soil.csv"
OUTPUT_DIR = "reports"  # We'll save graphs here

def load_and_merge_data():
    """Loads both datasets and merges them for comparison."""
    print(f"Loading simulated data from: {SIMULATED_DATA_PATH}")
    df_sim = pd.read_csv(SIMULATED_DATA_PATH, parse_dates=['date'], usecols=['date', 'district', 'real_soil_moisture_mm'])
    df_sim.rename(columns={'real_soil_moisture_mm': 'simulated_moisture'}, inplace=True)
    
    print(f"Loading real data from: {REAL_DATA_PATH}")
    df_real = pd.read_csv(REAL_DATA_PATH, parse_dates=['date'], usecols=['date', 'district', 'real_soil_moisture_mm'])
    df_real.rename(columns={'real_soil_moisture_mm': 'real_moisture'}, inplace=True)
    
    # Merge the two dataframes on date and district
    print("Merging the two datasets...")
    df_merged = pd.merge(df_real, df_sim, on=['date', 'district'], how='inner')
    
    if df_merged.empty:
        print("ERROR: Could not merge datasets. Check if 'date' and 'district' columns match.")
        return None
        
    print(f"Successfully merged {len(df_merged)} common data points.")
    return df_merged

def print_statistics(df):
    """Prints a descriptive statistics comparison."""
    print("\n--- Statistical Comparison ---")
    print(df[['real_moisture', 'simulated_moisture']].describe())
    
    # Calculate correlation
    correlation = df['real_moisture'].corr(df['simulated_moisture'])
    print(f"\nCorrelation between real and simulated data: {correlation:.4f}")
    print("---------------------------------")

def plot_time_series(df, district):
    """Plots the two time series overlaid for a specific district."""
    print(f"Generating time series plot for '{district}'...")
    
    df_dist = df[df['district'] == district].copy()
    if df_dist.empty:
        print(f"No data found for district: {district}")
        return

    plt.figure(figsize=(15, 7))
    plt.plot(df_dist['date'], df_dist['real_moisture'], label='Real (Satellite) Moisture', color='blue', alpha=0.9)
    plt.plot(df_dist['date'], df_dist['simulated_moisture'], label='Simulated (Water-Balance) Moisture', color='orange', alpha=0.8)
    
    plt.title(f'Real vs. Simulated Soil Moisture Time Series ({district})')
    plt.xlabel('Date')
    plt.ylabel('Soil Moisture (mm)')
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, f'comparison_timeseries_{district}.png')
    plt.savefig(save_path)
    print(f" -> Time series plot saved to: {save_path}")

def plot_distributions(df):
    """Plots the histograms of both datasets."""
    print("Generating distribution plot (histogram)...")
    
    plt.figure(figsize=(12, 6))
    plt.hist(df['real_moisture'], bins=100, alpha=0.7, label='Real (Satellite) Moisture', color='blue')
    plt.hist(df['simulated_moisture'], bins=100, alpha=0.7, label='Simulated (Water-Balance) Moisture', color='orange')
    
    plt.title('Distribution of Soil Moisture Values')
    plt.xlabel('Soil Moisture (mm)')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Save the figure
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, 'comparison_distributions.png')
    plt.savefig(save_path)
    print(f" -> Distribution plot saved to: {save_path}")

def plot_scatter(df):
    """Plots a scatter plot to show correlation."""
    print("Generating scatter plot...")
    
    plt.figure(figsize=(8, 8))
    plt.scatter(df['simulated_moisture'], df['real_moisture'], alpha=0.3, label='Data Points')
    
    # Add a 1:1 line (perfect correlation)
    min_val = min(df['simulated_moisture'].min(), df['real_moisture'].min())
    max_val = max(df['simulated_moisture'].max(), df['real_moisture'].max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Match (1:1 Line)')
    
    plt.title('Real vs. Simulated Moisture Correlation')
    plt.xlabel('Simulated Moisture (mm)')
    plt.ylabel('Real (Satellite) Moisture (mm)')
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, 'comparison_scatter.png')
    plt.savefig(save_path)
    print(f" -> Scatter plot saved to: {save_path}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Compare Simulated vs. Real Soil Moisture data.")
    parser.add_argument(
        '--district', 
        type=str, 
        default='Meerut',
        help="The district to plot for the time series comparison (default: 'Meerut')."
    )
    args = parser.parse_args()

    # Load and merge
    merged_df = load_and_merge_data()
    
    if merged_df is not None:
        # 1. Print Statistics
        print_statistics(merged_df)
        
        # 2. Plot Time Series
        plot_time_series(merged_df, args.district)
        
        # 3. Plot Distributions
        plot_distributions(merged_df)
        
        # 4. Plot Scatter
        plot_scatter(merged_df)
        
        print("\n--- Comparison Complete! ---")
        print(f"All graphs have been saved to the '{OUTPUT_DIR}' folder.")