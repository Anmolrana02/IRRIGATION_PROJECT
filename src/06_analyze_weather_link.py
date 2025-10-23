import pandas as pd
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns # New import for heatmaps

# --- File Paths ---
REAL_DATA_PATH = "data/FINAL_DATA_with_REAL_Soil.csv"
OUTPUT_DIR = "reports"  # We'll save graphs here

def load_data():
    """Loads the real dataset with all necessary weather columns."""
    print(f"Loading real data from: {REAL_DATA_PATH}")
    
    # Check if file exists
    if not os.path.exists(REAL_DATA_PATH):
        print(f"ERROR: Real data file not found at: {REAL_DATA_PATH}")
        print("Please run 'src/01_b_process_real_soil.py' first.")
        return None
        
    # We need all the key weather features for this analysis
    use_cols = [
        'date', 'district', 'real_soil_moisture_mm', 
        'precip_mm', 'ETo_mm', 'temperature_C'
    ]
    df = pd.read_csv(REAL_DATA_PATH, parse_dates=['date'], usecols=use_cols)
    
    # Rename for clarity in plots
    df.rename(columns={'real_soil_moisture_mm': 'real_moisture'}, inplace=True)
    
    print(f"Successfully loaded {len(df)} data points.")
    return df

def plot_precipitation_link(df, district):
    """
    Plots a dual-axis time series to show the direct link 
    between precipitation and soil moisture spikes.
    """
    print(f"Generating precipitation-link plot for '{district}'...")
    
    df_dist = df[df['district'] == district].copy()
    if df_dist.empty:
        print(f"No data found for district: {district}")
        return

    # To make the plot less crowded, let's just plot two years
    df_dist = df_dist[(df_dist['date'] >= '2019-01-01') & (df_dist['date'] <= '2020-12-31')]

    fig = plt.figure(figsize=(15, 7))
    ax1 = fig.add_subplot(111)
    
    # 1. Plot Soil Moisture on the left axis (Blue Line)
    ax1.plot(df_dist['date'], df_dist['real_moisture'], color='blue', label='Real Soil Moisture')
    ax1.set_ylabel('Soil Moisture (mm)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xlabel('Date')
    
    # 2. Create a second axis sharing the same x-axis
    ax2 = ax1.twinx()
    
    # 3. Plot Precipitation on the right axis (Green Bars)
    ax2.bar(df_dist['date'], df_dist['precip_mm'], color='green', alpha=0.5, label='Precipitation (Rain)')
    ax2.set_ylabel('Precipitation (mm)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    # Add legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.title(f'Soil Moisture vs. Precipitation (Rain) for {district}')
    
    # Save the figure
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, f'weatherlink_precipitation_{district}.png')
    plt.savefig(save_path)
    print(f" -> Plot saved to: {save_path}")

def plot_drying_link(df):
    """
    Plots a scatter plot showing how ETo (drying factor)
    affects the change in soil moisture on non-rainy days.
    """
    print("Generating drying (ETo) link plot...")
    
    # Calculate the day-to-day change in soil moisture
    # We must group by district so the .diff() doesn't cross between districts
    df['soil_delta'] = df.groupby('district')['real_moisture'].diff()
    
    # Filter for non-rainy days where the soil is actively drying
    df_drying = df[(df['precip_mm'] == 0) & (df['soil_delta'] < 0)].copy()
    
    if df_drying.empty:
        print("Could not find any drying days to plot.")
        return

    plt.figure(figsize=(10, 6))
    
    # Plot ETo vs. the change in soil moisture
    # Use low alpha to see density
    plt.scatter(df_drying['ETo_mm'], df_drying['soil_delta'], alpha=0.1)
    
    plt.title('Impact of Evaporation (ETo) on Soil Drying (on Non-Rainy Days)')
    plt.xlabel('ETo (Evaporation, mm)')
    plt.ylabel('Daily Soil Moisture Change (mm)')
    plt.grid(True)
    
    # Save the figure
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, 'weatherlink_drying_eto.png')
    plt.savefig(save_path)
    print(f" -> Plot saved to: {save_path}")

def plot_correlation_heatmap(df):
    """
    Plots a heatmap of the correlation matrix to show all
    relationships in one graph.
    """
    print("Generating correlation heatmap...")
    
    # Select only the numeric columns for correlation
    df_numeric = df[['real_moisture', 'precip_mm', 'ETo_mm', 'temperature_C']]
    corr = df_numeric.corr()
    
    plt.figure(figsize=(8, 6))
    
    # Use seaborn to create a heatmap
    # annot=True shows the numbers, cmap is the color scheme
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
    
    plt.title('Weather & Soil Moisture Correlation Matrix')
    
    # Save the figure
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, 'weatherlink_correlation_heatmap.png')
    plt.savefig(save_path, bbox_inches='tight')
    print(f" -> Plot saved to: {save_path}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Analyze the link between weather and real soil moisture.")
    parser.add_argument(
        '--district', 
        type=str, 
        default='Meerut',
        help="The district to plot for the time series analysis (default: 'Meerut')."
    )
    args = parser.parse_args()

    # Load data
    df = load_data()
    
    if df is not None:
        # 1. Plot Rain vs. Soil Moisture
        plot_precipitation_link(df, args.district)
        
        # 2. Plot Drying (ETo) vs. Soil Moisture
        plot_drying_link(df)
        
        # 3. Plot Correlation Heatmap
        plot_correlation_heatmap(df)
        
        print("\n--- Weather Link Analysis Complete! ---")
        print(f"All graphs have been saved to the '{OUTPUT_DIR}' folder.")