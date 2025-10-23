import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- File Paths ---
REAL_DATA_PATH = "data/FINAL_DATA_with_REAL_Soil.csv"
OUTPUT_DIR = "reports"  # We'll save graphs here

def load_data():
    """Loads the real dataset with all necessary columns."""
    print(f"Loading real data from: {REAL_DATA_PATH}")
    
    if not os.path.exists(REAL_DATA_PATH):
        print(f"ERROR: Real data file not found at: {REAL_DATA_PATH}")
        return None
        
    df = pd.read_csv(REAL_DATA_PATH, parse_dates=['date'])
    df.rename(columns={'real_soil_moisture_mm': 'real_moisture'}, inplace=True)
    
    print(f"Successfully loaded {len(df)} data points.")
    return df

def plot_full_correlation_heatmap(df):
    """
    Plots a heatmap of the correlation matrix for ALL numeric variables
    to show all relationships in one graph.
    """
    print("Generating FULL correlation heatmap...")
    
    # Select ALL numeric columns for correlation
    df_numeric = df[[
        'real_moisture', 'precip_mm', 'ETo_mm', 
        'temperature_C', 'wind_m_s', 'solar_rad_MJ_m2'
    ]]
    
    # Rename for cleaner plot labels
    df_numeric.rename(columns={
        'real_moisture': 'Soil Moisture',
        'precip_mm': 'Precipitation',
        'ETo_mm': 'Evaporation (ETo)',
        'temperature_C': 'Temperature',
        'wind_m_s': 'Wind Speed',
        'solar_rad_MJ_m2': 'Solar Radiation'
    }, inplace=True)
    
    corr = df_numeric.corr()
    
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
    
    plt.title('Full Weather & Soil Moisture Correlation Matrix')
    
    # Save the figure
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, 'analysis_full_correlation_heatmap.png')
    plt.savefig(save_path, bbox_inches='tight')
    print(f" -> Plot saved to: {save_path}")

def plot_seasonal_cycle(df):
    """
    Plots the average soil moisture by month to show the clear
    seasonal (monsoon) cycle.
    """
    print("Generating seasonal cycle plot...")
    
    # Extract month number and name
    df['month'] = df['date'].dt.month
    df['month_name'] = df['date'].dt.strftime('%b') # e.g., 'Jan', 'Feb'
    
    # Group by month and get the average moisture
    df_monthly_avg = df.groupby('month')[['real_moisture', 'month_name']].first().reset_index()
    
    # Get average moisture for the whole month
    monthly_mean = df.groupby('month')['real_moisture'].mean()
    
    plt.figure(figsize=(12, 6))
    
    # Create a bar plot
    plt.bar(df_monthly_avg['month_name'], monthly_mean, color='blue')
    
    plt.title('Average Soil Moisture by Month (Shows Monsoon Cycle)')
    plt.xlabel('Month')
    plt.ylabel('Average Soil Moisture (mm)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the figure
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, 'analysis_seasonal_cycle.png')
    plt.savefig(save_path)
    print(f" -> Plot saved to: {save_path}")

def plot_crop_distribution(df):
    """
    Plots a box plot to show the distribution of soil moisture
    during the assigned 'Wheat', 'Rice', and 'None' seasons.
    """
    print("Generating moisture distribution by crop season...")
    
    plt.figure(figsize=(10, 7))
    
    # Create a box plot
    sns.boxplot(
        x='crop', 
        y='real_moisture', 
        data=df, 
        order=['Rice', 'Wheat', 'None'] # Re-order for logical flow
    )
    
    plt.title('Real Soil Moisture Distribution by Crop Season')
    plt.xlabel('Crop Season (as defined in simulation)')
    plt.ylabel('Real Soil Moisture (mm)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the figure
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, 'analysis_crop_distribution.png')
    plt.savefig(save_path)
    print(f" -> Plot saved to: {save_path}")


if __name__ == "__main__":

    # Load data
    df = load_data()
    
    if df is not None:
        # 1. Plot Full Correlation Heatmap
        plot_full_correlation_heatmap(df)
        
        # 2. Plot Seasonal (Monsoon) Cycle
        plot_seasonal_cycle(df)
        
        # 3. Plot Distribution by Crop
        plot_crop_distribution(df)
        
        print("\n--- Advanced Analysis Complete! ---")
        print(f"All graphs have been saved to the '{OUTPUT_DIR}' folder.")