import pandas as pd
import numpy as np
import requests
import os

# --- Constants for Data Fetching ---
# Define the districts and their central coordinates (Lat, Lon)
DISTRICTS = {
    "Baghpat": (28.94, 77.22),
    "Shamli": (29.45, 77.31),
    "Meerut": (28.98, 77.70),
    "Muzaffarnagar": (29.47, 77.68),
    "Ghaziabad": (28.66, 77.42)
}
START_DATE = "2015-01-01"
END_DATE = "2024-12-31"
NASA_POWER_API = "https://power.larc.nasa.gov/api/temporal/daily/point" # Using point endpoint
RAW_DATA_PATH = "data/raw_nasa_weather_data_districts.csv"
FINAL_DATA_PATH = "data/FINAL_DATA_for_LSTM_Training.csv"

# --- PART 1: DATA FETCHING ---
def fetch_weather_data_for_districts(districts, start, end):
    """Fetches daily weather data for a list of districts from the NASA POWER API."""
    print("Step 1.1: Fetching weather data for all districts...")
    all_district_dfs = []

    for district_name, (lat, lon) in districts.items():
        print(f" -> Fetching data for {district_name} (Lat: {lat}, Lon: {lon})...")
        params = {
            "parameters": "T2M,PRECTOTCORR,WS2M,ALLSKY_SFC_SW_DWN",
            "community": "AG",
            "longitude": lon,
            "latitude": lat,
            "start": start.replace("-", ""),
            "end": end.replace("-", ""),
            "format": "JSON"
        }
        try:
            response = requests.get(NASA_POWER_API, params=params)
            response.raise_for_status() # Will raise an error for bad status codes (4xx or 5xx)
            data = response.json()["properties"]["parameter"]
            
            # --- ROBUSTNESS CHECK ---
            # Check if the T2M data is a valid dictionary before proceeding
            if not isinstance(data.get("T2M"), dict) or not data["T2M"]:
                print(f"    -> WARNING: No valid 'T2M' (temperature) data found for {district_name}. Skipping this district.")
                continue

            # Explicitly convert keys and values to lists for stability
            dates = list(data["T2M"].keys())
            temps = list(data["T2M"].values())
            precip = list(data["PRECTOTCORR"].values())
            wind = list(data["WS2M"].values())
            solar = list(data["ALLSKY_SFC_SW_DWN"].values())

            df = pd.DataFrame({
                "date": pd.to_datetime(dates, format='%Y%m%d'),
                "temperature_C": temps,
                "precip_mm": precip,
                "wind_m_s": wind,
                "solar_rad_MJ_m2": solar,
            })
            
            df['district'] = district_name # Add the new district column
            df['latitude'] = lat # Store latitude for ETo calculation
            all_district_dfs.append(df)

        except requests.exceptions.RequestException as e:
            print(f"    -> ERROR: Failed to fetch data for {district_name}: {e}")
            continue
        except KeyError:
            print(f"    -> ERROR: Received unexpected JSON format for {district_name}.")
            continue
    
    if not all_district_dfs:
        print(" -> ERROR: No data was fetched for any district. Halting.")
        return None

    # Combine all dataframes into one
    combined_df = pd.concat(all_district_dfs, ignore_index=True)
    # Replace the API's -999 placeholder for missing data with NaN and interpolate
    combined_df.replace(-999.0, np.nan, inplace=True)
    combined_df.interpolate(method='linear', inplace=True)

    combined_df.to_csv(RAW_DATA_PATH, index=False)
    print(f"\n -> Successfully fetched and saved raw weather data for all districts to '{RAW_DATA_PATH}'")
    return combined_df

# --- PART 2: DATA SIMULATION (This part remains the same) ---

def assign_crops(df):
    """Assigns a crop to each day based on the month (Kharif/Rabi cycle)."""
    month = df["date"].dt.month
    df["crop"] = np.select(
        [(month >= 6) & (month <= 10), (month >= 11) | (month <= 4)],
        ["Rice", "Wheat"],
        default="None"
    )
    return df

def calculate_eto_manual(df):
    """Calculates Reference Evapotranspiration (ETo) manually for each row using its latitude."""
    print("Step 1.2: Calculating Evapotranspiration (ETo_mm)...")
    try:
        df['J'] = df['date'].dt.dayofyear
        phi = np.deg2rad(df['latitude'])
        delta = 0.409 * np.sin(((2 * np.pi / 365) * df['J']) - 1.39)
        ws_arg = -np.tan(phi) * np.tan(delta)
        ws = np.arccos(np.clip(ws_arg, -1.0, 1.0))
        Gsc = 0.0820
        dr = 1 + 0.033 * np.cos((2 * np.pi / 365) * df['J'])
        Ra_MJ_m2_day = ((24 * 60) / np.pi) * Gsc * dr * ((ws * np.sin(phi) * np.sin(delta)) + (np.cos(phi) * np.cos(delta) * np.sin(ws)))
        
        Ra_MJ_m2_day[Ra_MJ_m2_day <= 0] = 0.0001
        
        Rs_over_Ra = np.clip(df['solar_rad_MJ_m2'] / Ra_MJ_m2_day, 0.3, 1.0)
        T_diff_est = 8.0 + (16.0 - 8.0) * Rs_over_Ra
        
        T_avg = df['temperature_C']
        Ra_mm_day = Ra_MJ_m2_day * 0.408
        
        ETo_mm_day = 0.0023 * (T_avg + 17.8) * np.sqrt(T_diff_est) * Ra_mm_day
        df['ETo_mm'] = ETo_mm_day
        
        df['ETo_mm'] = df['ETo_mm'].fillna(0).apply(lambda x: max(0, x))
        print(" -> Successfully calculated ETo for all districts.")
    except Exception as e:
        print(f" -> Error in ETo calculation: {e}")
        df['ETo_mm'] = 0.0
    return df

def run_multi_crop_simulation(df, crop_profiles):
    """Runs a daily soil water balance simulation, now grouped by district."""
    print("Step 1.3: Running multi-crop soil and irrigation simulation for each district...")
    all_district_sims = []
    
    for district in df['district'].unique():
        print(f" -> Simulating for {district}...")
        district_df = df[df['district'] == district].copy()
        
        mad_logic_state = {'Wheat': crop_profiles['Wheat']['FC_mm']}
        ponding_logic_state = {'Rice': crop_profiles['Rice']['PONDING_TARGET_mm']}
        
        simulated_moisture_list, generated_irrigated_list = [], []
            
        for index, row in district_df.iterrows():
            crop_type = row['crop']
            irrigation_applied = 0.0
            current_moisture_value = 0.0
            
            if crop_type == 'Wheat':
                profile = crop_profiles['Wheat']
                current_moisture = mad_logic_state['Wheat']
                new_moisture = current_moisture + row['precip_mm']
                
                is_growing_season = row['date'].month in profile['GROWING_SEASON']
                if (new_moisture < profile['TRIGGER_mm']) and is_growing_season:
                    irrigation_applied = 1.0
                    new_moisture += profile['IRR_AMOUNT_mm']
                
                new_moisture -= row['ETo_mm']
                
                if new_moisture > profile['FC_mm']: new_moisture = profile['FC_mm']
                if new_moisture < profile['PWP_mm']: new_moisture = profile['PWP_mm']
                    
                mad_logic_state['Wheat'] = new_moisture
                current_moisture_value = new_moisture

            elif crop_type == 'Rice':
                profile = crop_profiles['Rice']
                current_ponding_level = ponding_logic_state['Rice']
                new_ponding_level = current_ponding_level + row['precip_mm'] - row['ETo_mm'] - profile['PERCOLATION_mm_day']
                
                is_growing_season = row['date'].month in profile['GROWING_SEASON']
                if (new_ponding_level < profile['PONDING_TRIGGER_mm']) and is_growing_season:
                    irrigation_applied = 1.0
                    new_ponding_level = profile['PONDING_TARGET_mm']

                if new_ponding_level < 0: new_ponding_level = 0
                
                ponding_logic_state['Rice'] = new_ponding_level
                current_moisture_value = new_ponding_level
            
            simulated_moisture_list.append(current_moisture_value)
            generated_irrigated_list.append(irrigation_applied)
            
        district_df['real_soil_moisture_mm'] = simulated_moisture_list
        district_df['generated_irrigated'] = generated_irrigated_list
        all_district_sims.append(district_df)
        
    final_df = pd.concat(all_district_sims, ignore_index=True)
    print(" -> All district simulations complete.")
    return final_df

# --- Main execution ---
if __name__ == "__main__":
    
    if os.path.exists(FINAL_DATA_PATH):
        print(f"'{FINAL_DATA_PATH}' already exists. Skipping data creation.")
        print("Delete the file to re-run.")
    else:
        SOIL_FC_percent, SOIL_PWP_percent = 0.25, 0.15
        WHEAT_ROOT_mm, WHEAT_MAD = 900.0, 0.50
        WHEAT_FC_mm = SOIL_FC_percent * WHEAT_ROOT_mm
        WHEAT_PWP_mm = SOIL_PWP_percent * WHEAT_ROOT_mm
        WHEAT_TRIGGER_mm = WHEAT_FC_mm - ((WHEAT_FC_mm - WHEAT_PWP_mm) * WHEAT_MAD)
        
        CROP_PROFILES = {
            'Wheat': {'GROWING_SEASON': [10, 11, 12, 1, 2, 3, 4], 'FC_mm': WHEAT_FC_mm, 'PWP_mm': WHEAT_PWP_mm, 'TRIGGER_mm': WHEAT_TRIGGER_mm, 'IRR_AMOUNT_mm': 70.0},
            'Rice': {'GROWING_SEASON': [6, 7, 8, 9, 10, 11], 'PONDING_TARGET_mm': 50.0, 'PONDING_TRIGGER_mm': 20.0, 'PERCOLATION_mm_day': 5.0}
        }
        
        df_weather = fetch_weather_data_for_districts(DISTRICTS, START_DATE, END_DATE)
        
        if df_weather is not None:
            df_with_crops = assign_crops(df_weather)
            df_with_eto = calculate_eto_manual(df_with_crops)
            df_final = run_multi_crop_simulation(df_with_eto, CROP_PROFILES)
            
            columns_to_save = ['date', 'district', 'crop', 'temperature_C', 'precip_mm', 'wind_m_s', 'solar_rad_MJ_m2', 'ETo_mm', 'generated_irrigated', 'real_soil_moisture_mm']
            df_final[columns_to_save].to_csv(FINAL_DATA_PATH, index=False)
            print(f"\nSUCCESS: Your final, clean dataset has been saved as '{FINAL_DATA_PATH}'")

