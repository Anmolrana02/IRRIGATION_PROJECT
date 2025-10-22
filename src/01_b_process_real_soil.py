import pandas as pd
import numpy as np
import os
import glob
import xarray as xr  # You may need to run: pip install xarray netcdf4
import zipfile
# --- Constants ---
# These must match the districts from your 01_fetch file
DISTRICTS = {
    "Baghpat": (28.94, 77.22),
    "Shamli": (29.45, 77.31),
    "Meerut": (28.98, 77.70),
    "Muzaffarnagar": (29.47, 77.68),
    "Ghaziabad": (28.66, 77.42)
}

# --- File Paths ---
RAW_WEATHER_INPUT_PATH = "data/raw_nasa_weather_data_districts.csv"
SOIL_DATA_DIR = "data/soil_moisture_data" # Path to your .nc files
FINAL_OUTPUT_PATH = "data/FINAL_DATA_with_REAL_Soil.csv" # New output file

# --- Crop/Soil Physics Constants ---
# We still need these for ETo calculation and unit conversion
WHEAT_ROOT_mm = 900.0
RICE_ROOT_mm = 300.0 # Typical puddled soil depth
CROP_ROOT_DEPTHS = {
    'Wheat': WHEAT_ROOT_mm,
    'Rice': RICE_ROOT_mm, 
    'None': WHEAT_ROOT_mm # Fallback
}


# --- PART 1: REAL SOIL DATA PROCESSING ---

def load_and_merge_real_soil_data(weather_df):
    """Loads, processes, and merges real .nc soil moisture data with weather data."""
    print("Step 1.2: Loading and processing real NetCDF soil moisture data...")
    
    # 1. Find all .nc files
    nc_files_raw = glob.glob(os.path.join(SOIL_DATA_DIR, "*.nc"))
    if not nc_files_raw:
        print(f"ERROR: No .nc files found in '{SOIL_DATA_DIR}'.")
        print("Please run 'soil_moisture_data.py' first.")
        return None

    # 2. --- NEW 'UNZIP-IN-PLACE' LOGIC ---
    print(" -> Checking for and extracting zipped .nc files...")
    
    for f_path in nc_files_raw:
        if not zipfile.is_zipfile(f_path):
            continue  # It's already a valid .nc file, skip it

        print(f"    -> Found zipped file: {os.path.basename(f_path)}. Unzipping in-place...")
        try:
            # Read the zip file into memory
            with zipfile.ZipFile(f_path, 'r') as zip_ref:
                
                # Find the .nc file *inside* the zip (e.g., 'data_0.nc')
                nc_file_name_inside = None
                for name in zip_ref.namelist():
                    if name.endswith('.nc'):
                        nc_file_name_inside = name
                        break
                
                if nc_file_name_inside is None:
                    print(f"    -> WARNING: Could not find a .nc file inside {f_path}. Skipping.")
                    continue
                
                # Extract the .nc file's content into memory
                nc_data = zip_ref.read(nc_file_name_inside)
            
            # Now, overwrite the original .zip file with the extracted .nc data
            with open(f_path, 'wb') as f_out:
                f_out.write(nc_data)
                
        except Exception as e:
            print(f"    -> ERROR processing {f_path}: {e}")
    
    print(" -> Unzipping complete.")
    # -----------------------------------------------------

    # 3. Open all *unzipped* files as a single xarray Dataset
    #    We can use nc_files_raw because all paths now point to valid .nc files
    print(f" -> Loading {len(nc_files_raw)} NetCDF files...")
    try:
        ds = xr.open_mfdataset(
            nc_files_raw, 
            combine='by_coords',  # This tells xarray to use coords like 'time'
            engine='netcdf4'
        )
    except Exception as e:
        print(f" -> ERROR loading NetCDF files: {e}")
        return None
        
    # 4. Rename variable for clarity
    # New fixed line
    ds = ds.rename({'swvl1': 'sm_volumetric'})
    
    # 5. Aggregate from hourly to daily mean
    print(" -> Aggregating hourly data to daily mean...")
    ds_daily = ds.resample(valid_time='1D').mean()
    
    # 6. Loop through districts, extract data, and merge
    print(" -> Extracting and merging data for each district...")
    all_merged_dfs = []
    
    for district_name, (lat, lon) in DISTRICTS.items():
        print(f"    -> Processing {district_name}...")
        
        current_weather_df = weather_df[weather_df['district'] == district_name].copy()
        if current_weather_df.empty:
            continue
            
        sm_series = ds_daily['sm_volumetric'].sel(
            latitude=lat, 
            longitude=lon, 
            method='nearest'
        )
        
        sm_df = sm_series.to_dataframe().reset_index()
        sm_df.rename(columns={'valid_time': 'date', 'sm_volumetric': 'real_soil_moisture_m3_m3'}, inplace=True)
        
        current_weather_df['date'] = pd.to_datetime(current_weather_df['date'])
        sm_df['date'] = pd.to_datetime(sm_df['date'])

        # 7. Merge weather and soil data
        merged_df = pd.merge(
            current_weather_df, 
            sm_df[['date', 'real_soil_moisture_m3_m3']], 
            on='date', 
            how='inner'
        )
        all_merged_dfs.append(merged_df)
        
    if not all_merged_dfs:
        print(" -> ERROR: No data could be merged.")
        return None

    final_df = pd.concat(all_merged_dfs, ignore_index=True)
    print(" -> Successfully merged all district weather and soil moisture data.")
    return final_df

# ... (The rest of your file, assign_crops etc., remains unchanged) ...

# ... (Rest of your file `assign_crops`, `calculate_eto_manual`, etc. stays the same) ...

# --- PART 2: FEATURE ENGINEERING (Copied from 01_...py) ---

def assign_crops(df):
    """Assigns a crop to each day based on the month (Kharif/Rabi cycle)."""
    print("\nStep 1.3: Assigning crop cycles...")
    month = df["date"].dt.month
    df["crop"] = np.select(
        [(month >= 6) & (month <= 10), (month >= 11) | (month <= 4)],
        ["Rice", "Wheat"],
        default="None"
    )
    return df

def calculate_eto_manual(df):
    """Calculates Reference Evapotranspiration (ETo) manually for each row."""
    print("\nStep 1.4: Calculating Evapotranspiration (ETo_mm)...")
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
        print(" -> Successfully calculated ETo.")
    except Exception as e:
        print(f" -> Error in ETo calculation: {e}")
        df['ETo_mm'] = 0.0
    return df

def convert_volumetric_to_mm(df, root_depths):
    """Converts volumetric (m3/m3) soil moisture to depth (mm)."""
    print("\nStep 1.5: Converting soil moisture from m3/m3 to mm...")
    
    df['root_depth_mm'] = df['crop'].map(root_depths)
    fallback_depth = root_depths.get('Wheat', 900.0)
    df['root_depth_mm'].fillna(fallback_depth, inplace=True)

    # The conversion: (m3/m3) * depth_in_mm = mm
    df['real_soil_moisture_mm'] = df['real_soil_moisture_m3_m3'] * df['root_depth_mm']
    print(" -> Conversion complete.")
    return df

# --- Main execution ---
if __name__ == "__main__":
    
    if os.path.exists(FINAL_OUTPUT_PATH):
        print(f"'{FINAL_OUTPUT_PATH}' already exists. Skipping data creation.")
        print("Delete the file to re-run.")
    else:
        # STEP 1: Load raw weather data
        if not os.path.exists(RAW_WEATHER_INPUT_PATH):
            print(f"ERROR: Raw weather file not found at '{RAW_WEATHER_INPUT_PATH}'")
            print("Please run '01_fetch_and_simulate.py' first to generate this file.")
        else:
            print(f"Loading raw weather data from '{RAW_WEATHER_INPUT_PATH}'...")
            df_weather = pd.read_csv(RAW_WEATHER_INPUT_PATH, parse_dates=['date'])
            
            # STEP 2: Load and merge real soil data
            df_merged = load_and_merge_real_soil_data(df_weather)
            
            if df_merged is not None:
                # STEP 3: Assign crops
                df_with_crops = assign_crops(df_merged)
                
                # STEP 4: Calculate ETo
                df_with_eto = calculate_eto_manual(df_with_crops)
                
                # STEP 5: Convert m3/m3 to mm
                df_final = convert_volumetric_to_mm(df_with_eto, CROP_ROOT_DEPTHS)

                # Define final columns
                # We drop the simulation column 'generated_irrigated'
                # and the intermediate 'real_soil_moisture_m3_m3'
                columns_to_save = [
                    'date', 'district', 'crop', 'temperature_C', 'precip_mm', 
                    'wind_m_s', 'solar_rad_MJ_m2', 'ETo_mm', 
                    'real_soil_moisture_mm' # This is now the REAL data in mm
                ]
                
                # Drop any rows with NaN values that might have been created
                final_df_cleaned = df_final[columns_to_save].dropna()
                
                final_df_cleaned.to_csv(FINAL_OUTPUT_PATH, index=False)
                print(f"\nSUCCESS: Your final dataset with REAL soil moisture has been saved as '{FINAL_OUTPUT_PATH}'")