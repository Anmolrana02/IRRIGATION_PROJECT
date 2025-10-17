import cdsapi
import os

c = cdsapi.Client()

# --- Configuration ---
years_to_download = [
    '2015', '2016', '2017', '2018', '2019', 
    '2020', '2021', '2022', '2023', '2024'
]
months_to_download = [
    '01', '02', '03', '04', '05', '06', 
    '07', '08', '09', '10', '11', '12'
]
output_directory = "data/soil_moisture_data"
# ---------------------

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

print(f"Starting download. Files will be saved in '{output_directory}'")

# --- NESTED LOOP: Loop through each year, then each month ---
for year in years_to_download:
    for month in months_to_download:
        
        # This will be the name of the file, e.g., "soil_moisture_data/soil_moisture_2015-01.nc"
        output_filename = os.path.join(output_directory, f"soil_moisture_{year}-{month}.nc")
        
        # Check if file already exists to avoid re-downloading
        if os.path.exists(output_filename):
            print(f"File {output_filename} already exists. Skipping.")
            continue

        print(f"--- Requesting data for: {year}-{month} ---")
        
        try:
            c.retrieve(
                'reanalysis-era5-land',
                {
                    'variable': 'volumetric_soil_water_layer_1',
                    'year': year,     # Requesting ONE year
                    'month': month,   # Requesting ONE month
                    'day': [
                        '01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
                        '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                        '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31'
                    ],
                    'time': [ 
                        '00:00', '01:00', '02:00', '03:00', '04:00', '05:00',
                        '06:00', '07:00', '08:00', '09:00', '10:00', '11:00',
                        '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
                        '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'
                    ],
                    'area': [30, 77, 26, 80],  # Bounding box for Western UP [N, W, S, E]
                    'format': 'netcdf',
                },
                output_filename
            )
            print(f"Successfully downloaded: {output_filename}")

        except Exception as e:
            print(f"Error downloading data for {year}-{month}: {e}")
            # If a request fails, it will print the error and continue to the next month
            pass 

print("--- All downloads complete. ---")