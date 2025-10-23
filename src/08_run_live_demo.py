import pandas as pd
import numpy as np
import os
import joblib
from tensorflow.keras.models import load_model

# --- Constants ---
MODEL_PATH = "models/real_model.keras"
SCALER_PATH = "models/real_scaler.gz"
DATA_PATH = "data/prediction_test.csv" # The new 40-row file you just made
SEQUENCE_LENGTH = 30 

# --- Agricultural Rules (Must match 03_prediction.py) ---
WHEAT_TRIGGER_mm = 225.0 # Assuming Wheat for this demo
CROP_PROFILES = {
    'Wheat': {'TRIGGER_mm': WHEAT_TRIGGER_mm, 'LOGIC': 'MAD'},
    'Rice': {'TRIGGER_mm': 20.0, 'LOGIC': 'Ponding'}
}

def prepare_prediction_data(last_30_days_df, scaler):
    """Prepares the last 30 days of data for prediction."""
    
    # We use a copy to avoid warnings
    df_processed = last_30_days_df.copy()
    
    # Drop the target variable if it exists (it shouldn't for the last row)
    if 'real_moisture' in df_processed.columns:
        df_processed = df_processed.drop(columns=['real_moisture'])
        
    # --- Re-create all preprocessing steps ---
    # 1. One-Hot Encode
    df_encoded = pd.get_dummies(df_processed, columns=['district', 'crop'], prefix=['dist', 'crop'])
    
    # 2. Re-create all possible columns to match the scaler
    scaler_feature_names = scaler.feature_names_in_
    
    for col in scaler_feature_names:
        if col not in df_encoded.columns and col != 'real_soil_moisture_mm':
            df_encoded[col] = 0
            
    # Add a dummy target column for scaling (it won't be used)
    if 'real_soil_moisture_mm' not in df_encoded.columns:
        df_encoded['real_soil_moisture_mm'] = 0.0

    # 3. Ensure column order is identical
    df_encoded = df_encoded[scaler_feature_names]

    # 4. Scale the data
    scaled_data = scaler.transform(df_encoded)
    
    # 5. Reshape for LSTM input (1 sample, 30 timesteps, N features)
    return np.array([scaled_data])


def inverse_transform_prediction(prediction_scaled, scaler):
    """Converts the scaled prediction back to the original soil moisture value in mm."""
    num_features = scaler.n_features_in_
    dummy_row = np.zeros((1, num_features))
    target_col_index = list(scaler.feature_names_in_).index('real_soil_moisture_mm')
    dummy_row[0, target_col_index] = prediction_scaled
    unscaled_row = scaler.inverse_transform(dummy_row)
    return unscaled_row[0, target_col_index]


if __name__ == "__main__":
    
    print("--- Running Live Prediction Demo ---")
    
    # --- 1. Load Model and Scaler ---
    print(f" -> Loading model: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    print(f" -> Loading scaler: {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)

    # --- 2. Load Sample Data ---
    print(f" -> Loading last 40 days from {DATA_PATH}...")
    full_df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    
    # Get the last 30 days of data (rows 10 to 40)
    # This simulates having 30 days of history to predict "tomorrow"
    input_df = full_df.tail(SEQUENCE_LENGTH).copy()
    
    # Get data for "today" (the 2nd to last day)
    today_date = input_df.iloc[-2]['date']
    today_actual_moisture = full_df[full_df['date'] == today_date]['real_soil_moisture_mm'].values[0]

    # Get data for "tomorrow" (the last day in the file)
    tomorrow_date = input_df.iloc[-1]['date']
    tomorrow_crop = input_df.iloc[-1]['crop']
    
    print(f" -> Simulating for: {tomorrow_date.strftime('%Y-%m-%d')}")
    print(f" -> Crop Type: {tomorrow_crop}")

    # --- 3. Prepare Data and Predict ---
    print(" -> Preparing data sequence...")
    input_sequence = prepare_prediction_data(input_df, scaler)
    
    print(" -> Making prediction with LSTM model...")
    predicted_scaled_value = model.predict(input_sequence)[0][0]
    
    predicted_moisture_mm = inverse_transform_prediction(predicted_scaled_value, scaler)
    
    # --- 4. Make the Final Decision ---
    trigger_level = CROP_PROFILES[tomorrow_crop]['TRIGGER_mm']
    
    decision = "NO IRRIGATION REQUIRED"
    if predicted_moisture_mm < trigger_level:
        decision = "IRRIGATION IS REQUIRED"

    print("\n" + "="*40)
    print("--- LIVE PREDICTION REPORT ---")
    print("="*40)
    print(f"Date of Prediction:  {tomorrow_date.strftime('%Y-%m-%d')}")
    print(f"Crop Type:           {tomorrow_crop}")
    print(f"\n--- Model Inputs ---")
    print(f"Moisture on {today_date.strftime('%Y-%m-%d')}: {today_actual_moisture:.2f} mm")
    print(f"Using 30-day history from {input_df.iloc[0]['date'].strftime('%Y-%m-%d')} to {today_date.strftime('%Y-%m-%d')}")
    print("\n--- Model Output ---")
    print(f"Predicted Moisture:  {predicted_moisture_mm:.2f} mm")
    print(f"Irrigation Trigger:  {trigger_level:.2f} mm")
    print("\n--- FINAL DECISION ---")
    print(f"     {decision}     ")
    print("="*40)
