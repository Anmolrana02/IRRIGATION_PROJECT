import pandas as pd
import numpy as np
import os
import joblib
from tensorflow.keras.models import load_model

# --- Constants ---
DATA_PATH = "data/FINAL_DATA_for_LSTM_Training.csv"
MODEL_PATH = "models/irrigation_lstm_model.keras"
SCALER_PATH = "models/data_scaler.gz"
SEQUENCE_LENGTH = 30 # Must be the same as in the training script

# Define Agricultural "Rulebooks" for the decision engine
SOIL_FC_percent, SOIL_PWP_percent = 0.25, 0.15
WHEAT_ROOT_mm, WHEAT_MAD = 900.0, 0.50
WHEAT_FC_mm = SOIL_FC_percent * WHEAT_ROOT_mm
WHEAT_PWP_mm = SOIL_PWP_percent * WHEAT_ROOT_mm
WHEAT_TRIGGER_mm = WHEAT_FC_mm - ((WHEAT_FC_mm - WHEAT_PWP_mm) * WHEAT_MAD)

RICE_PONDING_TARGET_mm = 50.0
RICE_PONDING_TRIGGER_mm = 20.0

CROP_PROFILES = {
    'Wheat': {'TRIGGER_mm': WHEAT_TRIGGER_mm, 'LOGIC': 'MAD'},
    'Rice': {'TRIGGER_mm': RICE_PONDING_TRIGGER_mm, 'LOGIC': 'Ponding'}
}


def prepare_prediction_data(last_30_days_df, scaler):
    """Prepares the last 30 days of data for prediction."""
    # This function must replicate the exact preprocessing from the training script
    
    # 1. One-Hot Encode
    df_encoded = pd.get_dummies(last_30_days_df, columns=['district', 'crop'], prefix=['dist', 'crop'])
    
    # 2. Re-create all possible columns to match the scaler's expectations
    # Get the feature names from the scaler
    scaler_feature_names = scaler.feature_names_in_
    
    for col in scaler_feature_names:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    # Ensure the column order is identical to the one used for training
    df_encoded = df_encoded[scaler_feature_names]

    # 3. Scale the data
    scaled_data = scaler.transform(df_encoded)
    
    # 4. Reshape for LSTM input (1 sample, 30 timesteps, N features)
    return np.array([scaled_data])


def inverse_transform_prediction(prediction_scaled, scaler):
    """Converts the scaled prediction back to the original soil moisture value in mm."""
    # The scaler expects a 2D array with the same number of features it was trained on
    # We create a dummy array, place our prediction in the correct column, and inverse transform
    num_features = scaler.n_features_in_
    dummy_row = np.zeros((1, num_features))
    
    # Find the index of our target variable
    target_col_index = list(scaler.feature_names_in_).index('real_soil_moisture_mm')
    
    dummy_row[0, target_col_index] = prediction_scaled
    
    # Inverse transform the entire dummy row
    unscaled_row = scaler.inverse_transform(dummy_row)
    
    # Extract our unscaled prediction
    return unscaled_row[0, target_col_index]


def run_prediction():
    """Loads the model and runs a sample prediction."""
    print("--- Step 3: Running Prediction ---")
    
    # --- 1. Load Model and Scaler ---
    if not all([os.path.exists(MODEL_PATH), os.path.exists(SCALER_PATH), os.path.exists(DATA_PATH)]):
        print("ERROR: Model, scaler, or data file not found.")
        print("Please run the data creation and training scripts first.")
        return

    print(" -> Loading saved model and scaler...")
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    # --- 2. Get Sample Data for Prediction ---
    # We will use the last 30 days of data from a specific district to simulate a real-world scenario
    print(" -> Getting sample data (last 30 days for Meerut)...")
    full_df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    
    # Get the last 30 days for a specific district
    sample_df = full_df[full_df['district'] == 'Meerut'].tail(SEQUENCE_LENGTH).copy()
    
    # What is the crop we are predicting for?
    # We assume the crop on the last day is the one we are forecasting for
    predicting_for_crop = sample_df.iloc[-1]['crop']
    print(f" -> Predicting for crop: {predicting_for_crop}")
    
    if predicting_for_crop not in CROP_PROFILES:
        print(f" -> No irrigation rules defined for '{predicting_for_crop}'. Cannot make a decision.")
        return
        
    # --- 3. Prepare Data and Predict ---
    print(" -> Preparing data for prediction...")
    input_sequence = prepare_prediction_data(sample_df, scaler)
    
    # Make the prediction
    print(" -> Making prediction with the LSTM model...")
    predicted_scaled_value = model.predict(input_sequence)[0][0]
    
    # Convert the prediction back to 'mm'
    predicted_moisture_mm = inverse_transform_prediction(predicted_scaled_value, scaler)
    
    # --- 4. Make the Final Decision ---
    print("\n--- Prediction Results ---")
    print(f"Model's Predicted Soil Moisture for Tomorrow: {predicted_moisture_mm:.2f} mm")
    
    # Get the correct "danger level" for the current crop
    irrigation_trigger = CROP_PROFILES[predicting_for_crop]['TRIGGER_mm']
    print(f"Irrigation Trigger Level for {predicting_for_crop}: {irrigation_trigger:.2f} mm")
    
    if predicted_moisture_mm < irrigation_trigger:
        print("\nDECISION: IRRIGATION IS REQUIRED.")
    else:
        print("\nDECISION: No irrigation is needed.")


if __name__ == "__main__":
    run_prediction()
