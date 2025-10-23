import pandas as pd
import numpy as np
import os
import joblib
import argparse  # New import
from tensorflow.keras.models import load_model

# --- Constants ---
SEQUENCE_LENGTH = 30  # Must be the same as in the training script

# --- Data and Model Path Configuration ---
PATH_CONFIG = {
    'simulated': {
        'data': 'data/FINAL_DATA_for_LSTM_Training.csv',  # Used to get sample data
        'model': 'models/simulated_model.keras',
        'scaler': 'models/simulated_scaler.gz'
    },
    'real': {
        'data': 'data/FINAL_DATA_with_REAL_Soil.csv',  # Used to get sample data
        'model': 'models/real_model.keras',
        'scaler': 'models/real_scaler.gz'
    }
}

# Define Agricultural "Rulebooks" (remains the same)
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
    
    df_encoded = pd.get_dummies(last_30_days_df, columns=['district', 'crop'], prefix=['dist', 'crop'])
    
    scaler_feature_names = scaler.feature_names_in_
    
    for col in scaler_feature_names:
        if col not in df_encoded.columns:
            # This handles 'generated_irrigated' being in the simulated scaler
            # but not in the real data.
            df_encoded[col] = 0
            
    df_encoded = df_encoded[scaler_feature_names]

    scaled_data = scaler.transform(df_encoded)
    
    return np.array([scaled_data])


def inverse_transform_prediction(prediction_scaled, scaler):
    """Converts the scaled prediction back to the original soil moisture value in mm."""
    num_features = scaler.n_features_in_
    dummy_row = np.zeros((1, num_features))
    
    target_col_index = list(scaler.feature_names_in_).index('real_soil_moisture_mm')
    
    dummy_row[0, target_col_index] = prediction_scaled
    
    unscaled_row = scaler.inverse_transform(dummy_row)
    
    return unscaled_row[0, target_col_index]


def run_prediction(model_type):
    """Loads the model and runs a sample prediction."""
    print(f"--- Step 3: Running Prediction using '{model_type}' model ---")
    
    paths = PATH_CONFIG[model_type]
    DATA_PATH = paths['data']
    MODEL_PATH = paths['model']
    SCALER_PATH = paths['scaler']
    
    # --- 1. Load Model and Scaler ---
    if not all([os.path.exists(MODEL_PATH), os.path.exists(SCALER_PATH), os.path.exists(DATA_PATH)]):
        print(f"ERROR: Model, scaler, or data file not found for '{model_type}'.")
        print("Please run the data creation and training scripts first.")
        return

    print(f" -> Loading saved model ({MODEL_PATH}) and scaler ({SCALER_PATH})...")
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    # --- 2. Get Sample Data for Prediction ---
    print(f" -> Getting sample data from '{DATA_PATH}' (last 30 days for Meerut)...")
    full_df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    
    sample_df = full_df[full_df['district'] == 'Meerut'].tail(SEQUENCE_LENGTH).copy()
    
    if len(sample_df) < SEQUENCE_LENGTH:
        print(f"ERROR: Not enough data for prediction (found {len(sample_df)}, need {SEQUENCE_LENGTH}).")
        return

    predicting_for_crop = sample_df.iloc[-1]['crop']
    print(f" -> Predicting for crop: {predicting_for_crop}")
    
    if predicting_for_crop not in CROP_PROFILES:
        print(f" -> No irrigation rules defined for '{predicting_for_crop}'. Cannot make a decision.")
        return
        
    # --- 3. Prepare Data and Predict ---
    print(" -> Preparing data for prediction...")
    input_sequence = prepare_prediction_data(sample_df, scaler)
    
    print(" -> Making prediction with the LSTM model...")
    predicted_scaled_value = model.predict(input_sequence)[0][0]
    
    predicted_moisture_mm = inverse_transform_prediction(predicted_scaled_value, scaler)
    
    # --- 4. Make the Final Decision ---
    print("\n--- Prediction Results ---")
    print(f"Model's Predicted Soil Moisture for Tomorrow: {predicted_moisture_mm:.2f} mm")
    
    irrigation_trigger = CROP_PROFILES[predicting_for_crop]['TRIGGER_mm']
    print(f"Irrigation Trigger Level for {predicting_for_crop}: {irrigation_trigger:.2f} mm")
    
    if predicted_moisture_mm < irrigation_trigger:
        print("\nDECISION: IRRIGATION IS REQUIRED.")
    else:
        print("\nDECISION: No irrigation is needed.")


if __name__ == "__main__":
    
    # --- New: Use argparse to get user's choice ---
    parser = argparse.ArgumentParser(description="Run prediction using a specific model.")
    parser.add_argument(
        'model_type', 
        type=str, 
        choices=['simulated', 'real'], 
        help="The type of model to use for prediction ('simulated' or 'real')."
    )
    args = parser.parse_args()
    
    run_prediction(args.model_type)