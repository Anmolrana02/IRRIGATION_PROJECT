import pandas as pd
import numpy as np
import os
import argparse
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Constants ---
SEQUENCE_LENGTH = 30  # Must be the same as in the training script

# --- Data and Model Path Configuration ---
PATH_CONFIG = {
    'simulated': {
        'data': 'data/FINAL_DATA_for_LSTM_Training.csv',
        'model': 'models/simulated_model.keras',
        'scaler': 'models/simulated_scaler.gz'
    },
    'real': {
        'data': 'data/FINAL_DATA_with_REAL_Soil.csv',
        'model': 'models/real_model.keras',
        'scaler': 'models/real_scaler.gz'
    }
}

def create_sequences(data, columns, sequence_length):
    """Creates sequences of data for the LSTM model."""
    print(" -> Creating test sequences...")
    X, y = [], []
    target_col_index = list(columns).index('real_soil_moisture_mm')
    
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length, target_col_index])
        
    return np.array(X), np.array(y)

def unscale_predictions(scaled_values_1d, scaler_obj, all_columns):
    """
    Converts a 1D array of scaled target predictions back to their
    original 'mm' values.
    """
    target_col_index = list(all_columns).index('real_soil_moisture_mm')
    dummy_array = np.zeros((len(scaled_values_1d), scaler_obj.n_features_in_))
    dummy_array[:, target_col_index] = scaled_values_1d.ravel()
    unscaled_array = scaler_obj.inverse_transform(dummy_array)
    return unscaled_array[:, target_col_index]


if __name__ == "__main__":
    
    # --- 1. Get User's Choice ---
    parser = argparse.ArgumentParser(description="Evaluate a trained LSTM model.")
    parser.add_argument(
        'model_type', 
        type=str, 
        choices=['simulated', 'real'], 
        help="The type of model to evaluate ('simulated' or 'real')."
    )
    args = parser.parse_args()
    
    paths = PATH_CONFIG[args.model_type]
    DATA_PATH = paths['data']
    MODEL_PATH = paths['model']
    SCALER_PATH = paths['scaler']
    
    print(f"--- Starting Evaluation for '{args.model_type}' model ---")

    # --- 2. Load Scaler and Data ---
    print(f" -> Loading scaler from '{SCALER_PATH}'...")
    if not all([os.path.exists(MODEL_PATH), os.path.exists(SCALER_PATH), os.path.exists(DATA_PATH)]):
        print("ERROR: Model, scaler, or data file not found. Run training first.")
        exit()
        
    scaler = joblib.load(SCALER_PATH)
    
    print(f" -> Loading data from '{DATA_PATH}'...")
    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    
    # --- 3. Pre-process Data (exactly as in training) ---
    print(" -> Pre-processing data...")
    df = pd.get_dummies(df, columns=['district', 'crop'], prefix=['dist', 'crop'])
    df.set_index('date', inplace=True)
    
    # Ensure all columns the scaler expects are present
    scaler_feature_names = scaler.feature_names_in_
    for col in scaler_feature_names:
        if col not in df.columns:
            df[col] = 0
    
    # Ensure column order is identical to the one used for training
    df = df[scaler_feature_names]
    columns = df.columns
    
    # Transform data (do NOT fit)
    scaled_data = scaler.transform(df)

    # --- 4. Create and Split Sequences (exactly as in training) ---
    X, y = create_sequences(scaled_data, columns, SEQUENCE_LENGTH)
    
    # Re-create the *exact* same test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    print(f" -> Loaded {X_test.shape[0]} test sequences.")

    # --- 5. Load Model and Predict ---
    print(f" -> Loading model from '{MODEL_PATH}'...")
    model = load_model(MODEL_PATH)
    
    print(" -> Making predictions on test set...")
    y_pred_scaled = model.predict(X_test)
    
    # --- 6. Un-scale and Calculate Metrics ---
    print(" -> Un-scaling predictions to real 'mm' values...")
    y_test_real = unscale_predictions(y_test, scaler, columns)
    y_pred_real = unscale_predictions(y_pred_scaled, scaler, columns)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    mae = mean_absolute_error(y_test_real, y_pred_real)
    r2 = r2_score(y_test_real, y_pred_real)
    
    print("\n--- FINAL MODEL PERFORMANCE (on real 'mm' values) ---")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f} mm")
    print(f"  Mean Absolute Error (MAE):    {mae:.4f} mm")
    print(f"  R-squared ($R^2$):               {r2:.4f}")
    print("---------------------------------------------------------")