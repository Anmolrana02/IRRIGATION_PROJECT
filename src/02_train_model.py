import pandas as pd
import numpy as np
import os
import argparse  # New import
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib

# --- Constants ---
SEQUENCE_LENGTH = 30  # Use the last 30 days of data to predict the next day

# --- Data and Model Path Configuration ---
# A dictionary to map the user's choice to the correct file paths
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


def load_and_preprocess_data(data_path, scaler_save_path):
    """Loads, encodes, and scales the dataset."""
    print(f"--- Step 2.1: Loading and Pre-processing Data from '{data_path}' ---")
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found at '{data_path}'.")
        print("Please run the data creation scripts first.")
        return None, None, None

    df = pd.read_csv(data_path, parse_dates=['date'])
    
    print(" -> Encoding categorical features (district, crop)...")
    if 'generated_irrigated' in df.columns:
        print(" -> (Found 'generated_irrigated' column in simulated data)")
    
    df = pd.get_dummies(df, columns=['district', 'crop'], prefix=['dist', 'crop'])
    
    df.set_index('date', inplace=True)

    print(" -> Scaling numerical features...")
    scaler = MinMaxScaler()
    
    scaled_data = scaler.fit_transform(df)
    
    # Save the scaler
    os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
    joblib.dump(scaler, scaler_save_path)
    print(f" -> Scaler saved to '{scaler_save_path}'")
    
    return scaled_data, df.columns


def create_sequences(data, columns, sequence_length):
    """Creates sequences of data for the LSTM model."""
    print(" -> Creating sequences...")
    X, y = [], []
    target_col_index = list(columns).index('real_soil_moisture_mm')
    
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length, target_col_index])
        
    return np.array(X), np.array(y)


def build_and_train_model(X_train, y_train, X_test, y_test, model_save_path):
    """Builds, compiles, and trains the LSTM model."""
    print("\n--- Step 2.2: Building the LSTM Model ---")
    
    model = Sequential([
        LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()
    
    print("\n--- Step 2.3: Training the Model ---")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=64,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print(f"\n -> Model training complete. Model saved to '{model_save_path}'")
    
    return model


if __name__ == "__main__":
    
    # --- New: Use argparse to get user's choice ---
    parser = argparse.ArgumentParser(description="Train the LSTM model on a specific dataset.")
    parser.add_argument(
        'model_type', 
        type=str, 
        choices=['simulated', 'real'], 
        help="The type of data to train on ('simulated' or 'real')."
    )
    args = parser.parse_args()
    
    # Get the correct paths based on user's choice
    paths = PATH_CONFIG[args.model_type]
    DATA_PATH = paths['data']
    MODEL_SAVE_PATH = paths['model']
    SCALER_SAVE_PATH = paths['scaler']
    
    print(f"--- Starting Training for '{args.model_type}' model ---")
    
    scaled_data, columns = load_and_preprocess_data(DATA_PATH, SCALER_SAVE_PATH)
    
    if scaled_data is not None:
        X, y = create_sequences(scaled_data, columns, SEQUENCE_LENGTH)
        
        # Chronological 80/20 split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False) 
        
        print(f" -> Data split into training and testing sets:")
        print(f"    Training sequences: {X_train.shape[0]}")
        print(f"    Testing sequences: {X_test.shape[0]}")

        build_and_train_model(X_train, y_train, X_test, y_test, MODEL_SAVE_PATH)
        
        print(f"\nSUCCESS: Model training for '{args.model_type}' is complete.")