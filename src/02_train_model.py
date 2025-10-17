import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib

# --- Constants ---
DATA_PATH = "data/FINAL_DATA_for_LSTM_Training.csv"
MODEL_SAVE_PATH = "models/irrigation_lstm_model.keras"
SCALER_SAVE_PATH = "models/data_scaler.gz"
SEQUENCE_LENGTH = 30 # Use the last 30 days of data to predict the next day

# --- 1. Load and Pre-process Data ---
def load_and_preprocess_data(data_path):
    """Loads, encodes, and scales the dataset."""
    print("--- Step 2.1: Loading and Pre-processing Data ---")
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found at '{data_path}'.")
        print("Please run 'src/01_fetch_and_simulate.py' first.")
        return None, None, None

    df = pd.read_csv(data_path, parse_dates=['date'])
    
    # --- One-Hot Encode Categorical Features ---
    print(" -> Encoding categorical features (district, crop)...")
    df = pd.get_dummies(df, columns=['district', 'crop'], prefix=['dist', 'crop'])
    
    # Ensure all expected columns are present, even if some crops/dists weren't in the data
    expected_cols = ['dist_Baghpat', 'dist_Shamli', 'dist_Meerut', 'dist_Muzaffarnagar', 'dist_Ghaziabad', 'crop_Wheat', 'crop_Rice', 'crop_None']
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
            
    # Set date as index
    df.set_index('date', inplace=True)
    
    # --- Scale Numerical Features ---
    print(" -> Scaling numerical features...")
    scaler = MinMaxScaler()
    # Fit and transform the data
    scaled_data = scaler.fit_transform(df)
    
    # Save the scaler for later use in prediction
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f" -> Scaler saved to '{SCALER_SAVE_PATH}'")
    
    return scaled_data, df.columns

def create_sequences(data, columns, sequence_length):
    """Creates sequences of data for the LSTM model."""
    print(" -> Creating sequences...")
    X, y = [], []
    # Find the index of our target variable
    target_col_index = list(columns).index('real_soil_moisture_mm')
    
    for i in range(len(data) - sequence_length):
        # The sequence of input features
        X.append(data[i:(i + sequence_length)])
        # The target value (soil moisture) at the end of the sequence
        y.append(data[i + sequence_length, target_col_index])
        
    return np.array(X), np.array(y)

# --- 2. Build and Train LSTM Model ---
def build_and_train_model(X_train, y_train, X_test, y_test):
    """Builds, compiles, and trains the LSTM model."""
    print("\n--- Step 2.2: Building the LSTM Model ---")
    
    model = Sequential([
        LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1) # Output layer: predicts a single value
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()
    
    print("\n--- Step 2.3: Training the Model ---")
    # Using a portion of the training data for validation during training
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=64,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Save the trained model
    model.save(MODEL_SAVE_PATH)
    print(f"\n -> Model training complete. Model saved to '{MODEL_SAVE_PATH}'")
    
    return model

# --- Main execution ---
if __name__ == "__main__":
    
    # Load and process the data
    scaled_data, columns = load_and_preprocess_data(DATA_PATH)
    
    if scaled_data is not None:
        # Create sequences
        X, y = create_sequences(scaled_data, columns, SEQUENCE_LENGTH)
        
        # Split data into training and testing sets (chronological split)
        # We'll use the last 2 years for testing
        test_set_size = 365 * 2 * 5 # 2 years * 5 districts
        
        X_train = X[:-test_set_size]
        y_train = y[:-test_set_size]
        X_test = X[-test_set_size:]
        y_test = y[-test_set_size:]
        
        print(f" -> Data split into training and testing sets:")
        print(f"    Training sequences: {X_train.shape[0]}")
        print(f"    Testing sequences: {X_test.shape[0]}")

        # Build and train the model
        build_and_train_model(X_train, y_train, X_test, y_test)
        
        print("\nSUCCESS: Model training phase is complete.")
