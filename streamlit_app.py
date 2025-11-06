import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model # Assuming TensorFlow is available

# --- Global Configuration and Constants ---

SEQUENCE_LENGTH = 30 
DISTRICTS = ["Baghpat", "Shamli", "Meerut", "Muzaffarnagar", "Ghaziabad"]

# Path configuration maps
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

# Agricultural Rules (from 03_prediction.py)
# CORRECTION: WHEAT_TRIGGER_mm calculated as Field Capacity (225 mm) - (Available Water (90 mm) * MAD (0.50)) = 180.0 mm
WHEAT_TRIGGER_mm = 180.0 
CROP_PROFILES = {
    'Wheat': {'TRIGGER_mm': WHEAT_TRIGGER_mm, 'LOGIC': 'MAD'},
    'Rice': {'TRIGGER_mm': 20.0, 'LOGIC': 'Ponding'},
    'None': {'TRIGGER_mm': WHEAT_TRIGGER_mm} # Fallback
}

# --- Caching Functions (Crucial for Streamlit Performance) ---

# Target column name MUST match the name used when the scaler was trained
TARGET_COL_NAME = 'real_soil_moisture_mm'

@st.cache_resource
def load_assets(model_type):
    """Loads and caches the model, scaler, and full dataset."""
    paths = PATH_CONFIG[model_type]
    
    if not all(os.path.exists(paths[k]) for k in ['model', 'scaler', 'data']):
        st.error(f"ERROR: One or more required files for the '{model_type}' model are missing.")
        st.info("Please run the data creation and model training scripts first (`01_...py`, `01_b_...py`, `02_train_model.py`).")
        return None, None, None
        
    try:
        model = load_model(paths['model'])
        scaler = joblib.load(paths['scaler'])
        df = pd.read_csv(paths['data'], parse_dates=['date'])
        
        # Rename target column for consistent use in the app
        # NOTE: The variable in the app is 'real_moisture', but the scaler still uses 'real_soil_moisture_mm'
        df.rename(columns={'real_soil_moisture_mm': 'real_moisture'}, inplace=True)
        
        return df, model, scaler
    except Exception as e:
        st.error(f"Failed to load assets for '{model_type}': {e}")
        return None, None, None


# --- Prediction Logic (Adapted from 03_prediction.py) ---

def prepare_prediction_data(last_30_days_df, scaler):
    """Prepares the last 30 days of data for prediction, ensuring column consistency."""
    df_processed = last_30_days_df.copy().set_index('date')
    
    # 1. One-Hot Encode (only district and crop are necessary from input)
    df_encoded = pd.get_dummies(df_processed, columns=['district', 'crop'], prefix=['dist', 'crop'])
    
    scaler_feature_names = scaler.feature_names_in_
    
    # 2. Add/Zero-out missing features to match the scaler's training data features
    for col in scaler_feature_names:
        # Check against the original column name used in the scaler: TARGET_COL_NAME
        if col not in df_encoded.columns and col != TARGET_COL_NAME: 
            df_encoded[col] = 0.0
            
    # Add a dummy target column for proper scaling if it's missing
    if TARGET_COL_NAME not in df_encoded.columns:
        df_encoded[TARGET_COL_NAME] = 0.0

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
    
    # FIX: Use the original column name (TARGET_COL_NAME) to find the index in the scaler
    target_col_index = list(scaler.feature_names_in_).index(TARGET_COL_NAME)
    
    # Place the prediction into the target column position
    dummy_row[0, target_col_index] = prediction_scaled
    
    unscaled_row = scaler.inverse_transform(dummy_row)
    
    return unscaled_row[0, target_col_index]

def run_prediction_logic(df, model, scaler, district_name):
    """Runs the full prediction pipeline for the selected district."""
    
    # Get the last 30 days of data for the chosen district
    sample_df = df[df['district'] == district_name].tail(SEQUENCE_LENGTH + 1).copy() # +1 to get the actual result for the prediction day

    if len(sample_df) < SEQUENCE_LENGTH + 1:
        st.warning(f"Insufficient data in '{district_name}' to make a {SEQUENCE_LENGTH}-day sequence prediction.")
        return None, None, None, None

    # The model uses the first 30 days of this slice to predict the 31st day
    input_df = sample_df.head(SEQUENCE_LENGTH) 

    # Identify prediction date, crop, and current moisture for the display
    prediction_date = sample_df.iloc[-1]['date']
    today_date = sample_df.iloc[-2]['date']
    
    # Use the crop from the prediction day
    predicting_for_crop = sample_df.iloc[-1]['crop']
    
    # The 'real_moisture' of the last known day (i.e., day 30)
    current_moisture = input_df.iloc[-1]['real_moisture'] 
    
    # --- Predict ---
    # Temporarily rename 'real_moisture' back to the original name for prediction/scaling
    input_df.rename(columns={'real_moisture': TARGET_COL_NAME}, inplace=True)
    
    input_sequence = prepare_prediction_data(input_df, scaler)
    predicted_scaled_value = model.predict(input_sequence, verbose=0)[0][0]
    predicted_moisture_mm = inverse_transform_prediction(predicted_scaled_value, scaler)
    
    # Restore the column name in the input_df copy, though it's not strictly necessary here.
    input_df.rename(columns={TARGET_COL_NAME: 'real_moisture'}, inplace=True) 
    
    return prediction_date, predicting_for_crop, current_moisture, predicted_moisture_mm


# --- Visualization Logic (Adapted from 05_compare_datasets.py and 06_analyze_weather_link.py) ---

def plot_comparison_timeseries(df_real, df_sim, district_name):
    """Plots the time series of Real vs. Simulated moisture."""
    
    if df_real is None or df_sim is None:
        st.warning("Cannot generate comparison plot: data is missing or failed to load.")
        return
        
    # Use copies to avoid modifying the cached dataframes
    df_real_copy = df_real.copy().rename(columns={'real_moisture': 'Real (Satellite) Moisture'})
    df_sim_copy = df_sim.copy().rename(columns={'real_moisture': 'Simulated (Water-Balance) Moisture'})

    df_real_dist = df_real_copy[df_real_copy['district'] == district_name][['date', 'Real (Satellite) Moisture']]
    df_sim_dist = df_sim_copy[df_sim_copy['district'] == district_name][['date', 'Simulated (Water-Balance) Moisture']]

    df_merged = pd.merge(df_real_dist, df_sim_dist, on='date', how='inner')
    df_merged.set_index('date', inplace=True)
    
    st.subheader(f"Real vs. Simulated Moisture Time Series ({district_name})")
    st.line_chart(df_merged)


def plot_correlation_heatmap(df):
    """Plots a correlation heatmap for key features."""
    
    st.subheader("Correlation Heatmap of Features and Soil Moisture")
    
    df_numeric = df[[
        'real_moisture', 'precip_mm', 'ETo_mm', 
        'temperature_C', 'wind_m_s', 'solar_rad_MJ_m2'
    ]].copy()
    
    df_numeric.rename(columns={
        'real_moisture': 'Soil Moisture (Target)',
        'precip_mm': 'Precipitation',
        'ETo_mm': 'Evaporation (ETo)',
        'temperature_C': 'Temperature',
        'wind_m_s': 'Wind Speed',
        'solar_rad_MJ_m2': 'Solar Radiation'
    }, inplace=True)
    
    corr = df_numeric.corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1, ax=ax)
    plt.title('Feature Correlation Matrix')
    st.pyplot(fig)


# --- Main Streamlit Application ---

def main():
    
    st.set_page_config(
        page_title="LSTM-Powered Irrigation Advisor",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🌾 LSTM-Powered Irrigation Advisor")
    st.markdown("Forecasting soil moisture for improved water management in Western Uttar Pradesh.")

    # --- Sidebar Configuration ---
    st.sidebar.header("Configuration")
    
    model_type = st.sidebar.radio(
        "Select Model/Data Source:",
        ('real', 'simulated'),
        index=0,
        help="Choose between the model trained on Simulated data or the model trained on Real (Satellite-derived) data."
    )
    
    selected_district = st.sidebar.selectbox(
        "Select District:",
        DISTRICTS,
        index=2
    )
    
    df_real, model_real, scaler_real = load_assets('real')
    df_sim, model_sim, scaler_sim = load_assets('simulated')
    
    # --- Tab Layout ---
    tab1, tab2 = st.tabs(["💧 Irrigation Prediction", "📈 Data & Model Analysis"])

    # --- Tab 1: Irrigation Prediction ---
    with tab1:
        st.header(f"Next-Day Forecast for {selected_district}")
        
        # Determine which model and data to use
        if model_type == 'real':
            df, model, scaler = df_real, model_real, scaler_real
        else:
            df, model, scaler = df_sim, model_sim, scaler_sim

        if df is not None and model is not None and scaler is not None:
            
            # Run the prediction logic
            prediction_date, predicting_for_crop, current_moisture, predicted_moisture_mm = run_prediction_logic(
                df, model, scaler, selected_district
            )

            if predicted_moisture_mm is not None:
                trigger_level = CROP_PROFILES.get(predicting_for_crop, CROP_PROFILES['None'])['TRIGGER_mm']
                
                # Decision logic
                if predicted_moisture_mm < trigger_level:
                    decision = "IRRIGATION IS REQUIRED"
                    color = "red"
                    icon = "🚨"
                else:
                    decision = "NO IRRIGATION NEEDED"
                    color = "green"
                    icon = "✅"
                
                # --- Display Results ---
                col_info, col_pred, col_dec = st.columns(3)

                with col_info:
                    # current_moisture and today_date are calculated correctly
                    st.metric(
                        label="Last Recorded Moisture", 
                        value=f"{current_moisture:.2f} mm",
                        help=f"Moisture on {prediction_date.strftime('%Y-%m-%d')} used as base for prediction." # Note: today_date is not passed from run_prediction_logic
                    )
                    st.info(f"Crop: **{predicting_for_crop}** (Trigger: {trigger_level:.2f} mm)")

                with col_pred:
                    st.metric(
                        label=f"Predicted Moisture for {prediction_date.strftime('%Y-%m-%d')}",
                        value=f"{predicted_moisture_mm:.2f} mm",
                        delta=f"{predicted_moisture_mm - current_moisture:.2f} mm change"
                    )
                
                with col_dec:
                    st.markdown(f"<div style='background-color: {color}; padding: 20px; border-radius: 10px; text-align: center;'><h2>{icon} {decision}</h2></div>", unsafe_allow_html=True)
                
                
                st.markdown("---")
                
                # Plot recent history vs. prediction
                recent_data = df[df['district'] == selected_district].tail(SEQUENCE_LENGTH * 2).copy()
                recent_data.set_index('date', inplace=True)
                
                # Prepare data for chart
                chart_data = recent_data['real_moisture'].tail(SEQUENCE_LENGTH)
                
                # Inject the prediction point
                prediction_point = pd.Series([predicted_moisture_mm], index=[prediction_date])
                chart_data = pd.concat([chart_data, prediction_point])
                
                st.subheader("Recent Soil Moisture Trend")
                st.line_chart(chart_data)
                st.markdown(f"*The last point is the predicted value. The horizontal line shows the trigger level of **{trigger_level:.2f} mm**.*")
                st.markdown(f"<hr style='border: 1px solid {color}'>", unsafe_allow_html=True)


        else:
            st.error("Cannot run prediction: Model assets failed to load. Please check the console output and ensure all required files exist.")


    # --- Tab 2: Data Analysis ---
    with tab2:
        st.header("Project Insights and Validation")
        
        # Section 1: Data Comparison (Requires both real and simulated data)
        st.markdown("## Dataset Comparison (Real vs. Simulated)")
        col_comp_1, col_comp_2 = st.columns(2)
        
        with col_comp_1:
            st.info("The real-world satellite data and the simulated water-balance data are compared here.")
            if df_real is not None and df_sim is not None:
                # Calculate correlation between real and simulated time series for the selected district
                df_real_moist = df_real[df_real['district'] == selected_district].set_index('date')['real_moisture']
                df_sim_moist = df_sim[df_sim['district'] == selected_district].set_index('date')['real_moisture']
                
                df_corr = pd.merge(df_real_moist.rename('Real'), df_sim_moist.rename('Simulated'), left_index=True, right_index=True, how='inner').dropna()
                
                if not df_corr.empty:
                    correlation = df_corr['Real'].corr(df_corr['Simulated'])
                    st.metric(label="Correlation (Real vs. Simulated)", value=f"{correlation:.3f}")
                else:
                    st.warning("Not enough overlapping data for correlation calculation.")

        with col_comp_2:
            if df_real is not None and df_sim is not None:
                # Plot comparison time series
                # Need to use the full dataframes, not the ones potentially filtered in Tab 1
                df_full_sim = pd.read_csv(PATH_CONFIG['simulated']['data'], parse_dates=['date']).rename(columns={'real_soil_moisture_mm': 'real_moisture'})
                
                plot_comparison_timeseries(df_real, df_full_sim, selected_district)
            else:
                st.warning("Cannot plot comparison: Both 'real' and 'simulated' datasets are required.")


        st.markdown("---")

        # Section 2: Feature Analysis (Requires only real data)
        st.markdown("## Feature and Weather Analysis")
        if df_real is not None:
            plot_correlation_heatmap(df_real)
        else:
            st.warning("Cannot plot correlation: Real dataset is missing or failed to load.")


if __name__ == "__main__":
    main()