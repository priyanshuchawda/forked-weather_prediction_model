import os
import time
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class EnsembleModel:
    def __init__(self, n_estimators_rf=200, n_estimators_xgb=300):
        self.rf = RandomForestRegressor(n_estimators=n_estimators_rf, random_state=42)
        self.xgb = XGBRegressor(
            n_estimators=n_estimators_xgb, learning_rate=0.05, max_depth=5, 
            verbosity=0, random_state=42
        )
        self.lr = LinearRegression()
        
    def fit(self, X, y):
        self.rf.fit(X, y)
        self.xgb.fit(X, y)
        self.lr.fit(X, y)
        return self
        
    def predict(self, X):
        return (self.rf.predict(X) + self.xgb.predict(X) + self.lr.predict(X)) / 3

def should_retrain_model(data_path='pune_weather_cleaned.csv', model_path='weather_model_enhanced.pkl'):
    """Check if the model needs to be retrained."""
    if not os.path.exists(model_path):
        return True
    
    data_mod_time = os.path.getmtime(data_path)
    model_mod_time = os.path.getmtime(model_path)
    
    return data_mod_time > model_mod_time

def train_and_save_model():
    """Train the weather prediction model and save it to a file."""
    # ... (rest of the training code remains the same)
    # -------------------------
    # STEP 1 — LOAD DATA
    # -------------------------
    df = pd.read_csv('pune_weather_cleaned.csv')
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time')

    # Drop unused or empty columns
    drop_cols = [col for col in ['snow', 'wdwr', 'wpgt', 'tsun'] if col in df.columns]
    df = df.drop(columns=drop_cols, errors='ignore')

    # Process main columns first
    main_cols = ['tavg', 'prcp', 'wspd']
    print("Initial data shape:", df.shape)

    # Drop unnecessary columns that might have missing data
    df = df.drop(['wdir', 'tmax', 'pres', 'temp_diff'], axis=1, errors='ignore')

    # Forward fill first, then backward fill any remaining gaps
    df[main_cols] = df[main_cols].ffill().bfill()

    # Drop any invalid dates or duplicates
    df = df.drop_duplicates(subset=['time'])
    df = df.sort_values('time').reset_index(drop=True)
    print("Data shape after cleaning:", df.shape)

    # -------------------------
    # STEP 2 — FEATURE ENGINEERING
    # -------------------------
    # Create a copy for safety
    df_temp = df.copy()

    # Add seasonality first (no dependencies)
    df_temp['month_sin'] = np.sin(2 * np.pi * df_temp['month'] / 12)
    df_temp['month_cos'] = np.cos(2 * np.pi * df_temp['month'] / 12)

    # Store the latest date for reference
    latest_date = df_temp['time'].max()
    with open('last_update.txt', 'w') as f:
        f.write(latest_date.strftime('%Y-%m-%d'))

    # Add lag features carefully
    for lag in [1, 3, 7]:
        for col in main_cols:
            df_temp[f'{col}_lag{lag}'] = df_temp[col].shift(lag)

    # Add rolling averages
    for window in [3, 7]:
        for col in main_cols:
            df_temp[f'{col}_roll{window}'] = df_temp[col].rolling(window, min_periods=1).mean()

    # Check data quality
    print("\nFeature engineering summary:")
    print("----------------------------")
    print("Shape after features added:", df_temp.shape)

    # Forward fill any NaN values in derived features
    lag_cols = [col for col in df_temp.columns if '_lag' in col]
    roll_cols = [col for col in df_temp.columns if '_roll' in col]

    for col in lag_cols:
        df_temp[col] = df_temp[col].ffill().bfill()
    for col in roll_cols:
        df_temp[col] = df_temp[col].ffill().bfill()

    # Final data quality check
    null_counts = df_temp.isna().sum()
    if null_counts.sum() > 0:
        print("\nWarning: Found columns with null values:")
        print(null_counts[null_counts > 0])
    else:
        print("\nNo null values found in the dataset")

    df = df_temp  # Replace original with cleaned version
    print(f"\nFinal data shape: {df.shape}")
    print("Remaining null values:", df.isna().sum().sum())

    # -------------------------
    # STEP 3 — TRAIN/TEST SPLIT
    # -------------------------
    features = [col for col in df.columns if col not in ['time', 'tavg', 'prcp', 'wspd']]
    X = df[features]
    y = df[['tavg', 'prcp', 'wspd']]

    # Use last 10% for testing
    split_idx = int(len(df) * 0.9)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    yT_train, yP_train, yW_train = y_train['tavg'], y_train['prcp'], y_train['wspd']
    yT_test, yP_test, yW_test = y_test['tavg'], y_test['prcp'], y_test['wspd']

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -------------------------
    # STEP 4 — MODEL TRAINING
    # -------------------------
    def train_model(y_train):
        model = EnsembleModel()
        model.fit(X_train_scaled, y_train)
        return model


    # Train each model
    models_temp = train_model(yT_train)
    models_prcp = train_model(yP_train)
    models_wspd = train_model(yW_train)

    # -------------------------
    # STEP 5 — MODEL EVALUATION
    # -------------------------
    print("\nModel 1 (Hybrid Ensemble) Evaluation:")
    print("=====================================")
    for name, y_true, model in [
        ('Temperature', yT_test, models_temp),
        ('Precipitation', yP_test, models_prcp),
        ('Wind Speed', yW_test, models_wspd),
    ]:
        y_pred = model.predict(X_test_scaled)
        print(f"\n{name} Model Evaluation:")
        print("MAE :", round(mean_absolute_error(y_true, y_pred), 3))
        print("RMSE:", round(np.sqrt(mean_squared_error(y_true, y_pred)), 3))
        print("R²  :", round(r2_score(y_true, y_pred), 3))

    # Save models
    model_data = {
        'temp_model': models_temp,
        'prcp_model': models_prcp,
        'wspd_model': models_wspd,
        'scaler': scaler,
        'features': features
    }

    joblib.dump(model_data, 'weather_model_enhanced.pkl')
    print("\n✅ Enhanced weather model trained and saved successfully.")
    return model_data

# -------------------------
# STEP 6 — FORECAST NEXT 7 DAYS
# -------------------------
def forecast_next_7_days(target_date=None):
    if should_retrain_model():
        print("Retraining model...")
        model_data = train_and_save_model()
    else:
        print("Loading pre-trained model...")
        model_data = joblib.load('weather_model_enhanced.pkl')

    # Load df directly from CSV
    df = pd.read_csv('pune_weather_cleaned.csv')
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time')

    # Drop unused or empty columns
    drop_cols = [col for col in ['snow', 'wdwr', 'wpgt', 'tsun'] if col in df.columns]
    df = df.drop(columns=drop_cols, errors='ignore')

    # Process main columns first
    main_cols = ['tavg', 'prcp', 'wspd']
    df[main_cols] = df[main_cols].ffill().bfill()

    # Drop any invalid dates or duplicates
    df = df.drop_duplicates(subset=['time'])
    df = df.sort_values('time').reset_index(drop=True)

    # Re-create features for df
    df_temp = df.copy()
    df_temp['month_sin'] = np.sin(2 * np.pi * df_temp['month'] / 12)
    df_temp['month_cos'] = np.cos(2 * np.pi * df_temp['month'] / 12)
    for lag in [1, 3, 7]:
        for col in main_cols:
            df_temp[f'{col}_lag{lag}'] = df_temp[col].shift(lag)
    for window in [3, 7]:
        for col in main_cols:
            df_temp[f'{col}_roll{window}'] = df_temp[col].rolling(window, min_periods=1).mean()
    
    lag_cols = [col for col in df_temp.columns if '_lag' in col]
    roll_cols = [col for col in df_temp.columns if '_roll' in col]

    for col in lag_cols:
        df_temp[col] = df_temp[col].ffill().bfill()
    for col in roll_cols:
        df_temp[col] = df_temp[col].ffill().bfill()
    df = df_temp

    features = model_data['features']
    scaler = model_data['scaler']
    temp_model = model_data['temp_model']
    prcp_model = model_data['prcp_model']
    wspd_model = model_data['wspd_model']

    if target_date is None:
        target_date = datetime.now()
    
    # Get the last 7 days of data
    last_known_data = df.copy().iloc[-7:].reset_index(drop=True)
    forecasts = []
    
    for i in range(7):
        current_date = target_date + timedelta(days=i)
        
        # Update features for the latest data
        latest = pd.DataFrame(index=[0])
        
        # Add lag features
        for lag in [1, 3, 7]:
            if i >= lag:
                # Use predicted values for lags
                idx = i - lag
                latest[f'tavg_lag{lag}'] = [forecasts[idx]['temp']]
                latest[f'prcp_lag{lag}'] = [forecasts[idx]['prcp']]
                latest[f'wspd_lag{lag}'] = [forecasts[idx]['wspd']]
            else:
                # Use actual historical data for initial predictions
                row_idx = -lag
                latest[f'tavg_lag{lag}'] = [last_known_data.iloc[row_idx]['tavg']]
                latest[f'prcp_lag{lag}'] = [last_known_data.iloc[row_idx]['prcp']]
                latest[f'wspd_lag{lag}'] = [last_known_data.iloc[row_idx]['wspd']]
        
        # Add rolling averages
        for window in [3, 7]:
            if i >= window:
                # Use predicted values for the window
                temps = [f['temp'] for f in forecasts[max(0, i-window):i]]
                prcps = [f['prcp'] for f in forecasts[max(0, i-window):i]]
                wspds = [f['wspd'] for f in forecasts[max(0, i-window):i]]
                
                # Fill missing with historical data
                if len(temps) < window:
                    hist_data = last_known_data.iloc[-(window-len(temps)):][['tavg', 'prcp', 'wspd']]
                    temps = list(hist_data['tavg']) + temps
                    prcps = list(hist_data['prcp']) + prcps
                    wspds = list(hist_data['wspd']) + wspds
                
                latest[f'tavg_roll{window}'] = [np.mean(temps)]
                latest[f'prcp_roll{window}'] = [np.mean(prcps)]
                latest[f'wspd_roll{window}'] = [np.mean(wspds)]
            else:
                # Use historical data for initial windows
                latest[f'tavg_roll{window}'] = [last_known_data['tavg'].iloc[-window:].mean()]
                latest[f'prcp_roll{window}'] = [last_known_data['prcp'].iloc[-window:].mean()]
                latest[f'wspd_roll{window}'] = [last_known_data['wspd'].iloc[-window:].mean()]
        
        # Add seasonality features
        latest['month'] = current_date.month
        latest['month_sin'] = np.sin(2 * np.pi * current_date.month / 12)
        latest['month_cos'] = np.cos(2 * np.pi * current_date.month / 12)
        
        # Ensure all required features are present
        missing_cols = set(features) - set(latest.columns)
        for col in missing_cols:
            latest[col] = last_known_data[col].iloc[-1]
        
        # Scale and predict
        X_latest = latest[features]
        X_scaled = scaler.transform(X_latest)
        
        temp_pred = temp_model.predict(X_scaled)[0]
        prcp_pred = prcp_model.predict(X_scaled)[0]
        wspd_pred = wspd_model.predict(X_scaled)[0]
        
        forecasts.append({
            'date': current_date,
            'temp': temp_pred,
            'prcp': prcp_pred,
            'wspd': wspd_pred
        })
    
    return forecasts

if __name__ == "__main__":
    # -------------------------
    # STEP 7 — EXECUTE FORECAST
    # -------------------------
    current_date = datetime.now()
    forecasts = forecast_next_7_days(current_date)

    print(f"\n7-Day Weather Forecast Starting {current_date.strftime('%B %d, %Y')}:")
    print("=" * (len(f"7-Day Weather Forecast Starting {current_date.strftime('%B %d, %Y')}:") + 5))
    for f in forecasts:
        date_str = f['date'].strftime('%Y-%m-%d')
        print(f"{date_str}: Temp={f['temp']:.1f}°C, Rain={f['prcp']:.1f}mm, Wind={f['wspd']:.1f} km/h")