# Train ML Model - RandomForest for PM2.5 and AQI Forecasting

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from datetime import datetime


def load_cleaned_data(csv_path='data/cleaned/station_day_clean.csv'):
    """Load cleaned data from ETL pipeline"""
    print("Loading cleaned data...")
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    print(f"Loaded shape: {df.shape}")
    return df


def prepare_features(df):
    """Select features for training"""
    print("\nPreparing features...")
    
    # Define feature columns (exclude target and non-numeric)
    exclude_cols = ['StationId', 'Date', 'PM2.5', 'AQI', 'AQI_Bucket']
    feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
    
    # Remove rows with lag features = NaN (first few days)
    df = df.dropna(subset=feature_cols)
    
    print(f"Feature columns ({len(feature_cols)}): {feature_cols[:10]}...")
    print(f"Training shape after dropna: {df.shape}")
    
    return df, feature_cols


def train_pm25_model(df, feature_cols):
    """Train RandomForest model for PM2.5 forecasting"""
    print("\n" + "="*50)
    print("TRAINING PM2.5 MODEL")
    print("="*50)
    
    X = df[feature_cols]
    y = df['PM2.5']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False  # Time series: don't shuffle
    )
    
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Train model
    print("\nTraining RandomForest...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    print("\n📊 PM2.5 Model Performance:")
    print(f"Train MAE: {train_mae:.2f}, RMSE: {train_rmse:.2f}, R²: {train_r2:.3f}")
    print(f"Test MAE: {test_mae:.2f}, RMSE: {test_rmse:.2f}, R²: {test_r2:.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🔝 Top 10 Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    return model, test_mae, test_r2


def train_aqi_model(df, feature_cols):
    """Train RandomForest model for AQI forecasting"""
    print("\n" + "="*50)
    print("TRAINING AQI MODEL")
    print("="*50)
    
    X = df[feature_cols]
    y = df['AQI']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Train model
    print("\nTraining RandomForest...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    print("\n📊 AQI Model Performance:")
    print(f"Train MAE: {train_mae:.2f}, RMSE: {train_rmse:.2f}, R²: {train_r2:.3f}")
    print(f"Test MAE: {test_mae:.2f}, RMSE: {test_rmse:.2f}, R²: {test_r2:.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🔝 Top 10 Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    return model, test_mae, test_r2


def save_models(pm25_model, aqi_model, feature_cols, metrics):
    """Save trained models and metadata"""
    os.makedirs('models', exist_ok=True)
    
    # Save models
    joblib.dump(pm25_model, 'models/pm25_forecast.pkl')
    joblib.dump(aqi_model, 'models/aqi_forecast.pkl')
    joblib.dump(feature_cols, 'models/feature_columns.pkl')
    
    # Save metadata
    metadata = {
        'trained_at': datetime.now().isoformat(),
        'feature_count': len(feature_cols),
        'pm25_test_mae': metrics['pm25_mae'],
        'pm25_test_r2': metrics['pm25_r2'],
        'aqi_test_mae': metrics['aqi_mae'],
        'aqi_test_r2': metrics['aqi_r2'],
        'model_type': 'RandomForestRegressor',
        'n_estimators': 100,
        'max_depth': 20
    }
    
    import json
    with open('models/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n✅ Models saved to models/:")
    print("  - pm25_forecast.pkl")
    print("  - aqi_forecast.pkl")
    print("  - feature_columns.pkl")
    print("  - metadata.json")


if __name__ == "__main__":
    # Load data
    df = load_cleaned_data()
    
    # Prepare features
    df, feature_cols = prepare_features(df)
    
    # Train models
    pm25_model, pm25_mae, pm25_r2 = train_pm25_model(df, feature_cols)
    aqi_model, aqi_mae, aqi_r2 = train_aqi_model(df, feature_cols)
    
    # Save models
    metrics = {
        'pm25_mae': pm25_mae,
        'pm25_r2': pm25_r2,
        'aqi_mae': aqi_mae,
        'aqi_r2': aqi_r2
    }
    save_models(pm25_model, aqi_model, feature_cols, metrics)
    
    print("\n" + "="*50)
    print("🎉 Model Training Completed Successfully!")
    print("="*50)
    print(f"\n📈 Final Metrics:")
    print(f"PM2.5 - Test MAE: {pm25_mae:.2f}, R²: {pm25_r2:.3f}")
    print(f"AQI - Test MAE: {aqi_mae:.2f}, R²: {aqi_r2:.3f}")
    print("\nModels ready for deployment! 🚀")
