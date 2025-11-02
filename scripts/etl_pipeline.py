# ETL Script - Clean and prepare station_day.csv for ML model training

import pandas as pd
import numpy as np
from datetime import datetime

def load_and_clean_data(csv_path='station_day.csv'):
    """Load CSV and perform initial cleaning"""
    print("Loading data...")
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    
    print(f"Original shape: {df.shape}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Stations: {df['StationId'].nunique()}")
    
    # Check missing values
    missing = df.isnull().sum()
    print("\nMissing values:")
    print(missing[missing > 0])
    
    return df


def feature_engineering(df):
    """Create features for ML model"""
    print("\nCreating features...")
    
    # Time features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Quarter'] = df['Date'].dt.quarter
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    
    # Binary features
    df['is_weekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    df['is_winter'] = df['Month'].isin([11, 12, 1, 2]).astype(int)
    
    # Lag features (previous days)
    for station in df['StationId'].unique():
        station_mask = df['StationId'] == station
        df.loc[station_mask, 'PM2.5_lag1'] = df.loc[station_mask, 'PM2.5'].shift(1)
        df.loc[station_mask, 'PM2.5_lag3'] = df.loc[station_mask, 'PM2.5'].shift(3)
        df.loc[station_mask, 'PM2.5_lag7'] = df.loc[station_mask, 'PM2.5'].shift(7)
        df.loc[station_mask, 'AQI_lag1'] = df.loc[station_mask, 'AQI'].shift(1)
        
        # Rolling statistics (moving averages)
        df.loc[station_mask, 'PM2.5_ma3'] = df.loc[station_mask, 'PM2.5'].rolling(3, min_periods=1).mean()
        df.loc[station_mask, 'PM2.5_ma7'] = df.loc[station_mask, 'PM2.5'].rolling(7, min_periods=1).mean()
        df.loc[station_mask, 'PM2.5_ma30'] = df.loc[station_mask, 'PM2.5'].rolling(30, min_periods=1).mean()
        
        # Rolling std (volatility)
        df.loc[station_mask, 'PM2.5_std7'] = df.loc[station_mask, 'PM2.5'].rolling(7, min_periods=1).std()
    
    # Pollutant ratios
    df['PM_ratio'] = df['PM2.5'] / (df['PM10'] + 1)  # +1 to avoid division by zero
    df['NOx_total'] = df['NO'] + df['NO2']
    
    print(f"Features created. New shape: {df.shape}")
    
    return df


def handle_missing_values(df):
    """Handle missing values appropriately"""
    print("\nHandling missing values...")
    
    # For numeric columns, fill with median per station
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df.groupby('StationId')[col].transform(
                lambda x: x.fillna(x.median())
            )
    
    # If still missing after groupby fillna, use global median
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Drop rows where target (PM2.5 or AQI) is still missing
    df = df.dropna(subset=['PM2.5', 'AQI'])
    
    print(f"After handling missing: {df.shape}")
    print(f"Remaining missing values: {df.isnull().sum().sum()}")
    
    return df


def save_cleaned_data(df, output_path='data/cleaned/station_day_clean.csv'):
    """Save cleaned data"""
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"\nCleaned data saved to: {output_path}")
    print(f"Final shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    # Run ETL pipeline
    df = load_and_clean_data('station_day.csv')
    df = feature_engineering(df)
    df = handle_missing_values(df)
    save_cleaned_data(df)
    
    # Summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    print(df[['PM2.5', 'PM10', 'AQI', 'PM2.5_ma7']].describe())
    
    print("\nAQI Category distribution:")
    print(df['AQI_Bucket'].value_counts())
    
    print("\nETL Pipeline completed successfully! ✅")
