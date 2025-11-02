# ML Model Utilities - Load and use trained models for predictions

import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os


class WeatherMLPredictor:
    """Load and use trained ML models for weather forecasting"""
    
    def __init__(self):
        self.pm25_model = None
        self.aqi_model = None
        self.feature_cols = None
        self.metadata = None
        self.models_loaded = False
        
        # Try to load models on initialization
        self.load_models()
    
    def load_models(self):
        """Load trained models from disk"""
        try:
            if not os.path.exists('models/pm25_forecast.pkl'):
                print("⚠️  Models not found. Run scripts/train_model.py first.")
                return False
            
            self.pm25_model = joblib.load('models/pm25_forecast.pkl')
            self.aqi_model = joblib.load('models/aqi_forecast.pkl')
            self.feature_cols = joblib.load('models/feature_columns.pkl')
            
            # Load metadata
            import json
            with open('models/metadata.json', 'r') as f:
                self.metadata = json.load(f)
            
            self.models_loaded = True
            print(f"✅ Models loaded successfully! Trained at: {self.metadata['trained_at']}")
            print(f"📊 PM2.5 R²: {self.metadata['pm25_test_r2']:.3f}, AQI R²: {self.metadata['aqi_test_r2']:.3f}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {str(e)}")
            return False
    
    def prepare_features_from_history(self, historical_data: list[dict]) -> pd.DataFrame:
        """
        Convert historical weather data to model features
        
        Args:
            historical_data: List of dicts with keys: date, pm25, pm10, aqi, no, no2, etc.
        
        Returns:
            DataFrame with required features
        """
        # Convert to DataFrame
        df = pd.DataFrame(historical_data)
        df['Date'] = pd.to_datetime(df['date'])
        df = df.sort_values('Date')
        
        # Create time features
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Quarter'] = df['Date'].dt.quarter
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        df['is_weekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
        df['is_winter'] = df['Month'].isin([11, 12, 1, 2]).astype(int)
        
        # Create lag features (need at least 7 days of history)
        df['PM2.5_lag1'] = df['pm25'].shift(1)
        df['PM2.5_lag3'] = df['pm25'].shift(3)
        df['PM2.5_lag7'] = df['pm25'].shift(7)
        df['AQI_lag1'] = df['aqi'].shift(1)
        
        # Rolling statistics
        df['PM2.5_ma3'] = df['pm25'].rolling(3, min_periods=1).mean()
        df['PM2.5_ma7'] = df['pm25'].rolling(7, min_periods=1).mean()
        df['PM2.5_ma30'] = df['pm25'].rolling(30, min_periods=1).mean()
        df['PM2.5_std7'] = df['pm25'].rolling(7, min_periods=1).std()
        
        # Pollutant ratios
        df['PM_ratio'] = df['pm25'] / (df['pm10'] + 1)
        df['NOx_total'] = df.get('no', 0) + df.get('no2', 0)
        
        return df
    
    def predict_next_day(self, historical_data: list[dict]) -> dict:
        """
        Predict next day's PM2.5 and AQI
        
        Args:
            historical_data: List of past 30 days data
                Example: [
                    {'date': '2024-01-01', 'pm25': 45.2, 'pm10': 89.1, 'aqi': 102, ...},
                    {'date': '2024-01-02', 'pm25': 50.3, 'pm10': 95.2, 'aqi': 115, ...},
                ]
        
        Returns:
            {
                'success': bool,
                'pm25_forecast': float,
                'aqi_forecast': int,
                'forecast_date': str,
                'confidence': str,
                'model_info': dict
            }
        """
        if not self.models_loaded:
            return {
                'success': False,
                'error': 'Models not loaded. Train models first with scripts/train_model.py'
            }
        
        if len(historical_data) < 7:
            return {
                'success': False,
                'error': 'Need at least 7 days of historical data for accurate prediction'
            }
        
        try:
            # Prepare features
            df = self.prepare_features_from_history(historical_data)
            
            # Get latest row for prediction (most recent data)
            latest = df.iloc[-1]
            
            # Extract features in correct order
            X = latest[self.feature_cols].values.reshape(1, -1)
            
            # Handle any missing values
            X = np.nan_to_num(X, nan=np.nanmedian(X))
            
            # Make predictions
            pm25_pred = float(self.pm25_model.predict(X)[0])
            aqi_pred = int(self.aqi_model.predict(X)[0])
            
            # Ensure predictions are non-negative
            pm25_pred = max(0, pm25_pred)
            aqi_pred = max(0, aqi_pred)
            
            # Calculate forecast date (tomorrow)
            last_date = pd.to_datetime(historical_data[-1]['date'])
            forecast_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            
            # Confidence based on R² score
            avg_r2 = (self.metadata['pm25_test_r2'] + self.metadata['aqi_test_r2']) / 2
            if avg_r2 > 0.8:
                confidence = 'high'
            elif avg_r2 > 0.6:
                confidence = 'medium'
            else:
                confidence = 'low'
            
            return {
                'success': True,
                'pm25_forecast': round(pm25_pred, 1),
                'aqi_forecast': aqi_pred,
                'forecast_date': forecast_date,
                'confidence': confidence,
                'model_info': {
                    'pm25_r2': self.metadata['pm25_test_r2'],
                    'aqi_r2': self.metadata['aqi_test_r2'],
                    'trained_at': self.metadata['trained_at']
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Prediction error: {str(e)}'
            }
    
    def batch_predict(self, historical_data: list[dict], days_ahead: int = 7) -> list[dict]:
        """
        Predict multiple days ahead (iterative forecasting)
        
        Args:
            historical_data: Past data
            days_ahead: Number of days to forecast
        
        Returns:
            List of predictions for each day
        """
        predictions = []
        current_data = historical_data.copy()
        
        for day in range(days_ahead):
            result = self.predict_next_day(current_data)
            if not result['success']:
                break
            
            predictions.append(result)
            
            # Add prediction to history for next iteration
            last_date = pd.to_datetime(current_data[-1]['date'])
            new_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            
            # Create new data point with prediction
            new_point = {
                'date': new_date,
                'pm25': result['pm25_forecast'],
                'aqi': result['aqi_forecast'],
                'pm10': current_data[-1].get('pm10', 0),  # Use last known values
                'no': current_data[-1].get('no', 0),
                'no2': current_data[-1].get('no2', 0)
            }
            
            current_data.append(new_point)
        
        return predictions


# Global predictor instance
ml_predictor = WeatherMLPredictor()


if __name__ == "__main__":
    # Test the predictor
    import json
    
    # Example historical data (mock)
    historical_data = [
        {'date': '2024-01-01', 'pm25': 45.2, 'pm10': 89.1, 'aqi': 102, 'no': 12, 'no2': 34},
        {'date': '2024-01-02', 'pm25': 50.3, 'pm10': 95.2, 'aqi': 115, 'no': 15, 'no2': 38},
        {'date': '2024-01-03', 'pm25': 42.1, 'pm10': 85.3, 'aqi': 98, 'no': 11, 'no2': 32},
        {'date': '2024-01-04', 'pm25': 48.5, 'pm10': 92.4, 'aqi': 110, 'no': 14, 'no2': 36},
        {'date': '2024-01-05', 'pm25': 55.2, 'pm10': 102.1, 'aqi': 125, 'no': 18, 'no2': 42},
        {'date': '2024-01-06', 'pm25': 51.8, 'pm10': 98.5, 'aqi': 118, 'no': 16, 'no2': 40},
        {'date': '2024-01-07', 'pm25': 46.3, 'pm10': 88.2, 'aqi': 105, 'no': 13, 'no2': 35},
    ]
    
    predictor = WeatherMLPredictor()
    
    if predictor.models_loaded:
        # Single day prediction
        result = predictor.predict_next_day(historical_data)
        print("\n📊 Next Day Prediction:")
        print(json.dumps(result, indent=2))
        
        # Multi-day prediction
        batch_results = predictor.batch_predict(historical_data, days_ahead=3)
        print("\n📅 3-Day Forecast:")
        for i, pred in enumerate(batch_results, 1):
            print(f"Day {i}: PM2.5={pred['pm25_forecast']}, AQI={pred['aqi_forecast']}")
    else:
        print("❌ Models not loaded. Run scripts/train_model.py first.")
