# Hướng Dẫn Dự Báo Thời Tiết với Machine Learning

## 📋 Tổng Quan

Hệ thống dự báo thời tiết kết hợp **2 phương pháp**:

### A. Real-time API Integration (Đã hoàn thành ✅)
- **OpenWeatherMap**: Dữ liệu thời tiết hiện tại (nhiệt độ, độ ẩm, mô tả)
- **WAQI (World Air Quality Index)**: Chỉ số AQI, PM2.5, PM10 theo thời gian thực
- **Endpoints**:
  - `GET /weather/current?city=Hanoi` - Thời tiết và AQI hiện tại
  - `POST /weather/forecast` - Endpoint dành cho chatbot
  - `GET /weather/health` - Kiểm tra trạng thái API

### B. Machine Learning Forecasting (Vừa hoàn thành ✅)
- **Model**: RandomForest Regressor
- **Dự báo**: PM2.5 và AQI cho ngày tiếp theo (hoặc nhiều ngày)
- **Features**: 30+ features bao gồm lag values, rolling averages, time features
- **Endpoints**:
  - `POST /weather/forecast-ml` - Dự báo ML với dữ liệu lịch sử tùy chỉnh
  - `GET /weather/forecast-ml/batch/{days}` - Dự báo nhiều ngày (demo)

---

## 🚀 Quick Start - Huấn Luyện Model

### Bước 1: Chuẩn bị Dataset
Đảm bảo file `station_day.csv` đã có trong thư mục gốc:

```powershell
# Kiểm tra file
Test-Path station_day.csv
```

Dataset cần có các cột:
- `StationId`, `Date`, `PM2.5`, `PM10`, `NO`, `NO2`, `AQI`, `AQI_Bucket`, v.v.

### Bước 2: Chạy Pipeline Tự Động

**Cách 1: Sử dụng script tự động (KHUYẾN NGHỊ)**
```powershell
.\train_ml.ps1
```

Script này sẽ tự động:
1. Kích hoạt virtual environment
2. Cài đặt dependencies
3. Chạy ETL pipeline (làm sạch dữ liệu)
4. Huấn luyện 2 models (PM2.5 và AQI)
5. Lưu models vào thư mục `models/`
6. Hiển thị metrics (R², MAE, RMSE)

**Cách 2: Chạy từng bước thủ công**
```powershell
# Activate environment
.\.venv\Scripts\Activate.ps1

# Bước 1: ETL - Làm sạch và tạo features
python scripts/etl_pipeline.py

# Bước 2: Train models
python scripts/train_model.py
```

### Bước 3: Kiểm Tra Kết Quả

```powershell
# Xem metadata của models
cat models/metadata.json
```

Output mẫu:
```json
{
  "trained_at": "2024-01-15T10:30:45",
  "feature_count": 35,
  "pm25_test_mae": 12.5,
  "pm25_test_r2": 0.85,
  "aqi_test_mae": 15.2,
  "aqi_test_r2": 0.82,
  "model_type": "RandomForestRegressor",
  "n_estimators": 100,
  "max_depth": 20
}
```

**Giải thích metrics**:
- **R² (R-squared)**: Độ chính xác (0-1), càng gần 1 càng tốt
  - > 0.8: Rất tốt
  - 0.6-0.8: Tốt
  - < 0.6: Cần cải thiện
- **MAE (Mean Absolute Error)**: Sai số trung bình (càng nhỏ càng tốt)

---

## 📊 Sử Dụng ML Forecasting API

### 1. Khởi động server
```powershell
.\start.ps1
```

Đợi cho đến khi thấy:
```
INFO:     Application startup complete.
✅ Models loaded successfully!
📊 PM2.5 R²: 0.850, AQI R²: 0.820
```

### 2. Truy cập API Documentation
Mở trình duyệt: http://localhost:8000/docs

### 3. Test ML Endpoint

**Endpoint**: `POST /weather/forecast-ml`

**Request Body** (JSON):
```json
{
  "historical_data": [
    {
      "date": "2024-01-01",
      "pm25": 45.2,
      "pm10": 89.1,
      "aqi": 102,
      "no": 12.5,
      "no2": 34.2,
      "nox": 46.7,
      "nh3": 5.3,
      "co": 0.8,
      "so2": 8.1,
      "o3": 45.2,
      "benzene": 2.1,
      "toluene": 3.5,
      "xylene": 1.8
    },
    {
      "date": "2024-01-02",
      "pm25": 50.3,
      "pm10": 95.2,
      "aqi": 115,
      "no": 15.2,
      "no2": 38.1
    },
    ... (ít nhất 7 ngày dữ liệu)
  ],
  "days_ahead": 1
}
```

**Response**:
```json
{
  "success": true,
  "forecast_date": "2024-01-08",
  "pm25_forecast": 52.3,
  "aqi_forecast": 118,
  "aqi_category": "Moderate",
  "confidence": "high",
  "recommendation": "🟡 Không khí ở mức trung bình. Người nhạy cảm nên hạn chế hoạt động ngoài trời kéo dài.",
  "model_info": {
    "pm25_r2": 0.85,
    "aqi_r2": 0.82,
    "trained_at": "2024-01-15T10:30:45"
  }
}
```

### 4. Test với PowerShell

```powershell
# Dự báo 1 ngày
$body = @{
    historical_data = @(
        @{date="2024-01-01"; pm25=45.2; pm10=89.1; aqi=102; no=12; no2=34},
        @{date="2024-01-02"; pm25=50.3; pm10=95.2; aqi=115; no=15; no2=38},
        @{date="2024-01-03"; pm25=42.1; pm10=85.3; aqi=98; no=11; no2=32},
        @{date="2024-01-04"; pm25=48.5; pm10=92.4; aqi=110; no=14; no2=36},
        @{date="2024-01-05"; pm25=55.2; pm10=102.1; aqi=125; no=18; no2=42},
        @{date="2024-01-06"; pm25=51.8; pm10=98.5; aqi=118; no=16; no2=40},
        @{date="2024-01-07"; pm25=46.3; pm10=88.2; aqi=105; no=13; no2=35}
    )
    days_ahead = 1
} | ConvertTo-Json -Depth 10

Invoke-RestMethod -Uri "http://localhost:8000/weather/forecast-ml" `
    -Method POST `
    -Body $body `
    -ContentType "application/json"
```

### 5. Batch Forecast (Demo)

**Endpoint**: `GET /weather/forecast-ml/batch/7`

Dự báo 7 ngày tiếp theo (sử dụng mock data):

```powershell
curl http://localhost:8000/weather/forecast-ml/batch/7
```

Response:
```json
{
  "success": true,
  "forecasts": [
    {
      "date": "2024-01-08",
      "pm25": 52.3,
      "aqi": 118,
      "category": "Moderate",
      "confidence": "high"
    },
    {
      "date": "2024-01-09",
      "pm25": 54.8,
      "aqi": 122,
      "category": "Moderate",
      "confidence": "high"
    },
    ...
  ],
  "model_info": { ... }
}
```

---

## 🔧 Chi Tiết Kỹ Thuật

### ETL Pipeline (`scripts/etl_pipeline.py`)

**Chức năng**:
1. **Load data**: Đọc `station_day.csv`, parse Date
2. **Feature Engineering**:
   - Time features: Month, Day, DayOfWeek, Quarter, WeekOfYear, is_weekend, is_winter
   - Lag features: PM2.5_lag1, PM2.5_lag3, PM2.5_lag7, AQI_lag1
   - Rolling statistics: PM2.5_ma3, PM2.5_ma7, PM2.5_ma30, PM2.5_std7
   - Pollutant ratios: PM_ratio, NOx_total
3. **Handle missing values**: 
   - Fillna by station median
   - Global median fallback
   - Drop rows với target null
4. **Save cleaned data**: `data/cleaned/station_day_clean.csv`

**Output**:
- Original: 108,037 rows × 16 columns
- After ETL: ~107,000 rows × 35+ columns

### Model Training (`scripts/train_model.py`)

**Algorithm**: RandomForestRegressor
- `n_estimators=100` (100 decision trees)
- `max_depth=20` (maximum tree depth)
- `min_samples_split=10` (minimum samples to split)
- `min_samples_leaf=5` (minimum samples in leaf)
- `random_state=42` (reproducibility)
- `n_jobs=-1` (use all CPU cores)

**Train/Test Split**:
- 80% train, 20% test
- `shuffle=False` (preserve time series order)

**Models**:
1. **PM2.5 Forecast Model**: Dự đoán nồng độ PM2.5 ngày tiếp theo
2. **AQI Forecast Model**: Dự đoán chỉ số AQI ngày tiếp theo

**Saved Files**:
- `models/pm25_forecast.pkl` - PM2.5 model (joblib)
- `models/aqi_forecast.pkl` - AQI model (joblib)
- `models/feature_columns.pkl` - Feature names (để đảm bảo order)
- `models/metadata.json` - Metrics và thông tin training

### ML Predictor (`app/ml_predictor.py`)

**Class**: `WeatherMLPredictor`

**Methods**:
- `load_models()` - Load trained models
- `prepare_features_from_history()` - Convert historical data to features
- `predict_next_day()` - Dự báo 1 ngày
- `batch_predict()` - Dự báo nhiều ngày (iterative)

**Features**:
- Tự động load models khi khởi động FastAPI
- Graceful fallback nếu models chưa train
- Validation input (cần ít nhất 7 ngày lịch sử)
- Confidence scoring dựa trên R² test score

---

## 🔗 Tích Hợp với Zalo Chatbot

### Workflow n8n

Cập nhật workflow để sử dụng ML forecast:

```json
{
  "nodes": [
    {
      "name": "Webhook - Zalo",
      "type": "n8n-nodes-base.webhook"
    },
    {
      "name": "Parse Message",
      "type": "n8n-nodes-base.function",
      "javascript": "
        const text = $input.item.json.message.text.toLowerCase();
        let endpoint = '/weather/current';
        
        // Nếu user hỏi dự báo -> dùng ML
        if (text.includes('dự báo') || text.includes('ngày mai')) {
          endpoint = '/weather/forecast-ml/batch/1';
        }
        
        return [{ endpoint, city: extractCity(text) }];
      "
    },
    {
      "name": "Call Weather API",
      "type": "n8n-nodes-base.httpRequest",
      "url": "http://ai-assistant:8000/weather{{ $json.endpoint }}"
    },
    {
      "name": "Format Response",
      "type": "n8n-nodes-base.function",
      "javascript": "
        const data = $input.item.json;
        let message = '';
        
        if (data.success && data.pm25_forecast) {
          // ML forecast
          message = `
🔮 Dự báo thời tiết ngày mai (${data.forecast_date}):

💨 PM2.5: ${data.pm25_forecast} µg/m³
🌫️ AQI: ${data.aqi_forecast} (${data.aqi_category})
📊 Độ tin cậy: ${data.confidence === 'high' ? 'Cao ✅' : 'Trung bình ⚠️'}

${data.recommendation}

🤖 Dự báo từ AI Model (R²: ${(data.model_info.aqi_r2 * 100).toFixed(1)}%)
          `;
        } else {
          // Real-time API
          message = `
🌤️ Thời tiết hiện tại tại ${data.city}:

🌡️ Nhiệt độ: ${data.temperature}°C (cảm giác ${data.feels_like}°C)
💧 Độ ẩm: ${data.humidity}%
☁️ Tình trạng: ${data.description}

💨 PM2.5: ${data.pm25} µg/m³
🌫️ AQI: ${data.aqi} (${data.aqi_category})

${data.recommendation}

📡 Dữ liệu real-time từ OpenWeatherMap & WAQI
          `;
        }
        
        return [{ message }];
      "
    },
    {
      "name": "Save to Google Drive",
      "type": "n8n-nodes-base.httpRequest",
      "method": "POST",
      "url": "http://ai-assistant:8000/generate",
      "body": {
        "prompt": "Weather forecast log",
        "user_id": "={{ $json.user_id }}"
      }
    },
    {
      "name": "Reply to Zalo",
      "type": "n8n-nodes-base.httpRequest",
      "method": "POST",
      "url": "https://openapi.zalo.me/v2.0/oa/message",
      "body": {
        "recipient": { "user_id": "={{ $json.user_id }}" },
        "message": { "text": "={{ $json.message }}" }
      }
    }
  ]
}
```

### Ví dụ User Flow

**User**: "Dự báo thời tiết Hà Nội ngày mai"

**n8n Workflow**:
1. Nhận webhook từ Zalo
2. Parse message → phát hiện "dự báo" và "ngày mai"
3. Call `POST /weather/forecast-ml` (hoặc batch/1)
4. Format response với emoji và tiếng Việt
5. Lưu log vào Google Drive
6. Gửi reply về Zalo

**Zalo Reply**:
```
🔮 Dự báo thời tiết ngày mai (2024-01-16):

💨 PM2.5: 52.3 µg/m³
🌫️ AQI: 118 (Moderate)
📊 Độ tin cậy: Cao ✅

🟡 Không khí ở mức trung bình. Người nhạy cảm nên hạn chế hoạt động ngoài trời kéo dài.

🤖 Dự báo từ AI Model (R²: 82.0%)
```

---

## 🎯 Best Practices

### 1. Cập nhật Model định kỳ
- Train lại model hàng tuần/tháng với dữ liệu mới
- So sánh metrics (R², MAE) trước và sau
- Backup models cũ trước khi overwrite

```powershell
# Backup old models
Copy-Item models models_backup_$(Get-Date -Format 'yyyyMMdd') -Recurse

# Retrain
.\train_ml.ps1
```

### 2. Monitor Model Performance
- Log predictions và actual values
- Tính MAE/RMSE on production data
- Alert nếu accuracy giảm

### 3. Hybrid Approach
- Sử dụng **API** cho dữ liệu hiện tại
- Sử dụng **ML** cho dự báo tương lai
- Kết hợp cả 2 để tăng độ tin cậy

### 4. Error Handling
- Fallback to API nếu ML model fail
- Validate input data (check nulls, outliers)
- Return user-friendly error messages

---

## 🐛 Troubleshooting

### Lỗi: "Models not loaded"
**Nguyên nhân**: Chưa train models

**Giải pháp**:
```powershell
.\train_ml.ps1
```

### Lỗi: "Need at least 7 days of historical data"
**Nguyên nhân**: Request body không đủ dữ liệu

**Giải pháp**: Gửi ít nhất 7 ngày dữ liệu trong `historical_data`

### Lỗi: File "station_day.csv" not found
**Nguyên nhân**: Dataset không có trong root directory

**Giải pháp**: 
```powershell
# Download dataset hoặc copy vào thư mục gốc
Copy-Item "đường_dẫn/station_day.csv" .
```

### Model accuracy thấp (R² < 0.6)
**Nguyên nhân**: 
- Dữ liệu không đủ/không tốt
- Hyperparameters chưa tối ưu

**Giải pháp**:
1. Kiểm tra dữ liệu (missing values, outliers)
2. Tăng `n_estimators` (100 → 200)
3. Thử GridSearchCV để tìm best params
4. Thu thập thêm dữ liệu

### Import errors khi chạy API
**Nguyên nhân**: Dependencies chưa cài

**Giải pháp**:
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 📚 Tài Liệu Tham Khảo

- **RandomForest**: [scikit-learn docs](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- **Time Series Forecasting**: [Practical Guide](https://machinelearningmastery.com/time-series-forecasting/)
- **AQI Standards**: [AirNow.gov](https://www.airnow.gov/aqi/aqi-basics/)
- **Feature Engineering**: [Feature Engineering Book](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)

---

## ✅ Checklist Hoàn Thành

- [x] ETL pipeline (làm sạch dữ liệu, feature engineering)
- [x] Train PM2.5 model (RandomForest)
- [x] Train AQI model (RandomForest)
- [x] Save models với joblib
- [x] ML predictor utility class
- [x] FastAPI endpoints (`/forecast-ml`, `/batch`)
- [x] Integration với weather router
- [x] PowerShell script tự động (`train_ml.ps1`)
- [x] Documentation đầy đủ
- [ ] n8n workflow update (pending)
- [ ] Production deployment (pending)
- [ ] Model monitoring dashboard (future)

---

**Tác giả**: GitHub Copilot  
**Ngày cập nhật**: 2024-01-15  
**Phiên bản**: 1.0
