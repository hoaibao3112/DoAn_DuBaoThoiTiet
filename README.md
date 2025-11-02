# Hệ Thống Dự Báo Thời Tiết & AQI với Zalo Chatbot

Ứng dụng dự báo thời tiết và chất lượng không khí tích hợp Zalo, kết hợp **Real-time APIs** và **Machine Learning**.

## 🌟 Tính Năng

### 1. Real-time Weather & AQI 🌤️
- **OpenWeatherMap**: Nhiệt độ, độ ẩm, tình trạng thời tiết
- **WAQI**: Chỉ số AQI, PM2.5, PM10 real-time
- **Smart Recommendations**: Khuyến nghị sức khỏe tiếng Việt

### 2. Machine Learning Forecasting 🤖
- **RandomForest Models**: Dự báo PM2.5 và AQI
- **Dataset**: 108K records từ 75 trạm quan trắc
- **Accuracy**: R² > 0.8

### 3. Zalo Chatbot 💬
- Tự động xử lý tin nhắn qua n8n
- Phản hồi tiếng Việt với emoji
- Lưu log vào Google Drive

---

## 🚀 Quick Start

### Bước 1: Cài đặt Dependencies
```powershell
# Tạo virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install packages
pip install -r requirements.txt
```

### Bước 2: Cấu hình API Keys
Copy `.env.example` thành `.env` và điền keys:

```env
# Weather APIs
OPENWEATHER_API_KEY=your_key        # openweathermap.org/api
WAQI_TOKEN=your_token               # aqicn.org/data-platform/token/

# Zalo
ZALO_ACCESS_TOKEN=your_token        # developers.zalo.me

# Google Drive
GDRIVE_FOLDER_ID=your_folder_id
GOOGLE_APPLICATION_CREDENTIALS=/secrets/service_account.json
```

### Bước 3: Train ML Models
```powershell
# Đảm bảo có station_day.csv
Test-Path station_day.csv

# Chạy training (3-5 phút)
.\train_ml.ps1
```

Kết quả:
- ✅ `models/pm25_forecast.pkl`
- ✅ `models/aqi_forecast.pkl`
- ✅ R² scores hiển thị

### Bước 4: Start Services
```powershell
# Khởi động Docker containers
.\start.ps1
```

Truy cập:
- 📖 API Docs: http://localhost:8000/docs
- 🔧 n8n: http://localhost:5678

### Bước 5: Test System
```powershell
.\test_weather.ps1
```

---

## 📊 API Endpoints

### Real-time Weather
```bash
# Current weather + AQI
GET /weather/current?city=Hanoi

# Chatbot endpoint
POST /weather/forecast
{
  "city": "Hanoi",
  "user_id": "123"
}
```

### ML Forecasting
```bash
# Custom forecast
POST /weather/forecast-ml
{
  "historical_data": [
    {"date": "2024-01-01", "pm25": 45.2, "aqi": 102, ...},
    ...
  ],
  "days_ahead": 1
}

# Batch forecast (demo)
GET /weather/forecast-ml/batch/3
```

---

## 🏗️ Architecture

```
User (Zalo) 
    ↓
n8n Workflow Engine
    ↓
FastAPI Service
    ├─ Real-time APIs (OpenWeatherMap + WAQI)
    └─ ML Models (RandomForest)
    ↓
Response → Google Drive (log) + Zalo (reply)
```

---

## 📂 Project Structure

```
Doan_PTDL/
├── app/
│   ├── main.py              # FastAPI app
│   ├── weather.py           # Weather endpoints
│   ├── ml_predictor.py      # ML model loader
│   ├── gdrive.py            # Google Drive
│   └── zalo.py              # Zalo messaging
├── scripts/
│   ├── etl_pipeline.py      # Data cleaning
│   └── train_model.py       # Model training
├── models/                  # Trained models (generated)
├── n8n-workflows/           # n8n workflow JSON
├── docker-compose.yml       # Docker setup
├── requirements.txt         # Python deps
├── station_day.csv          # Dataset (108K records)
├── train_ml.ps1             # ML automation
├── test_weather.ps1         # Testing
└── start.ps1, stop.ps1      # Docker management
```

---

## 🔧 Configuration

### n8n Workflow
1. Mở http://localhost:5678
2. Import `n8n-workflows/Zalo_AI_Assistant.json`
3. Configure credentials (Zalo, Google Drive)
4. Activate workflow
5. Paste webhook URL vào Zalo OA settings

### Google Service Account
1. Tạo service account tại console.cloud.google.com
2. Enable Google Drive API
3. Download JSON key
4. Đặt vào `secrets/service_account.json`
5. Share Drive folder với service account email

---

## 📖 Documentation

- **ML_FORECASTING_GUIDE.md** - Chi tiết ML pipeline, training, API usage
- **QUY_TRINH_XU_LY_DU_LIEU.md** - Quy trình xử lý dữ liệu 6 giai đoạn
- **API Docs**: http://localhost:8000/docs (Swagger UI)

---

## 🎯 Example Usage

**Real-time Weather:**
```
User: "Thời tiết Hà Nội"
Bot:  🌤️ Thời tiết hiện tại tại Hanoi:
      🌡️ Nhiệt độ: 18.5°C
      💧 Độ ẩm: 75%
      💨 PM2.5: 42.3 µg/m³
      🌫️ AQI: 102 (Moderate)
      🟡 Không khí ở mức trung bình...
```

**ML Forecast:**
```
User: "Dự báo ngày mai"
Bot:  🔮 Dự báo thời tiết ngày mai:
      💨 PM2.5: 52.3 µg/m³
      🌫️ AQI: 118 (Moderate)
      📊 Độ tin cậy: Cao ✅
      🤖 Dự báo từ AI Model (R²: 82.1%)
```

---

## 🐛 Troubleshooting

### Models not loaded
```powershell
# Train lại models
.\train_ml.ps1
```

### API keys không hoạt động
- Kiểm tra `.env` có đúng keys không
- Test API trực tiếp:
  ```bash
  curl "https://api.openweathermap.org/data/2.5/weather?q=Hanoi&appid=YOUR_KEY"
  ```

### Docker không start
```powershell
# Check Docker Desktop đang chạy
docker version

# Xem logs
docker-compose logs ai-assistant
docker-compose logs n8n
```

### Import errors
```powershell
# Reinstall dependencies
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt --upgrade
```

---

## 📈 Model Performance

| Model | Test R² | Test MAE | Features |
|-------|---------|----------|----------|
| PM2.5 | 0.847 | 12.34 µg/m³ | 35 |
| AQI | 0.821 | 15.67 | 35 |

**Top Features:**
- PM2.5_lag1 (45%)
- PM2.5_ma7 (19%)
- AQI_lag1 (8%)

---

## 🔐 Security

- Không commit `.env` hoặc `secrets/` vào Git
- Sử dụng `.env.example` làm template
- Rotate API keys định kỳ
- Service Account với least privilege

---

## 📞 Support

Nếu gặp vấn đề:
1. Đọc **ML_FORECASTING_GUIDE.md** (troubleshooting section)
2. Chạy `.\test_weather.ps1` để kiểm tra
3. Check logs: `docker-compose logs`

---

## 📄 License

MIT License

---

**Made with ❤️ and ☕**

*Dự báo chính xác, sống khỏe mạnh! 🌤️🌱*
