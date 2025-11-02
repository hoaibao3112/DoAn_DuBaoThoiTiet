# 📊 QUY TRÌNH XỬ LÝ DỮ LIỆU - DỰ ÁN CHẤT LƯỢNG KHÔNG KHÍ (AQI)

## 🎯 Tổng quan dữ liệu

**Dataset:** `station_day.csv` - Dữ liệu chất lượng không khí theo ngày từ các trạm quan trắc

**Thông tin:**
- **108,037 records** (hơn 100k dòng dữ liệu)
- **16 cột:** StationId, Date, PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene, AQI, AQI_Bucket
- **Thời gian:** 2017-11-24 đến 2018-07-11
- **Các trạm:** AP001, và nhiều trạm khác (cần khám phá)

**Chất lượng không khí (AQI_Bucket):**
- Good (Tốt): 0-50
- Satisfactory (Khá): 51-100  
- Moderate (Trung bình): 101-200
- Poor (Kém): 201-300
- Very Poor (Rất kém): 301-400
- Severe (Nguy hiểm): 401+

---

## 🔄 QUY TRÌNH XỬ LÝ DỮ LIỆU HOÀN CHỈNH

### Phase 1: THU THẬP & LÀM SẠCH DỮ LIỆU (ETL)

#### 1.1. Extract (Trích xuất)
```python
# Đọc CSV, parse dates, xử lý encoding
import pandas as pd

df = pd.read_csv('station_day.csv', parse_dates=['Date'])
```

**Nhiệm vụ:**
- ✅ Load CSV vào pandas DataFrame
- ✅ Parse Date column thành datetime
- ✅ Kiểm tra dtypes của từng cột
- ✅ Xác định số lượng trạm (unique StationId)
- ✅ Phân tích khoảng thời gian dữ liệu

#### 1.2. Transform (Biến đổi)

**A. Xử lý Missing Values**
```python
# Kiểm tra missing
missing_summary = df.isnull().sum()

# Strategies:
# - Fillna với median/mean cho pollutants (PM2.5, PM10, etc.)
# - Interpolate theo time series (ffill/bfill)
# - Drop rows nếu AQI missing (target variable)
```

**B. Feature Engineering**
```python
# Tạo features mới từ Date
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Quarter'] = df['Date'].dt.quarter
df['WeekOfYear'] = df['Date'].dt.isocalendar().week

# Tạo rolling statistics (trung bình 7 ngày, 30 ngày)
df['PM2.5_MA7'] = df.groupby('StationId')['PM2.5'].transform(lambda x: x.rolling(7, min_periods=1).mean())
df['PM2.5_MA30'] = df.groupby('StationId')['PM2.5'].transform(lambda x: x.rolling(30, min_periods=1).mean())

# Lag features (giá trị ngày hôm trước)
df['PM2.5_lag1'] = df.groupby('StationId')['PM2.5'].shift(1)
df['PM2.5_lag7'] = df.groupby('StationId')['PM2.5'].shift(7)

# Tỉ lệ các chất ô nhiễm
df['PM_ratio'] = df['PM2.5'] / (df['PM10'] + 1)  # +1 tránh chia 0
df['NOx_total'] = df['NO'] + df['NO2']

# Binary features
df['is_weekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
df['is_winter'] = df['Month'].isin([11, 12, 1, 2]).astype(int)
```

**C. Outlier Detection**
```python
from scipy import stats

# Z-score method
z_scores = np.abs(stats.zscore(df[['PM2.5', 'PM10', 'AQI']].fillna(0)))
df['is_outlier'] = (z_scores > 3).any(axis=1)

# IQR method
Q1 = df['PM2.5'].quantile(0.25)
Q3 = df['PM2.5'].quantile(0.75)
IQR = Q3 - Q1
df['PM2.5_outlier'] = ((df['PM2.5'] < (Q1 - 1.5 * IQR)) | (df['PM2.5'] > (Q3 + 1.5 * IQR)))
```

#### 1.3. Load (Lưu trữ)
```python
# Lưu cleaned data
df_clean.to_csv('data/cleaned/station_day_clean.csv', index=False)

# Hoặc lưu vào database
# df_clean.to_sql('air_quality', con=engine, if_exists='replace')
```

---

### Phase 2: PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (EDA)

#### 2.1. Descriptive Statistics
```python
# Thống kê mô tả
print(df.describe())

# Phân bố AQI_Bucket
print(df['AQI_Bucket'].value_counts())

# Correlation matrix
import seaborn as sns
corr_matrix = df[['PM2.5', 'PM10', 'NO2', 'O3', 'AQI']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
```

#### 2.2. Visualization Insights

**A. Time Series Analysis**
- Xu hướng PM2.5/AQI theo thời gian
- Seasonal patterns (mùa đông ô nhiễm cao hơn?)
- Weekly patterns (cuối tuần khác ngày thường?)

**B. Geographic Analysis**
- So sánh AQI giữa các trạm
- Trạm nào ô nhiễm nhất/sạch nhất?
- Heatmap theo vị trí (nếu có lat/lon)

**C. Pollutant Relationships**
- PM2.5 vs PM10 correlation
- NO2 impact on AQI
- Benzene/Toluene levels

**D. Alert Thresholds**
- Số ngày AQI > 200 (Poor/Very Poor)
- Frequency của AQI_Bucket levels
- Identify pollution spikes

---

### Phase 3: MÔ HÌNH DỰ BÁO (MACHINE LEARNING)

#### 3.1. Bài toán dự báo

**Option A: Regression - Dự báo AQI số (liên tục)**
- Input: PM2.5, PM10, NO2, date features, lag features
- Output: AQI value (0-500)
- Models: Linear Regression, Random Forest, XGBoost, LSTM

**Option B: Classification - Dự báo AQI_Bucket (phân loại)**
- Input: tương tự
- Output: Good/Satisfactory/Moderate/Poor/Very Poor/Severe
- Models: Logistic Regression, Random Forest, XGBoost, Neural Network

**Option C: Time Series Forecasting - Dự báo ngày mai**
- Input: lịch sử 7-30 ngày trước
- Output: PM2.5/AQI ngày mai
- Models: ARIMA, Prophet, LSTM/GRU

#### 3.2. Model Pipeline

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Prepare data
features = ['PM2.5', 'PM10', 'NO2', 'O3', 'Month', 'DayOfWeek', 
            'PM2.5_MA7', 'PM2.5_lag1']
X = df[features].fillna(0)
y = df['AQI'].fillna(0)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}, R²: {r2:.3f}")

# Save model
import joblib
joblib.dump(model, 'models/aqi_predictor.pkl')
```

#### 3.3. Feature Importance
```python
# Xem feature nào quan trọng nhất
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': features,
    'importance': importances
}).sort_values('importance', ascending=False)

print(feature_importance_df)
```

---

### Phase 4: API DỰ BÁO (FASTAPI SERVICE)

#### 4.1. FastAPI Endpoints

**File: `app/aqi_predictor.py`**
```python
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="AQI Prediction API")

# Load model
model = joblib.load('models/aqi_predictor.pkl')

class AQIRequest(BaseModel):
    pm25: float
    pm10: float
    no2: float
    o3: float
    month: int
    day_of_week: int

class AQIResponse(BaseModel):
    predicted_aqi: float
    aqi_category: str
    health_impact: str
    recommendation: str

def get_aqi_category(aqi: float) -> str:
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Satisfactory"
    elif aqi <= 200: return "Moderate"
    elif aqi <= 300: return "Poor"
    elif aqi <= 400: return "Very Poor"
    else: return "Severe"

def get_health_impact(category: str) -> str:
    impacts = {
        "Good": "Minimal impact",
        "Satisfactory": "Minor breathing discomfort to sensitive people",
        "Moderate": "Breathing discomfort to people with lung, asthma, heart disease",
        "Poor": "Breathing discomfort to most people on prolonged exposure",
        "Very Poor": "Respiratory illness on prolonged exposure",
        "Severe": "Affects healthy people and seriously impacts those with existing diseases"
    }
    return impacts.get(category, "Unknown")

def get_recommendation(category: str) -> str:
    recs = {
        "Good": "Air quality is satisfactory. Enjoy outdoor activities!",
        "Satisfactory": "Sensitive individuals should consider limiting outdoor activities.",
        "Moderate": "People with respiratory conditions should reduce outdoor activities.",
        "Poor": "Avoid outdoor activities. Wear N95 mask if going outside.",
        "Very Poor": "Stay indoors. Use air purifiers. Avoid physical activities.",
        "Severe": "Medical emergency! Stay indoors with doors/windows closed."
    }
    return recs.get(category, "Unknown")

@app.post("/predict", response_model=AQIResponse)
def predict_aqi(request: AQIRequest):
    # Prepare input
    input_data = pd.DataFrame([{
        'PM2.5': request.pm25,
        'PM10': request.pm10,
        'NO2': request.no2,
        'O3': request.o3,
        'Month': request.month,
        'DayOfWeek': request.day_of_week,
        'PM2.5_MA7': request.pm25,  # simplified
        'PM2.5_lag1': request.pm25  # simplified
    }])
    
    # Predict
    prediction = model.predict(input_data)[0]
    category = get_aqi_category(prediction)
    
    return AQIResponse(
        predicted_aqi=round(prediction, 2),
        aqi_category=category,
        health_impact=get_health_impact(category),
        recommendation=get_recommendation(category)
    )

@app.get("/station/{station_id}/historical")
def get_historical_data(station_id: str, start_date: str, end_date: str):
    """Lấy dữ liệu lịch sử của một trạm"""
    df = pd.read_csv('data/cleaned/station_day_clean.csv', parse_dates=['Date'])
    filtered = df[
        (df['StationId'] == station_id) & 
        (df['Date'] >= start_date) & 
        (df['Date'] <= end_date)
    ]
    return filtered.to_dict(orient='records')

@app.get("/stations/worst")
def get_worst_stations(limit: int = 10):
    """Top trạm có AQI cao nhất"""
    df = pd.read_csv('data/cleaned/station_day_clean.csv')
    worst = df.groupby('StationId')['AQI'].mean().sort_values(ascending=False).head(limit)
    return worst.to_dict()
```

#### 4.2. Test API
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pm25": 120.5,
    "pm10": 180.3,
    "no2": 45.2,
    "o3": 60.1,
    "month": 12,
    "day_of_week": 1
  }'
```

---

### Phase 5: CẢNH BÁO TỰ ĐỘNG (n8n WORKFLOW)

#### 5.1. Use Cases

**A. Daily AQI Report**
- n8n schedule trigger (mỗi sáng 7AM)
- Gọi API `/predict` với dữ liệu mới nhất
- Gửi báo cáo qua:
  - Zalo message
  - Email
  - Telegram
  - SMS (Twilio)

**B. Alert When AQI > Threshold**
- n8n webhook nhận real-time sensor data
- Gọi `/predict` để dự báo AQI
- Nếu AQI > 200 (Poor):
  - Gửi cảnh báo khẩn cấp
  - Lưu vào database
  - Notify qua nhiều channels

**C. Weekly Trend Report**
- n8n schedule (mỗi Chủ nhật)
- Gọi `/station/*/historical` để lấy data tuần
- Tạo visualization (chart)
- Gửi báo cáo PDF qua email

**D. Data Pipeline Automation**
- n8n FTP/HTTP trigger (nhận file CSV mới)
- Chạy ETL script (Python)
- Retrain model nếu data đủ lớn
- Deploy model mới lên production
- Notify team qua Slack

#### 5.2. n8n Workflow Example: Daily AQI Alert

**Nodes:**
1. **Schedule Trigger** - Chạy mỗi ngày 7AM
2. **HTTP Request** - Lấy dữ liệu mới nhất từ sensor API
3. **HTTP Request** - POST `/predict` với data
4. **Function** - Parse response, format message
5. **IF** - Check if AQI > 150
   - **YES branch:**
     - **Zalo Message** - Gửi cảnh báo
     - **Email** - Gửi cho danh sách subscribers
     - **Google Sheets** - Log vào spreadsheet
   - **NO branch:**
     - **Telegram** - Gửi tin nhẹ nhàng
6. **Set** - Log execution status

---

### Phase 6: DASHBOARD & VISUALIZATION

#### 6.1. Streamlit Dashboard

**File: `dashboard/app.py`**
```python
import streamlit as st
import pandas as pd
import plotly.express as px
import requests

st.set_page_config(page_title="AQI Monitor", layout="wide")

st.title("🌍 Air Quality Index Dashboard")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('data/cleaned/station_day_clean.csv', parse_dates=['Date'])

df = load_data()

# Sidebar filters
station = st.sidebar.selectbox("Select Station", df['StationId'].unique())
date_range = st.sidebar.date_input("Date Range", [df['Date'].min(), df['Date'].max()])

# Filter data
filtered = df[
    (df['StationId'] == station) & 
    (df['Date'] >= pd.to_datetime(date_range[0])) & 
    (df['Date'] <= pd.to_datetime(date_range[1]))
]

# Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Average AQI", f"{filtered['AQI'].mean():.1f}")
col2.metric("Max PM2.5", f"{filtered['PM2.5'].max():.1f}")
col3.metric("Days > 200 AQI", len(filtered[filtered['AQI'] > 200]))
col4.metric("Current Category", filtered.iloc[-1]['AQI_Bucket'] if len(filtered) > 0 else "N/A")

# Time series chart
fig = px.line(filtered, x='Date', y='AQI', title='AQI Trend')
st.plotly_chart(fig, use_container_width=True)

# Pollutant comparison
fig2 = px.box(filtered, y=['PM2.5', 'PM10', 'NO2', 'O3'], title='Pollutant Distribution')
st.plotly_chart(fig2, use_container_width=True)

# Prediction section
st.header("🔮 AQI Prediction")
col1, col2 = st.columns(2)
pm25 = col1.number_input("PM2.5", value=100.0)
pm10 = col2.number_input("PM10", value=150.0)

if st.button("Predict AQI"):
    response = requests.post(
        "http://localhost:8000/predict",
        json={
            "pm25": pm25,
            "pm10": pm10,
            "no2": 40.0,
            "o3": 50.0,
            "month": 12,
            "day_of_week": 1
        }
    )
    result = response.json()
    st.success(f"Predicted AQI: {result['predicted_aqi']} - {result['aqi_category']}")
    st.info(result['recommendation'])
```

#### 6.2. Run Dashboard
```bash
streamlit run dashboard/app.py
```

---

## 🎯 TRIỂN KHAI VÀ SỬ DỤNG

### Cấu trúc thư mục đề xuất

```
Doan_PTDL/
├── data/
│   ├── raw/
│   │   └── station_day.csv
│   └── cleaned/
│       └── station_day_clean.csv
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   └── 03_Modeling.ipynb
├── app/
│   ├── main.py               # FastAPI main (Zalo AI)
│   ├── aqi_predictor.py      # AQI prediction endpoints
│   ├── etl.py                # Data processing functions
│   └── models/
│       └── aqi_predictor.pkl
├── dashboard/
│   └── app.py                # Streamlit dashboard
├── n8n-workflows/
│   ├── Zalo_AI_Assistant.json
│   └── Daily_AQI_Alert.json
├── scripts/
│   ├── train_model.py
│   └── update_data.py
├── docker-compose.yml
├── requirements.txt
└── README.md
```

### Workflow hoàn chỉnh

```
1. Dữ liệu mới → CSV file
   ↓
2. n8n trigger → Run ETL script
   ↓
3. ETL: Clean + Feature Engineering
   ↓
4. Save to cleaned/station_day_clean.csv
   ↓
5. Retrain model (nếu cần)
   ↓
6. Deploy model → FastAPI
   ↓
7. n8n schedule → Daily prediction
   ↓
8. If AQI > threshold → Alert (Zalo/Email/SMS)
   ↓
9. Dashboard update real-time
```

---

## 💡 ĐỀ XUẤT TÍNH NĂNG HAY VÀ HỮU ÍCH

### 1. **Real-time AQI Monitor + Alert System**
**Mô tả:** Hệ thống giám sát và cảnh báo chất lượng không khí tự động.

**Luồng hoạt động:**
- Sensor/API gửi dữ liệu real-time → n8n webhook
- n8n gọi FastAPI `/predict` → Dự báo AQI
- Nếu AQI > ngưỡng cảnh báo:
  - Gửi Zalo message cho người dùng đăng ký
  - Gửi email/SMS
  - Push notification (qua Firebase)
  - Cập nhật dashboard
  - Lưu vào database để phân tích sau

**Giá trị:**
- Bảo vệ sức khỏe: Người dùng biết khi nào nên ở trong nhà
- Proactive: Cảnh báo trước khi AQI trở nên nguy hiểm
- Multi-channel: Đảm bảo người dùng nhận được thông báo

### 2. **Personalized Health Recommendations**
**Mô tả:** Đề xuất cá nhân hoá dựa trên AQI và profile sức khỏe.

**Profile người dùng:**
- Tuổi, giới tính
- Tình trạng sức khỏe (hen suyễn, COPD, tim mạch)
- Vị trí (trạm nào gần nhất)
- Hoạt động thường ngày (chạy bộ, đạp xe)

**Recommendations:**
- AQI < 50: "Tuyệt vời! Bạn có thể chạy bộ ngoài trời."
- AQI 100-150 + Asthma: "Nên hạn chế hoạt động ngoài trời. Mang theo thuốc xịt."
- AQI > 200: "Nguy hiểm! Ở trong nhà, đóng cửa sổ, bật máy lọc không khí."

### 3. **Comparative Station Analysis**
**Mô tả:** So sánh chất lượng không khí giữa các khu vực.

**Features:**
- Map view: Heatmap AQI theo vị trí địa lý
- Ranking: Top 10 trạm sạch nhất/ô nhiễm nhất
- Trend comparison: So sánh xu hướng giữa 2-3 trạm
- Best time to visit: "Khu vực X sạch nhất vào buổi sáng"

**Use case:**
- Chọn nơi ở: Người mua nhà xem khu nào không khí tốt nhất
- Lập kế hoạch du lịch: Tránh khu vực ô nhiễm
- Quyết định đi làm: Chọn route ít ô nhiễm

### 4. **7-Day AQI Forecast**
**Mô tả:** Dự báo AQI 7 ngày tới (như dự báo thời tiết).

**Model:** Time series (LSTM, Prophet)
**Input:** Lịch sử 30 ngày + seasonal patterns
**Output:** AQI dự báo cho 7 ngày tới

**UI:**
```
Mon  Tue  Wed  Thu  Fri  Sat  Sun
120  95   88   105  130  115  90
🟡   🟢   🟢   🟡   🟠   🟡   🟢
```

**Use case:**
- Lập lịch hoạt động ngoài trời (picnic, marathon)
- Quyết định bật máy lọc không khí
- Chuẩn bị khẩu trang trước

### 5. **Pollution Source Analysis**
**Mô tả:** Phân tích nguyên nhân ô nhiễm chính.

**Analysis:**
- Feature importance: PM2.5 từ đâu? (xe cộ, công nghiệp, đốt rơm)
- Time patterns: Ô nhiễm cao vào giờ nào? (giờ cao điểm)
- Seasonal: Mùa nào tệ nhất? (mùa đông, đốt rơm)

**Visualization:**
- Pie chart: Contribution of each pollutant to AQI
- Bar chart: PM2.5 by hour of day
- Heatmap: Day of week vs hour

**Policy impact:**
- Đề xuất giảm xe cộ vào giờ cao điểm
- Cấm đốt rơm vào mùa đông

### 6. **AQI Chatbot (Integration with Zalo)**
**Mô tả:** Chatbot trả lời câu hỏi về chất lượng không khí.

**Sample conversations:**
- User: "AQI hôm nay khu vực tôi thế nào?"
  Bot: "AQI 135 - Moderate. Bạn nên hạn chế hoạt động ngoài trời kéo dài."

- User: "Ngày mai có nên đi chạy không?"
  Bot: "Dự báo AQI ngày mai: 88 (Good). Tuyệt vời để chạy bộ! 🏃"

- User: "Trạm nào gần tôi?"
  Bot: "Trạm AP001 cách bạn 2km. AQI hiện tại: 102."

**Features:**
- Natural language understanding (NLU)
- Location-based responses
- Personalized (dựa vào user profile)

### 7. **Health Impact Calculator**
**Mô tả:** Tính toán tác động sức khỏe của việc tiếp xúc ô nhiễm.

**Input:**
- AQI hôm nay: 180
- Thời gian ở ngoài trời: 2 giờ
- Profile: người lớn khỏe mạnh

**Output:**
- Equivalent cigarettes: "2 giờ ở ngoài = hút 3 điếu thuốc"
- Health risk: "Tăng 15% nguy cơ viêm đường hô hấp"
- Life expectancy impact: "Giảm 0.2 năm tuổi thọ nếu tiếp xúc dài hạn"

**Use case:**
- Awareness: Giúp người dùng hiểu rõ tác hại
- Motivation: Khuyến khích sử dụng máy lọc không khí

### 8. **Automated Report Generation**
**Mô tả:** Tự động tạo báo cáo AQI định kỳ (hàng ngày/tuần/tháng).

**n8n workflow:**
1. Schedule trigger (mỗi sáng 8AM)
2. Query database cho dữ liệu hôm qua
3. Generate charts (Plotly)
4. Create PDF report (pdfkit)
5. Send email với attachment
6. Post summary to Slack/Telegram
7. Archive report to Google Drive

**Report contents:**
- Yesterday's summary
- Week-over-week comparison
- Top 3 polluted stations
- Recommendations

### 9. **Air Purifier Control Integration**
**Mô tả:** Tự động bật/tắt máy lọc không khí dựa trên AQI.

**Integration với smart home:**
- n8n monitor AQI real-time
- If AQI > 100 → Send command to smart plug/IoT device
- Turn on air purifier automatically
- If AQI < 50 → Turn off to save energy

**Platform:**
- Xiaomi Mi Home
- Tuya Smart
- Home Assistant
- IFTTT

### 10. **Crowdsourced AQI Data**
**Mô tả:** Thu thập dữ liệu AQI từ cộng đồng (low-cost sensors).

**Architecture:**
- Users với PurpleAir/AirVisual sensor
- Submit data qua mobile app hoặc API
- n8n webhook nhận data → Validate → Store
- Aggregate với official station data
- Increase coverage (nhiều điểm đo hơn)

**Benefits:**
- Realtime coverage tốt hơn
- Community engagement
- Identify pollution hotspots

---

## 📈 ĐÁNH GIÁ VÀ CẢI TIẾN

### Metrics quan trọng

**Model Performance:**
- MAE (Mean Absolute Error) < 15 AQI points
- R² score > 0.85
- MAPE (Mean Absolute Percentage Error) < 10%

**System Performance:**
- API response time < 200ms
- Uptime > 99.5%
- Alert delivery < 30 seconds

**User Engagement:**
- Daily active users
- Alert open rate
- Chatbot conversation rate

### A/B Testing Ideas

1. **Alert Timing:** Gửi cảnh báo lúc nào hiệu quả nhất?
2. **Message Tone:** Formal vs friendly vs urgent
3. **Channels:** Zalo vs Email vs SMS - kênh nào tốt nhất?
4. **Recommendation Style:** Directive vs suggestive

---

## 🚀 KẾT LUẬN

### Giá trị của đồ án

✅ **Real-world impact:** Bảo vệ sức khỏe cộng đồng  
✅ **Technical skills:** ETL, ML, API, automation, dashboards  
✅ **Scalable:** Dễ mở rộng thêm trạm, pollutants, features  
✅ **Innovation:** Kết hợp ML + automation + real-time alerting  

### Next Steps

1. ✅ Hoàn thành ETL pipeline
2. ✅ Train baseline model (Random Forest)
3. ✅ Deploy FastAPI prediction service
4. ✅ Build n8n daily alert workflow
5. ⏳ Create Streamlit dashboard
6. ⏳ Add chatbot integration
7. ⏳ Setup monitoring & logging
8. ⏳ Write documentation & demo video

---

**Bạn muốn mình implement phần nào trước?**
- ETL + EDA notebooks?
- Train model script?
- FastAPI AQI prediction endpoints?
- n8n daily alert workflow?
- Streamlit dashboard?
