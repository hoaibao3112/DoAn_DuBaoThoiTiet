# Đồ án: Hệ thống Dự báo Thời tiết tự động

## Mô tả tổng quan
Đây là đồ án hệ thống dự báo thời tiết kết hợp:
- Backend FastAPI cung cấp API thời tiết và AQI (chỉ số chất lượng không khí).
- Frontend Streamlit cho phép người dùng tương tác, xem dự báo và biểu đồ.
- n8n để tự động hoá gửi email thời tiết hàng ngày (Gmail/SMTP) kèm biểu đồ.
- Hệ thống có khả năng ước lượng xác suất mưa (*) cho các trạm thiếu lịch sử bằng cách mượn dữ liệu trạm lân cận.

(*) "POP" = Probability of Precipitation — hệ thống xử lý trường hợp thiếu dữ liệu bằng thuật toán láng giềng.

## Tính năng chính
- Lấy dữ liệu OpenWeather và Air Quality (AQI), kết hợp dữ liệu trạm nội bộ.
- Endpoint trả về bản tóm tắt hiện tại (`/weather/current`) và endpoint có trả về hình ảnh biểu đồ nhúng base64 (`/weather/current_with_chart`).
- Tự động gửi email hàng ngày (cron 06:30) cho danh sách người nhận, kèm biểu đồ inline và khuyến nghị.
- Ẩn và quản lý khóa API và mật khẩu qua file `.env` (hoặc secret manager).

## Cấu trúc dự án (tóm tắt)
- `app/` — mã backend FastAPI
  - `weather.py` — logic lấy/ghi dữ liệu, endpoints `/weather/*`
  - `main.py` — khởi tạo FastAPI app
  - `ml_predictor.py` — mô-đun dự báo (nếu có)
- `frontend/` — Streamlit app (`streamlit_app.py`)
- `n8n-workflows/` — các json workflow n8n (Daily Weather Email, v.v.)
- `scripts/` — tiện ích (gửi email mẫu, render template, huấn luyện, v.v.)
- `docker-compose.yml`, `Dockerfile` — cấu hình chạy dịch vụ
- `requirements.txt` — dependencies Python

## Endpoints chính (ví dụ)
- `GET /weather/health` — kiểm tra trạng thái
- `GET /weather/current?city={Tên thành phố|station_id}` — trả JSON tóm tắt hiện tại
- `GET /weather/current_with_chart?city={Tên thành phố}` — trả JSON có thêm `chart_base64` (PNG base64)
- `GET /weather/neighbor_probability?station={id}` — trả POP ước lượng từ trạm lân cận

Lưu ý: tham số có thể là `city` hoặc `station` tuỳ cấu hình; xem `app/weather.py` để biết chính xác.

## Yêu cầu hệ thống
- Docker & Docker Compose (hoặc Docker Desktop trên Windows)
- Python 3.11 (để chạy local, phát triển)
- Tài khoản Gmail/SMTP nếu muốn gửi email (hoặc SMTP server khác)

## Cài đặt & chạy nhanh (Docker Compose)
Mệnh lệnh dưới đây chạy các dịch vụ chính (ai-assistant backend, n8n, streamlit, redis):

PowerShell (ở thư mục gốc dự án):

```powershell
# Build (nếu có thay đổi code backend)
docker compose build ai-assistant

# Khởi động dịch vụ (chỉ streamlit nếu muốn)
docker compose up -d streamlit
# Hoặc khởi động tất cả dịch vụ:
docker compose up -d

# Kiểm tra container đang chạy
docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Ports}}"

# Xem log streamlit (theo dõi)
docker logs -f streamlit
```

Sau khi chạy: mở trình duyệt tới `http://localhost:8501` để xem Streamlit, và `http://localhost:8000/docs` để xem Swagger UI của FastAPI (nếu bật).

## Thiết lập biến môi trường (file `.env`)
Tạo file `.env` ở gốc repo với các biến tối thiểu sau:

```
OPENWEATHER_API_KEY=your_openweather_key
EMAIL_USER=you@example.com
EMAIL_PASS=your_smtp_password
N8N_PASSWORD=somepassword
REDIS_URL=redis://redis:6379
```

Lưu ý: không commit file `.env` lên git. Sử dụng secret manager cho môi trường production và đổi mật khẩu API định kỳ.

## n8n workflow & gửi email tự động
- Các workflow n8n nằm trong `n8n-workflows/` (ví dụ `Daily Weather Email.json`).
- Luồng chính: Cron ➜ Load recipients ➜ HTTP Request (`/weather/current`) ➜ Send Email (SMTP).
- Khi import workflow vào n8n, kiểm tra:
  - Node `HTTP Request` có gửi `city` dưới `queryParameters` (không để lỗi khoảng trắng tên field);
  - Node `Send Email` sử dụng Mustache template trong field HTML (`{{$json["field"]}}`) và chế độ Fixed mode cho `html`.

## Thêm chart trong email
- Backend cung cấp base64 PNG ở endpoint `/weather/current_with_chart`.
- Để có chart trong email cần rebuild `ai-assistant` với code mới, sau đó n8n gọi endpoint này để nhận `chart_base64` và nhúng vào `<img src="data:image/png;base64,{{$json["chart_base64"]}}"/>`.

Lệnh rebuild + khởi động lại backend:

```powershell
docker compose build ai-assistant
docker compose up -d ai-assistant
```

## Phát triển & debug nhanh
- Thay đổi code backend —> rebuild image `ai-assistant`.
- Muốn test local nhanh (không Docker): tạo virtualenv, pip install -r requirements.txt, chạy `uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`.
- Kiểm tra kết nối giữa n8n và backend: từ container n8n dùng `curl http://<ai-assistant-ip>:8000/weather/health` hoặc tạm thời dùng `host.docker.internal` tuỳ Docker config.

## Gợi ý nâng cao / bảo mật
- Chuyển mật khẩu/keys sang secret manager (HashiCorp Vault, AWS Secrets Manager, hoặc Docker secrets).
- Hạn chế quyền SMTP (dùng app password cho Gmail) và bật 2FA.
- Đặt rate-limit cho endpoint truy cập OpenWeather để tránh vượt quota.

## Tài liệu & tham khảo
- FastAPI: https://fastapi.tiangolo.com/
- Streamlit: https://streamlit.io/
- n8n: https://n8n.io/

## Liên hệ
Nếu cần mình hỗ trợ tiếp:
- Yêu cầu: rebuild `ai-assistant` để bật chart, kích hoạt workflow n8n, hoặc sửa nội dung email template.
- Ghi chú: mình có thể cập nhật `README.md` chính (thay thế file gốc) nếu bạn muốn.

---
File này được tạo tự động — nếu bạn muốn thêm phần `Hướng dẫn sử dụng` chi tiết hơn (ví dụ: ví dụ request, mẫu payload recipients, hướng dẫn import workflow n8n), báo mình sẽ bổ sung.