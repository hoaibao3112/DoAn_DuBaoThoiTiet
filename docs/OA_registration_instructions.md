# Hướng dẫn: Đăng ký Official Account (OA) cá nhân - nhanh để thử nghiệm

Tài liệu này hướng dẫn từng bước để bạn tạo một Official Account (OA) Zalo dạng cá nhân để chạy thử workflow (n8n) và test trả lời tự động. Nội dung kèm mẫu text (copy/paste) cho các ô form.

---

## 0) Chuẩn bị trước khi bắt đầu
- Một tài khoản Zalo cá nhân (đã hoạt động, có thể nhận mã OTP SMS)
- Ảnh đại diện (avatar) 240x240 px, JPG/PNG, < 5MB
- Ảnh bìa (tùy chọn)
- Bạn có thể dùng laptop/PC; nếu trình duyệt có extension chặn popup hãy tắt.

## 1) Mở trang tạo OA
- Truy cập: https://oa.zalo.me/manage
- Đăng nhập bằng tài khoản Zalo cá nhân của bạn
- Chọn **Tạo Official Account** (Create OA) → Chọn loại **Cá nhân / Personal** (hoặc tương tự)

## 2) Điền form đăng ký (mẫu bạn có thể copy/paste)
- Danh mục hoạt động: `Dịch vụ khác` (hoặc `Giáo dục` nếu phù hợp)
- Tên Official Account: `Dự Báo Thời Tiết PTDL`

- Thông tin giới thiệu (bản ngắn - dùng ở box ngắn):

  `Dự án cá nhân môn Phân Tích Dữ Liệu: cung cấp dự báo thời tiết và chỉ số AQI kết hợp API realtime và mô hình học máy.`

- Thông tin giới thiệu (bản trung - dùng phần mô tả):

  `Dự Báo Thời Tiết PTDL là dự án đồ án cá nhân thuộc môn Phân Tích Dữ Liệu, cung cấp thông tin thời tiết hiện tại, dự báo nhiệt độ và chỉ số chất lượng không khí (AQI) cho các thành phố Việt Nam. Hệ thống kết hợp dữ liệu realtime từ API (OpenWeather, WAQI) và mô hình ML nội bộ để đưa ra dự báo ngắn hạn và khuyến cáo sức khỏe.`

- Địa chỉ (nếu bắt buộc): `Hanoi, Vietnam` (hoặc địa chỉ thật của bạn)
- Ảnh đại diện: upload ảnh logo 240x240
- Tích chọn: Tôi đã đọc và đồng ý Điều khoản…
- Nhấn: **Tạo tài khoản OA** / **Create OA**

> Ghi chú: nếu site yêu cầu nhiều giấy tờ (chỉ xảy ra nếu bạn chọn OA doanh nghiệp), hãy chuyển sang loại OA cá nhân.

## 3) Xác thực / OTP
- Zalo thường yêu cầu xác thực bằng số điện thoại (SMS). Kiểm tra điện thoại để nhận mã và nhập mã vào form.
- Nếu mã không tới: kiểm tra số, mạng, thử số khác.

## 4) Lấy Access Token OA (sau khi OA được tạo)
1. Đăng nhập vào OA dashboard: https://oa.zalo.me/manage
2. Vào **Cài đặt** (Settings) → tìm phần **Developer / API** (hoặc mục tương tự là `Quản lý API`)
3. Tạo hoặc lấy **Access Token** cho OA. Copy token (một chuỗi dài).

Lưu token an toàn, không commit vào git.

## 5) Bật webhook n8n (tạm) / test local
1. Nếu bạn dùng ngrok: chạy `ngrok http 5678` và copy HTTPS URL.
2. Trong n8n, mở workflow `Zalo AI Assistant` → mở node `Zalo Webhook` → copy webhook path (n8n hiển thị full URL khi node mở).
3. Trong OA dashboard → Developer / Webhook → dán URL và bật event `user_send_text` (hoặc event message tương tự). Lưu.

## 6) Test end-to-end
- Bước A (mô phỏng): Gọi webhook mà n8n cung cấp (ví dụ dùng PowerShell) — xem `docs/TESTS.md` (nếu có) hoặc dùng snippet dưới:

```powershell
$webhook = "https://<ngrok-url>/zalo-webhook/zalo-ai-assistant"
$payload = @{
  events = @(@{
    type = "user_send_text"
    sender = @{ id = "TEST_USER_ID" }
    message = @{ text = "Thời tiết Hà Nội hôm nay?" }
  })
} | ConvertTo-Json -Depth 10

Invoke-RestMethod -Uri $webhook -Method Post -Body $payload -ContentType "application/json"
```

- Bước B (thực tế): Dùng một tài khoản Zalo khác nhắn tin đến OA bằng nội dung "Thời tiết Hà Nội" → OA sẽ nhận sự kiện, n8n sẽ xử lý và trả lời (nếu workflow bật và token cấu hình đúng).

## 7) Test gửi tin nhắn qua API (nếu muốn gửi thử bằng script)
```powershell
$token = "PASTE_YOUR_ZALO_TOKEN"
$user_id = "RECIPIENT_USER_ID"  # user thực đã nhắn OA trước đó
$body = @{
  recipient = @{ user_id = $user_id }
  message = @{ text = "Hello từ hệ thống dự báo thời tiết!" }
} | ConvertTo-Json -Depth 10

Invoke-RestMethod -Uri "https://openapi.zalo.me/v2.0/oa/message?access_token=$token" -Method Post -Body $body -ContentType "application/json"
```

Phản hồi thành công (200/OK) nghĩa là token hợp lệ và OA có thể gửi tin.

## 8) Lưu ý bảo mật
- Không commit token vào kho mã. Lưu trong `.env` (gitignored) hoặc credentials trong n8n.
- Nếu dùng service-account cho Drive, cân nhắc Shared Drive (xem docs).

---
Mình đã tạo file này trong repo: `docs/OA_registration_instructions.md` — bạn có thể mở và copy nội dung nhanh.

Nếu bạn muốn, mình sẽ tiếp tục:
- (A) mô phỏng gửi webhook ngay (nhưng bạn cần bật ngrok và paste URL), hoặc
- (B) hướng dẫn chi tiết cách lấy Access Token bằng ảnh chụp màn hình (từng bước), hoặc
- (C) tạo 1 script nhỏ để lấy user_id khi có tin nhắn (ví dụ webhook debug handler).

Chọn A/B/C hoặc nói "Tự làm" nếu bạn chỉ cần tài liệu trên.
