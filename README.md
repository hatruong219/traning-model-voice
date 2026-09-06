# vi-tts-voice — dựng giọng đọc tiếng Việt riêng cho kênh kể truyện

Mục tiêu: từ **video có sẵn của chính chủ kênh** → một giọng TTS mang chất giọng đó, dùng
được cho dây chuyền `manga-narration`.

**Trạng thái: CHỈ CÓ PLAN.** Chưa cài gì, chưa chạy gì. Toàn bộ phần train chạy trên máy có GPU.

- Kế hoạch thi hành: [plans/plan.md](plans/plan.md)
- Script đã có: `scripts/check-phonetic-coverage.py` (chạy CPU, không cần GPU)

## Vì sao không cần đọc hết từ vựng tiếng Việt

TTS neural học ánh xạ **âm vị → âm thanh**, không ghép từ đã thu. Tiếng Việt lại có bộ âm
đóng và nhỏ (~22 âm đầu · ~16 vần · 6 thanh) và chính tả gần như 1:1 với phát âm, nên phủ
đủ **bộ âm** là tổng hợp được mọi âm tiết, kể cả chưa từng nghe.

Đo thực tế trên 1.658 âm tiết (~9 phút đọc): phủ **27/28 âm đầu**, đủ **6/6 thanh**.
Thiếu duy nhất âm đầu `p` — chỉ xuất hiện trong từ ngoại lai.

**Bạn không dạy model tiếng Việt. Bạn chỉ dạy nó chất giọng của bạn.** Kiến thức ngữ âm đến
từ checkpoint gốc đã train hàng trăm giờ. Đó là lý do 30–60 phút audio là đủ.
