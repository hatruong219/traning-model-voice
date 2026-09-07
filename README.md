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

## Không phải source venv nữa

`run-on-gpu.sh` **tự kích hoạt `.venv`** — mọi lệnh python/pip trong script đi qua
`.venv/bin/`, không phụ thuộc PATH của shell. Chỉ cần:

```bash
./run-on-gpu.sh setup          # tạo .venv rồi cài vào đó
./run-on-gpu.sh extract ~/videos
```

Chưa có `.venv` mà chạy lệnh khác → script nhắc chạy `setup` trước, không im lặng dùng
python hệ thống.

Đổi chỗ khác thì đặt biến: `VENV=~/envs/tts ./run-on-gpu.sh setup`

### Nếu muốn venv tự bật cả khi làm tay

Thêm vào `~/.bashrc` (hoặc `~/.zshrc`) — tự bật/tắt theo thư mục:

```bash
cd() {
    builtin cd "$@" || return
    if [ -n "${VIRTUAL_ENV:-}" ] && [[ "$PWD" != "$(dirname "$VIRTUAL_ENV")"* ]]; then
        deactivate 2>/dev/null
    fi
    [ -z "${VIRTUAL_ENV:-}" ] && [ -f .venv/bin/activate ] && source .venv/bin/activate
}
```

Sạch hơn thì dùng `direnv`:

```bash
sudo apt install -y direnv
echo 'eval "$(direnv hook bash)"' >> ~/.bashrc      # zsh thì đổi bash -> zsh
echo 'source .venv/bin/activate' > .envrc
direnv allow
```

`direnv` là cách đúng nếu bạn có nhiều project — nó tự bật đúng venv theo từng thư mục
và tự tắt khi ra ngoài. Hàm `cd()` ở trên là bản tự làm, không cần cài gì.
