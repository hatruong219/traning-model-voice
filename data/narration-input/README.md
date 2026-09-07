# narration-input — text để máy GPU đọc thành audio

84 đoạn lời kể, sinh từ dây chuyền `manga-narration` (`narration.tsv` → `build-narration.py`).
Đặt ở đây để máy GPU pull qua git thay vì scp.

## Vòng làm việc

| Máy | Việc |
|---|---|
| WSL (có Claude Code) | viết lời kể → sinh `tts-lines/*.txt` → commit vào đây |
| GPU | `./run-on-gpu.sh narrate data/narration-input` → ra `*.wav` |
| — | gửi wav về qua Drive |
| WSL | đặt wav vào `results/audio/` → `build-video.py` + `retime-from-audio.py` |

Tên file ra khớp tên vào: `S01.txt` → `S01.wav`.

**Nguồn sự thật vẫn là `narration.tsv`** bên `manga-narration`. Sửa lời thì sửa ở đó rồi
chạy lại `build-narration.py`, đừng sửa mấy file này.
