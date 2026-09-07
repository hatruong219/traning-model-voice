#!/usr/bin/env bash
# Chạy trên máy có GPU. Làm hết phần máy, DỪNG ở chỗ cần bạn nghe và quyết.
#
#   ./run-on-gpu.sh setup                      # tạo .venv + cài môi trường
#   ./run-on-gpu.sh extract ~/videos           # video -> audio -> tách giọng
#   ./run-on-gpu.sh ref                        # cắt 30s mẫu sạch nhất
#   ./run-on-gpu.sh coverage                   # đo phủ âm (chạy được cả trên CPU)
#   ./run-on-gpu.sh dataset                    # cắt câu + phiên âm -> metadata.csv
#
# Chưa test trên GPU thật — máy dev không có GPU. Lỗi ở đâu thì báo, đừng đoán.
set -euo pipefail
cd "$(dirname "$0")"
CMD="${1:-help}"

# Tự kích hoạt venv — khỏi phải source tay mỗi lần mở terminal.
# Mọi lệnh python/pip bên dưới đi qua $PY và $PIP, không phụ thuộc PATH của shell.
VENV="${VENV:-.venv}"
if [ -x "$VENV/bin/python" ]; then
    PY="$VENV/bin/python"; PIP="$VENV/bin/pip"
    export VIRTUAL_ENV="$PWD/$VENV"
    export PATH="$PWD/$VENV/bin:$PATH"
else
    PY="python3"; PIP="pip3"
    # Cảnh báo mà vẫn chạy tiếp bằng python hệ thống thì vô nghĩa — DỪNG luôn.
    # Trừ setup (nó tạo venv) và help (không cần python).
    case "$CMD" in
        setup|help|"") ;;
        *) echo "(!) chưa có $VENV — chạy trước:  ./run-on-gpu.sh setup" >&2; exit 1 ;;
    esac
fi

case "$CMD" in

setup)
  nvidia-smi --query-gpu=name,memory.total --format=csv || echo "(!) không thấy GPU"
  if [ ! -x "$VENV/bin/python" ]; then
      echo "tạo $VENV …"
      python3 -m venv "$VENV" || {
          echo "python3 -m venv lỗi — cài trước: sudo apt install -y python3-venv"; exit 1; }
      PY="$VENV/bin/python"; PIP="$VENV/bin/pip"
  fi
  "$PIP" install -q --upgrade pip
  "$PIP" install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
  "$PIP" install "audio-separator[gpu]" faster-whisper silero-vad soundfile
  "$PY" -c "import torch; print('torch', torch.__version__, '· cuda:', torch.cuda.is_available())"
  echo
  echo "Xong. Từ giờ chỉ cần ./run-on-gpu.sh <lệnh> — script tự dùng $VENV, không phải source."
  ;;

extract)
  SRC="${2:?đưa đường dẫn thư mục chứa video}"
  [ -d "$SRC" ] || { echo "không thấy thư mục: $SRC"; exit 1; }
  mkdir -p data/raw data/vocals

  # find thay vì brace glob: bắt được cả đuôi HOA (.MP4) và nhiều định dạng hơn.
  # Brace glob "$SRC"/*.{mp4,mkv} không nở khi không khớp -> truyền chuỗi nguyên vào ffmpeg.
  mapfile -d '' VIDS < <(find "$SRC" -maxdepth 1 -type f \
      \( -iname '*.mp4' -o -iname '*.mkv' -o -iname '*.webm' -o -iname '*.mov' \
         -o -iname '*.ts'  -o -iname '*.avi' -o -iname '*.flv'  -o -iname '*.m4a' \
         -o -iname '*.wav' -o -iname '*.mp3' \) -print0)

  if [ "${#VIDS[@]}" -eq 0 ]; then
      echo "Không thấy video/audio nào trong $SRC"
      echo "Thư mục đang có:"
      ls -la "$SRC" | head -20
      echo
      echo "Định dạng nhận: mp4 mkv webm mov ts avi flv m4a wav mp3 (không phân biệt hoa thường)"
      exit 1
  fi

  echo "${#VIDS[@]} file nguồn → rút audio 24 kHz mono"
  for v in "${VIDS[@]}"; do
      b=$(basename "${v%.*}")
      ffmpeg -y -loglevel error -i "$v" -vn -ac 1 -ar 24000 -c:a pcm_s16le "data/raw/$b.wav" \
          && echo "  ok: $b.wav" || echo "  LỖI: $v"
  done

  # Kiểm THẬT là có file rồi mới tách — không để glob rỗng lọt vào separator.
  mapfile -d '' RAWS < <(find data/raw -maxdepth 1 -name '*.wav' -print0)
  if [ "${#RAWS[@]}" -eq 0 ]; then
      echo "ffmpeg không tạo được file nào trong data/raw/ — xem lỗi ở trên"; exit 1
  fi

  echo
  echo "${#RAWS[@]} file → tách giọng khỏi nhạc nền"
  for w in "${RAWS[@]}"; do
      audio-separator "$w" --model_filename UVR-MDX-NET-Voc_FT.onnx \
        --output_dir data/vocals --output_format WAV
  done

  echo
  echo "NGHE THỬ 3 file trong data/vocals/ — còn nhạc hoặc tiếng rít thì đổi model:"
  echo "  sửa UVR-MDX-NET-Voc_FT.onnx thành Kim_Vocal_2.onnx trong script, chạy lại."
  ;;

ref)
  # lấy 30 giây liên tục có năng lượng đều nhất làm mẫu zero-shot
  mkdir -p data/ref
  F=$(ls data/vocals/*Vocals*.wav data/vocals/*vocals*.wav 2>/dev/null | head -1)
  [ -n "$F" ] || { echo "chưa có file giọng trong data/vocals/ — chạy extract trước"; exit 1; }
  ffmpeg -y -i "$F" -ss 30 -t 30 -ar 24000 -ac 1 data/ref/ref30.wav
  echo "mẫu: data/ref/ref30.wav — NGHE. Phải sạch, giọng đều, không nhạc, không ngắt câu giữa."
  echo "Không đạt thì đổi -ss 30 thành mốc khác rồi chạy lại."
  ;;

coverage)
  "$PY" scripts/check-phonetic-coverage.py \
    "${2:-data/dataset/metadata.csv}" --min 5
  ;;

dataset)
  mkdir -p data/dataset/wavs
  "$PY" - <<'PY'
import glob, os, torch, soundfile as sf
from faster_whisper import WhisperModel
vad, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)
get_ts, _, read_audio, _, _ = utils
asr = WhisperModel("large-v3", device="cuda", compute_type="float16")
rows, n = [], 0
for f in sorted(glob.glob("data/vocals/*.wav")):
    wav = read_audio(f, sampling_rate=16000)
    for seg in get_ts(wav, vad, sampling_rate=16000):
        dur = (seg['end'] - seg['start']) / 16000
        if not 2.0 <= dur <= 12.0:
            continue
        n += 1
        out = f"data/dataset/wavs/u{n:05d}.wav"
        os.system(f"ffmpeg -y -loglevel error -i '{f}' -ss {seg['start']/16000:.3f} "
                  f"-t {dur:.3f} -ar 24000 -ac 1 '{out}'")
        txt = " ".join(s.text.strip() for s in asr.transcribe(out, language="vi")[0])
        if len(txt.split()) >= 3:
            rows.append(f"u{n:05d}.wav|{txt}")
        else:
            os.remove(out)
open("data/dataset/metadata.csv", "w", encoding="utf-8").write("\n".join(rows) + "\n")
tot = sum(sf.info(f"data/dataset/wavs/{r.split('|')[0]}").duration for r in rows)
print(f"{len(rows)} câu · {tot/60:.1f} phút")
PY
  echo
  echo "ĐỌC TAY 20 dòng ngẫu nhiên trong data/dataset/metadata.csv, so với audio."
  echo "Phiên âm sai -> model học phát âm sai, đây là lỗi âm thầm và đắt nhất."
  shuf -n 20 data/dataset/metadata.csv
  ;;

*)
  sed -n '2,12p' "$0"
  ;;
esac
