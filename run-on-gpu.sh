#!/usr/bin/env bash
# Chạy trên máy có GPU. Làm hết phần máy, DỪNG ở chỗ cần bạn nghe và quyết.
#
#   ./run-on-gpu.sh setup                      # tạo .venv + cài môi trường
#   ./run-on-gpu.sh extract ~/videos           # video -> audio -> tách giọng
#   ./run-on-gpu.sh ref                        # cắt 30s mẫu sạch nhất
#   ./run-on-gpu.sh reftext                    # Whisper phiên âm đoạn mẫu (khỏi gõ tay)
#   ./run-on-gpu.sh zeroshot ["câu muốn thử"]  # clone giọng, KHÔNG train
#   ./run-on-gpu.sh narrate <thư mục .txt> [thư mục ra]   # đọc cả chapter
#   ./run-on-gpu.sh diag                       # đo mức âm mẫu + output
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

  # onnxruntime và onnxruntime-gpu cài vào CÙNG thư mục onnxruntime/ — sống chung thì
  # bản CPU thắng và GPU không được dùng. Uninstall một cái cũng xoá file dùng chung,
  # để lại package hỏng ("no attribute get_available_providers"). Nên dọn sạch rồi cài lại.
  if "$PY" -c "import onnxruntime" 2>/dev/null; then
      if ! "$PY" -c "import onnxruntime as o; assert 'CUDAExecutionProvider' in o.get_available_providers()" 2>/dev/null; then
          echo "dọn onnxruntime để bật GPU …"
          "$PIP" uninstall -y onnxruntime onnxruntime-gpu >/dev/null 2>&1 || true
          rm -rf "$VENV"/lib/python*/site-packages/onnxruntime*
          "$PIP" install -q onnxruntime-gpu
      fi
  fi

  "$PY" -c "import torch; print('torch', torch.__version__, '· cuda:', torch.cuda.is_available())"
  "$PY" -c "import onnxruntime as o; p=o.get_available_providers(); print('onnxruntime', o.__version__, p); \
import sys; sys.exit(0 if 'CUDAExecutionProvider' in p else 1)" || {
      echo
      echo "(!) onnxruntime chưa thấy CUDAExecutionProvider — tách giọng sẽ chạy CPU (chậm)."
      echo "    Thường do thiếu cuDNN. Thử:  $PIP install nvidia-cudnn-cu12"
      echo "    Vẫn không được thì chạy tiếp cũng OK, chỉ chậm hơn."
  }
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
  mkdir -p data/ref
  mapfile -d '' ALL < <(find data/vocals -maxdepth 1 -name '*.wav' -print0 2>/dev/null)
  if [ "${#ALL[@]}" -eq 0 ]; then
      echo "data/vocals/ rỗng — chạy ./run-on-gpu.sh extract <thư mục video> trước."
      ls -la data/vocals 2>/dev/null || echo "  (thư mục chưa tồn tại)"
      exit 1
  fi
  F=""
  for f in "${ALL[@]}"; do
      case "${f,,}" in *vocal*) F="$f"; break ;; esac
  done
  [ -n "$F" ] || F="${ALL[0]}"

  DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$F")
  # F5-TTS TỰ CẮT audio mẫu về 12s. Cắt dài hơn thì ref_text (phiên âm cả đoạn dài)
  # không còn khớp audio đã bị cắt -> căn lệch -> output rỗng. Nên giữ 10s cho chắc.
  LEN="${LEN:-10}"
  OFF="${OFF:-$("$PY" -c "d=float('$DUR'); print(round(min(d*0.25, max(0, d-float('$LEN'))),2))")}"
  echo "nguồn: $(basename "$F")  ($(printf '%.0f' "$DUR")s)"
  echo "cắt ${LEN}s từ mốc ${OFF}s   (đổi: OFF=90 LEN=10 ./run-on-gpu.sh ref)"

  # loudnorm: mẫu quá nhỏ tiếng thì F5-TTS ra output im lặng. Chuẩn về -16 LUFS.
  ffmpeg -y -loglevel warning -ss "$OFF" -t "$LEN" -i "$F" \
      -af "loudnorm=I=-16:TP=-1.5:LRA=11" -ar 24000 -ac 1 data/ref/ref.wav
  [ -s data/ref/ref.wav ] || { echo "(!) không cắt được — file dài $(printf '%.0f' "$DUR")s, mốc ${OFF}s"; exit 1; }
  rm -f data/ref/ref.txt data/ref/ref30.wav data/ref/ref30.txt   # transcript cũ + file tên cũ
  OK=$(ffprobe -v error -show_entries format=duration -of csv=p=0 data/ref/ref.wav 2>/dev/null)
  [ -n "$OK" ] || { echo "(!) file cắt ra không đọc được"; exit 1; }
  echo "xong: data/ref/ref.wav  ($(printf '%.1f' "$OK")s)"
  echo "NGHE. Sạch, giọng đều, không nhạc. Rồi chạy: ./run-on-gpu.sh reftext"
  ;;

reftext)
  # Zero-shot cần biết đoạn mẫu NÓI GÌ để căn âm với chữ. Không phải gõ tay —
  # Whisper đã có trong venv, để nó phiên âm rồi lưu ra file.
  [ -f data/ref/ref.wav ] || { echo "chưa có data/ref/ref.wav — chạy ./run-on-gpu.sh ref"; exit 1; }
  "$PY" - <<'PYX'
from faster_whisper import WhisperModel
import torch
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = WhisperModel("large-v3", device=dev, compute_type="float16" if dev=="cuda" else "int8")
txt = " ".join(s.text.strip() for s in m.transcribe("data/ref/ref.wav", language="vi")[0]).strip()
open("data/ref/ref.txt", "w", encoding="utf-8").write(txt + "\n")
print(txt)
PYX
  echo
  echo "Lưu ở data/ref/ref.txt — ĐỌC LẠI, sai chữ nào thì sửa file đó rồi chạy zeroshot."
  ;;

zeroshot)
  D=$(ffprobe -v error -show_entries format=duration -of csv=p=0 data/ref/ref.wav 2>/dev/null) \
    || { echo "chưa có mẫu — chạy ./run-on-gpu.sh ref rồi reftext"; exit 1; }
  "$PY" -c "import sys; sys.exit(0 if float('$D')<=12 else 1)" || {
      echo "(!) mẫu dài ${D}s — F5-TTS chỉ dùng 12s đầu, transcript sẽ lệch."
      echo "    LEN=10 ./run-on-gpu.sh ref && ./run-on-gpu.sh reftext"; exit 1; }
  [ -f data/ref/ref.txt ] || { echo "chưa có transcript — ./run-on-gpu.sh reftext"; exit 1; }
  "$PIP" show f5-tts >/dev/null 2>&1 || "$PIP" install -q f5-tts

  GEN="${2:-Lửa kín cả khung hình, không thấy trời cũng không thấy đất.}"
  # Qua tts-f5.py thay vì f5-tts_infer-cli: CLI không cho đổi dtype, mà GTX 16xx
  # cần fp32 (fp16 -> NaN -> waveform hằng số, file đầy nhưng không có tiếng).
  "$PY" scripts/tts-f5.py --ref data/ref/ref.wav --ref-text-file data/ref/ref.txt \
      --text "$GEN" --out data/zeroshot/test.wav ${DEVICE:+--device "$DEVICE"} \
      ${CKPT:+--ckpt "$CKPT"} ${VOCAB:+--vocab "$VOCAB"} \
    && { echo; echo "NGHE data/zeroshot/test.wav — ra giọng bạn thì DỪNG, khỏi train."; } \
    || { echo; echo "(!) chưa ra tiếng. Thử CPU:  DEVICE=cpu ./run-on-gpu.sh zeroshot"; exit 1; }
  ;;

narrate)
  SRC="${2:?đưa thư mục chứa các file .txt}"
  OUT="${3:-${SRC%/*}/audio}"
  [ -d "$SRC" ] || { echo "không thấy thư mục: $SRC"; exit 1; }
  [ -f data/ref/ref.wav ] && [ -f data/ref/ref.txt ] \
    || { echo "chưa có mẫu giọng — ./run-on-gpu.sh ref rồi reftext"; exit 1; }
  "$PIP" show f5-tts >/dev/null 2>&1 || "$PIP" install -q f5-tts

  "$PY" scripts/tts-f5.py --ref data/ref/ref.wav --ref-text-file data/ref/ref.txt \
      --in-dir "$SRC" --out-dir "$OUT" ${DEVICE:+--device "$DEVICE"} ${FORCE:+--force} \
      ${CKPT:+--ckpt "$CKPT"} ${VOCAB:+--vocab "$VOCAB"}
  echo
  echo "→ gửi $OUT về máy WSL, rồi:"
  echo "   python3 scripts/retime-from-audio.py <results>"
  echo "   python3 scripts/build-video.py <results>"
  ;;

diag)
  # Đo mức âm. Output im lặng gần như luôn do mẫu quá nhỏ tiếng hoặc rỗng.
  for f in data/ref/ref.wav data/zeroshot/infer_cli_basic.wav; do
      [ -f "$f" ] || { echo "$f — chưa có"; continue; }
      echo "── $f"
      ffprobe -v error -show_entries stream=sample_rate,channels,duration \
              -of default=noprint_wrappers=1 "$f" | sed 's/^/   /'
      ffmpeg -hide_banner -i "$f" -af "volumedetect" -f null - 2>&1 \
          | grep -E 'mean_volume|max_volume' | sed 's/^\[[^]]*\] /   /'
      echo
  done
  echo "Đọc: max_volume gần 0 dB là bình thường. Dưới -40 dB là gần như im lặng."
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
