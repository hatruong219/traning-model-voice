#!/usr/bin/env bash
# Chạy trên máy có GPU. Làm hết phần máy, DỪNG ở chỗ cần bạn nghe và quyết.
#
#   ./run-on-gpu.sh setup                      # tạo .venv + cài môi trường
#   ./run-on-gpu.sh extract ~/videos           # video -> audio -> tách giọng
#   ./run-on-gpu.sh ref                        # cắt 30s mẫu sạch nhất
#   ./run-on-gpu.sh reftext                    # Whisper phiên âm đoạn mẫu (khỏi gõ tay)
#   ./run-on-gpu.sh zeroshot ["câu muốn thử"]  # clone giọng, KHÔNG train
#   ./run-on-gpu.sh narrate <thư mục .txt> [thư mục ra]   # đọc cả chapter
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
  # Lấy MỌI wav trong data/vocals, ưu tiên file có "vocal" trong tên (tên do
  # audio-separator đặt, không đoán cứng pattern nữa).
  mapfile -d '' ALL < <(find data/vocals -maxdepth 1 -name '*.wav' -print0 2>/dev/null)
  if [ "${#ALL[@]}" -eq 0 ]; then
      echo "data/vocals/ rỗng — chạy ./run-on-gpu.sh extract <thư mục video> trước."
      echo "Đang có:"; ls -la data/vocals 2>/dev/null || echo "  (thư mục chưa tồn tại)"
      exit 1
  fi
  F=""
  for f in "${ALL[@]}"; do
      case "${f,,}" in *vocal*) F="$f"; break ;; esac
  done
  [ -n "$F" ] || F="${ALL[0]}"

  DUR=$("$PY" -c "import soundfile as sf,sys; print(sf.info(sys.argv[1]).duration)" "$F" 2>/dev/null \
        || ffprobe -v error -show_entries format=duration -of csv=p=0 "$F")
  echo "nguồn: $(basename "$F")  ($(printf '%.0f' "$DUR")s)"

  # Cắt 30s bắt đầu ở 25% file. Hardcode -ss 30 sẽ ra RỖNG nếu file ngắn hơn 30s.
  LEN=$("$PY" -c "d=float('$DUR'); print(30 if d>=40 else max(5, d*0.8))")
  OFF="${OFF:-$("$PY" -c "d=float('$DUR'); l=float('$LEN'); print(round(min(d*0.25, max(0, d-l)),2))")}"
  echo "cắt ${LEN}s từ mốc ${OFF}s   (đổi mốc: OFF=90 ./run-on-gpu.sh ref)"

  ffmpeg -y -loglevel error -ss "$OFF" -t "$LEN" -i "$F" -ar 24000 -ac 1 data/ref/ref30.wav

  # ffmpeg có thể exit 0 mà không ghi gì (seek quá cuối file) — phải kiểm file thật.
  if [ ! -s data/ref/ref30.wav ]; then
      echo "(!) không cắt được. File nguồn dài $(printf '%.0f' "$DUR")s, mốc yêu cầu ${OFF}s."
      exit 1
  fi
  OK=$("$PY" -c "import soundfile as sf; print(round(sf.info('data/ref/ref30.wav').duration,1))")
  echo "xong: data/ref/ref30.wav  (${OK}s)"
  echo "NGHE. Phải sạch, giọng đều, không nhạc, không cắt giữa câu."
  echo "Không đạt thì đổi mốc:  OFF=120 ./run-on-gpu.sh ref"
  ;;

reftext)
  # Zero-shot cần biết đoạn mẫu NÓI GÌ để căn âm với chữ. Không phải gõ tay —
  # Whisper đã có trong venv, để nó phiên âm rồi lưu ra file.
  [ -f data/ref/ref30.wav ] || { echo "chưa có data/ref/ref30.wav — chạy ./run-on-gpu.sh ref"; exit 1; }
  "$PY" - <<'PYX'
from faster_whisper import WhisperModel
import torch
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = WhisperModel("large-v3", device=dev, compute_type="float16" if dev=="cuda" else "int8")
txt = " ".join(s.text.strip() for s in m.transcribe("data/ref/ref30.wav", language="vi")[0]).strip()
open("data/ref/ref30.txt", "w", encoding="utf-8").write(txt + "\n")
print(txt)
PYX
  echo
  echo "Lưu ở data/ref/ref30.txt — ĐỌC LẠI, sai chữ nào thì sửa file đó rồi chạy zeroshot."
  ;;

zeroshot)
  [ -f data/ref/ref30.txt ] || { echo "chưa có ref text — chạy ./run-on-gpu.sh reftext"; exit 1; }
  GEN="${2:-Lửa kín cả khung hình, không thấy trời cũng không thấy đất.}"
  "$PIP" show f5-tts >/dev/null 2>&1 || "$PIP" install -q f5-tts
  "$VENV/bin/f5-tts_infer-cli" \
    --ref_audio data/ref/ref30.wav \
    --ref_text "$(cat data/ref/ref30.txt)" \
    --gen_text "$GEN" \
    --output_dir data/zeroshot
  echo
  echo "NGHE data/zeroshot/ — ra giọng bạn thì DỪNG, khỏi train."
  ;;

narrate)
  # Đọc CẢ THƯ MỤC text bằng giọng đã clone. Tên file ra khớp tên file vào:
  #   tts-lines/S01.txt -> audio/S01.wav
  # Dùng cho dây chuyền manga-narration; giữ đúng giao diện của tts-edge.py.
  SRC="${2:?đưa thư mục chứa các file .txt, vd ../manga-narration/.../tts-lines}"
  OUT="${3:-${SRC%/*}/audio}"
  [ -d "$SRC" ] || { echo "không thấy thư mục: $SRC"; exit 1; }
  [ -f data/ref/ref30.wav ] && [ -f data/ref/ref30.txt ] \
    || { echo "chưa có mẫu giọng — chạy ./run-on-gpu.sh ref rồi reftext"; exit 1; }
  "$PIP" show f5-tts >/dev/null 2>&1 || "$PIP" install -q f5-tts
  mkdir -p "$OUT"

  mapfile -d '' TXTS < <(find "$SRC" -maxdepth 1 -name '*.txt' -print0 | sort -z)
  [ "${#TXTS[@]}" -gt 0 ] || { echo "không thấy file .txt nào trong $SRC"; exit 1; }
  echo "${#TXTS[@]} đoạn → $OUT"

  REFTXT="$(cat data/ref/ref30.txt)"
  done_n=0; skip_n=0; fail_n=0
  for t in "${TXTS[@]}"; do
      b=$(basename "${t%.txt}")
      # F5-TTS đặt tên output theo nội bộ, nên đọc vào thư mục tạm rồi đổi tên.
      if [ -s "$OUT/$b.wav" ] && [ "${FORCE:-0}" != "1" ]; then skip_n=$((skip_n+1)); continue; fi
      tmp=$(mktemp -d)
      if "$VENV/bin/f5-tts_infer-cli" --ref_audio data/ref/ref30.wav \
             --ref_text "$REFTXT" --gen_text "$(cat "$t")" \
             --output_dir "$tmp" >/dev/null 2>&1; then
          w=$(find "$tmp" -name '*.wav' | head -1)
          if [ -s "$w" ]; then mv "$w" "$OUT/$b.wav"; done_n=$((done_n+1));
          else echo "  $b: không ra file"; fail_n=$((fail_n+1)); fi
      else
          echo "  $b: f5-tts lỗi"; fail_n=$((fail_n+1))
      fi
      rm -rf "$tmp"
      [ $((done_n % 10)) -eq 0 ] && [ "$done_n" -gt 0 ] && echo "  … $done_n đoạn"
  done
  echo
  echo "đọc $done_n · có sẵn $skip_n · lỗi $fail_n → $OUT"
  [ "$fail_n" -gt 0 ] && echo "(!) chạy lại để đọc tiếp phần lỗi; file đã có sẽ bỏ qua (FORCE=1 để đọc lại hết)"
  echo "→ tiếp, ở máy có manga-narration:"
  echo "   python3 scripts/build-video.py <results>"
  echo "   python3 scripts/retime-from-audio.py <results>"
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
