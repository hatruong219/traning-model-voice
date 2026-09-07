"""Đọc thư mục .txt thành .wav bằng Piper — giọng tiếng Việt, license CC-BY-4.0.

Vì sao Piper thay vì F5-TTS: mọi checkpoint F5-TTS tiếng Việt đều kế thừa CC-BY-NC từ
bộ Emilia, tức phi thương mại. Giọng Piper vi_VN-vais1000-medium là CC-BY-4.0 — dùng
thương mại được, chỉ cần ghi nguồn.

Đổi lại: Piper KHÔNG clone giọng từ mẫu. Đây là giọng có sẵn của bộ VAIS, không phải
giọng bạn. Muốn giọng riêng thì phải train một voice Piper mới.

Chạy CPU, nhanh hơn realtime — không cần GPU.

Chạy:
    python3 scripts/tts-piper.py --in-dir <thư mục .txt> --out-dir <thư mục wav>
    python3 scripts/tts-piper.py --text "câu thử" --out test.wav
"""
import argparse, inspect, sys, wave
from pathlib import Path

VOICE = "vi_VN-vais1000-medium"
HF = ("https://huggingface.co/rhasspy/piper-voices/resolve/main"
      "/vi/vi_VN/vais1000/medium")


def fetch_voice(dest: Path) -> Path:
    """Tải .onnx + .onnx.json nếu chưa có. Thiếu file .json là Piper không chạy."""
    import urllib.request
    dest.mkdir(parents=True, exist_ok=True)
    onnx = dest / f"{VOICE}.onnx"
    for name in (f"{VOICE}.onnx", f"{VOICE}.onnx.json"):
        f = dest / name
        if f.exists() and f.stat().st_size > 1000:
            continue
        print(f"tải {name} …")
        urllib.request.urlretrieve(f"{HF}/{name}", f)
        print(f"  {f.stat().st_size / 1e6:.1f} MB")
    return onnx


def load_voice(onnx: Path):
    from piper import PiperVoice
    return PiperVoice.load(str(onnx))


def synth(voice, text: str, out: Path) -> bool:
    """API Piper đổi tên hàm giữa các bản — dò thay vì đoán."""
    out.parent.mkdir(parents=True, exist_ok=True)
    names = [n for n in ("synthesize_wav", "synthesize") if hasattr(voice, n)]
    if not names:
        sys.exit("PiperVoice không có synthesize/synthesize_wav — bản này lạ, "
                 f"có: {[m for m in dir(voice) if 'syn' in m.lower()]}")
    fn = getattr(voice, names[0])
    with wave.open(str(out), "wb") as wf:
        try:
            fn(text, wf)
        except TypeError:
            # Một số bản nhận (text, wav_file=...) hoặc trả generator
            sig = inspect.signature(fn)
            if "wav_file" in sig.parameters:
                fn(text, wav_file=wf)
            else:
                raise
    return out.exists() and out.stat().st_size > 1000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--text")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--voice-dir", type=Path, default=Path("models/piper"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    onnx = fetch_voice(a.voice_dir)
    print(f"giọng: {VOICE}  (CC-BY-4.0 — ghi nguồn rhasspy/piper-voices khi đăng)")
    voice = load_voice(onnx)

    if a.text:
        dest = a.out or Path("test.wav")
        ok = synth(voice, a.text, dest)
        print(("xong: " if ok else "LỖI: ") + str(dest))
        return 0 if ok else 1

    if not a.in_dir:
        sys.exit("cần --in-dir, hoặc --text")
    out_dir = a.out_dir or (a.in_dir.parent / "audio")
    files = sorted(a.in_dir.glob("*.txt"))
    if not files:
        sys.exit(f"không thấy .txt nào trong {a.in_dir}")

    ok = skip = bad = 0
    for i, f in enumerate(files, 1):
        dest = out_dir / f"{f.stem}.wav"
        if dest.exists() and not a.force:
            skip += 1
            continue
        if synth(voice, f.read_text(encoding="utf-8").strip(), dest):
            ok += 1
        else:
            bad += 1
            dest.unlink(missing_ok=True)
            print(f"  LỖI {f.stem}")
        if i % 20 == 0:
            print(f"  … {i}/{len(files)}")
    print(f"\nđọc {ok} · có sẵn {skip} · lỗi {bad} → {out_dir}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
