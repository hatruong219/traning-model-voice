"""Gọi F5-TTS ép float32 — bắt buộc trên GPU dòng GTX 16xx.

Vì sao cần: GTX 16xx (Turing không tensor core) sinh NaN khi chạy fp16. NaN ghi ra wav
thành hằng số bão hoà: file "đầy" nhưng mean == max == 0 dB và KHÔNG có tiếng.
`f5-tts_infer-cli` không có cờ đổi dtype nên phải gọi API trực tiếp.

Chữ ký F5TTS.__init__ khác nhau giữa các phiên bản, nên script tự dò bằng inspect
thay vì đoán tham số.

Chạy:
    python3 scripts/tts-f5.py --ref data/ref/ref.wav --ref-text-file data/ref/ref.txt \
        --text "câu cần đọc" --out data/zeroshot/test.wav
    python3 scripts/tts-f5.py ... --in-dir <thư mục .txt> --out-dir <thư mục wav>
"""
import argparse, inspect, sys
from pathlib import Path

import numpy as np
import torch


VI_CHARS = "ăâđêôơưáàảãạếềệốồộớờợứừựíìỉĩị"


def check_vocab(vocab: Path) -> None:
    """Vocab không có ký tự tiếng Việt thì model KHÔNG THỂ đọc tiếng Việt.

    F5TTS_v1_Base đi kèm vocab tiếng Anh/Trung. Chạy text tiếng Việt qua đó sẽ ra
    âm thanh có dao động nhưng vô nghĩa — chữ có dấu bị bỏ hoặc map sai token.
    """
    if not vocab.exists():
        print(f"(!) không thấy vocab: {vocab}")
        return
    txt = vocab.read_text(encoding="utf-8", errors="ignore")
    hit = sum(1 for c in VI_CHARS if c in txt)
    if hit < len(VI_CHARS) // 2:
        print(f"(!) VOCAB KHÔNG PHẢI TIẾNG VIỆT — chỉ {hit}/{len(VI_CHARS)} ký tự có dấu.")
        print("    Model này không đọc được tiếng Việt, output sẽ vô nghĩa.")
        print("    Cần checkpoint tiếng Việt: --ckpt <file.safetensors> --vocab <vocab.txt>")
    else:
        print(f"vocab: {hit}/{len(VI_CHARS)} ký tự tiếng Việt — OK")


def build_tts(device: str, ckpt: Path | None, vocab: Path | None):
    from f5_tts.api import F5TTS
    sig = inspect.signature(F5TTS.__init__)
    kw, notes = {}, []
    if "device" in sig.parameters:
        kw["device"] = device
    # Checkpoint + vocab tiếng Việt. Tên tham số khác nhau giữa các bản nên dò.
    if ckpt:
        for name in ("ckpt_file", "ckpt_path", "model_path"):
            if name in sig.parameters:
                kw[name] = str(ckpt); notes.append(f"{name}={ckpt.name}"); break
    if vocab:
        for name in ("vocab_file", "vocab_path"):
            if name in sig.parameters:
                kw[name] = str(vocab); notes.append(f"{name}={vocab.name}"); break
        check_vocab(vocab)
    # Ép fp32 qua bất kỳ tên tham số nào phiên bản này dùng.
    for name in ("dtype", "torch_dtype", "precision"):
        if name in sig.parameters:
            kw[name] = torch.float32 if name != "precision" else "fp32"
            notes.append(f"{name}=float32")
            break
    else:
        notes.append("API không nhận dtype → đặt torch default dtype")
        torch.set_default_dtype(torch.float32)
    print(f"F5TTS({', '.join(f'{k}={v}' for k, v in kw.items())})  · {'; '.join(notes)}")
    return F5TTS(**kw)


def synth(tts, ref: Path, ref_text: str, text: str, out: Path) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    kw = dict(ref_file=str(ref), ref_text=ref_text, gen_text=text, file_wave=str(out))
    sig = inspect.signature(tts.infer)
    kw = {k: v for k, v in kw.items() if k in sig.parameters}
    with torch.inference_mode():
        tts.infer(**kw)
    if not out.exists() or out.stat().st_size < 1000:
        print("   (!) không ra file")
        return False
    import soundfile as sf
    y, _ = sf.read(out)
    if not np.isfinite(y).all():
        print("   (!) waveform có NaN/Inf")
        return False
    if float(np.std(y)) < 1e-4:
        print(f"   (!) waveform gần như hằng số (std={np.std(y):.2e}) → không có tiếng")
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--ref-text-file", type=Path, required=True)
    ap.add_argument("--text")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--in-dir", type=Path, help="đọc mọi .txt trong thư mục")
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--ckpt", type=Path, help="checkpoint tiếng Việt (.safetensors/.pt)")
    ap.add_argument("--vocab", type=Path, help="vocab.txt của checkpoint đó")
    a = ap.parse_args()

    ref_text = a.ref_text_file.read_text(encoding="utf-8").strip()
    if a.device == "cuda" and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    tts = build_tts(a.device, a.ckpt, a.vocab)

    if a.in_dir:
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
            print(f"[{i}/{len(files)}] {f.stem}")
            if synth(tts, a.ref, ref_text, f.read_text(encoding="utf-8").strip(), dest):
                ok += 1
            else:
                bad += 1
                dest.unlink(missing_ok=True)
        print(f"\nđọc {ok} · có sẵn {skip} · lỗi {bad} → {out_dir}")
        return 0 if bad == 0 else 1

    if not (a.text and a.out):
        sys.exit("cần --text và --out, hoặc --in-dir")
    return 0 if synth(tts, a.ref, ref_text, a.text, a.out) else 1


if __name__ == "__main__":
    sys.exit(main())
