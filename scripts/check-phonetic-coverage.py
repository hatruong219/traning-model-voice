"""Đo độ phủ ngữ âm tiếng Việt của một tập text đọc — cho việc chuẩn bị dataset TTS.

Vì sao cần: không ai đọc hết được từ vựng tiếng Việt, và cũng không cần. TTS neural học
ánh xạ âm vị → âm thanh nên tổng hợp được âm tiết chưa từng nghe. Yêu cầu thật là phủ đủ
BỘ ÂM (âm đầu × vần × thanh), không phải phủ đủ TỪ.

File này đếm xem tập text đã phủ bao nhiêu, và chỉ ra chỗ mỏng cần đọc thêm.

Chạy:
    python3 scripts/check-phonetic-coverage.py series/TWB/C1/results/narration.tsv
    python3 scripts/check-phonetic-coverage.py <file.txt> --min 5
"""
import argparse, re, sys, unicodedata
from collections import Counter
from pathlib import Path

# Âm đầu tiếng Việt — xếp dài trước để "ngh" không bị khớp thành "ng"
ONSETS = ["ngh", "ng", "nh", "ch", "gh", "gi", "kh", "ph", "qu", "th", "tr",
          "b", "c", "d", "đ", "g", "h", "k", "l", "m", "n", "p", "r", "s", "t", "v", "x"]
TONE_MARKS = {0x0300: "huyền", 0x0301: "sắc", 0x0309: "hỏi",
              0x0303: "ngã", 0x0323: "nặng"}


def strip_tone(syl: str) -> tuple[str, str]:
    """Trả về (âm tiết không thanh, tên thanh)."""
    d = unicodedata.normalize("NFD", syl)
    tone = "ngang"
    out = []
    for ch in d:
        cp = ord(ch)
        if cp in TONE_MARKS:
            tone = TONE_MARKS[cp]
        else:
            out.append(ch)
    return unicodedata.normalize("NFC", "".join(out)), tone


def split_syllable(syl: str) -> tuple[str, str]:
    """Tách âm đầu và vần. Âm đầu rỗng ghi là 'Ø'."""
    for o in ONSETS:
        if syl.startswith(o):
            return o, syl[len(o):]
    return "Ø", syl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--min", type=int, default=3,
                    help="dưới số lần này thì coi là MỎNG, cần đọc thêm")
    a = ap.parse_args()

    raw = a.path.read_text(encoding="utf-8")
    if a.path.suffix == ".tsv":
        raw = "\n".join(l.split("\t")[-1] for l in raw.splitlines()
                        if l.strip() and not l.lstrip().startswith("#"))

    sylls = [w.lower() for w in re.findall(r"[^\W\d_]+", raw, re.UNICODE)]
    if not sylls:
        sys.exit("không tìm được âm tiết nào")

    onsets, rhymes, tones, combos = Counter(), Counter(), Counter(), Counter()
    for s in sylls:
        base, tone = strip_tone(s)
        o, r = split_syllable(base)
        onsets[o] += 1
        rhymes[r] += 1
        tones[tone] += 1
        combos[(o, tone)] += 1

    print(f"{len(sylls)} âm tiết · {len(set(sylls))} âm tiết khác nhau")

    print(f"\n=== THANH ĐIỆU (6 thanh, cần đủ cả 6) ===")
    for t in ("ngang", "huyền", "sắc", "hỏi", "ngã", "nặng"):
        n = tones.get(t, 0)
        bar = "█" * max(1, int(n / max(tones.values()) * 28)) if n else ""
        flag = "" if n >= a.min * 10 else "  ← MỎNG"
        print(f"  {t:<7} {n:>5} {bar}{flag}")

    print(f"\n=== ÂM ĐẦU: phủ {len(onsets)}/{len(ONSETS) + 1} ===")
    miss = [o for o in ONSETS + ["Ø"] if o not in onsets]
    thin = sorted([(o, n) for o, n in onsets.items() if n < a.min], key=lambda x: x[1])
    if miss:
        print(f"  THIẾU HẲN ({len(miss)}): {' '.join(miss)}")
    if thin:
        print(f"  MỎNG (<{a.min} lần): " + " ".join(f"{o}({n})" for o, n in thin))
    if not miss and not thin:
        print("  đủ cả bộ")

    print(f"\n=== VẦN: {len(rhymes)} vần khác nhau ===")
    thin_r = sorted([(r, n) for r, n in rhymes.items() if n == 1])
    print(f"  chỉ xuất hiện 1 lần: {len(thin_r)} vần"
          + (f" — vd {' '.join(r for r, _ in thin_r[:12])}" if thin_r else ""))

    print(f"\n=== ÂM ĐẦU × THANH: phủ {len(combos)}/{(len(ONSETS) + 1) * 6} ô ===")
    weak = [f"{o}+{t}" for (o, t), n in combos.items() if n < a.min]
    print(f"  ô có dữ liệu nhưng MỎNG (<{a.min}): {len(weak)}")
    if weak:
        print("  " + " ".join(weak[:20]) + (" …" if len(weak) > 20 else ""))

    est_min = len(sylls) / 180
    print(f"\nƯớc lượng thời lượng đọc: ~{est_min:.0f} phút "
          f"(180 âm tiết/phút cho giọng kể)")
    print("Fine-tune giọng cần ~30-60 phút audio. Phủ âm đầu/thanh đủ thì "
          "âm tiết lạ vẫn tổng hợp được — không cần đọc hết từ vựng.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
