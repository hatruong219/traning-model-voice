# PLAN — dựng giọng TTS tiếng Việt từ video của chính chủ kênh

Chạy trên **máy có GPU**. Máy dev hiện tại không có GPU nên plan này viết để mang sang máy khác.

**Nguyên tắc xuyên suốt:** đi qua Phase 3 (zero-shot) trước. Nếu zero-shot đủ tốt thì **dừng
luôn, không train** — tiết kiệm cả ngày và tiền GPU. Fine-tune chỉ khi zero-shot không đạt.

---

## Bối cảnh đã chốt

| | |
|---|---|
| Nguồn audio | **video đã mix nhạc nền** → bắt buộc tách giọng |
| Loại giọng | **giọng người thật của chủ kênh** → fine-tune có nghĩa, không vướng quyền người khác |
| Đích tích hợp | `manga-narration/scripts/tts-*.py`, đọc `results/tts-lines/Sxx.txt` → `results/audio/Sxx.mp3` |
| Đã đo | 1.658 âm tiết (~9 phút) phủ 27/28 âm đầu, 6/6 thanh |

## License — mức độ quan tâm tỉ lệ với quy mô, không phải cửa chặn

**Giọng và bản thu của bạn: 100% của bạn, không có vấn đề gì.** Chuyện license nằm ở tầng khác:
fine-tune không tạo model từ số 0, nó nhích trọng số của một checkpoint có sẵn, nên file ra là
bản phái sinh của checkpoint đó. Giọng bạn ở tầng trên; tầng dưới vẫn là của người khác.

Nhưng đây **không phải chuyện nhị phân**:

| Bạn làm gì | Rủi ro thật |
|---|---|
| Zero-shot: chỉ dùng model sinh audio, không sửa không phát hành | **rất thấp** |
| Fine-tune, dùng riêng cho kênh mình, không chia sẻ model | thấp |
| Kênh lớn dần, có người soi | tăng |
| Muốn bán / cho thuê lại giọng hoặc cả dây chuyền | **đáng xử lý tử tế** |

Và phải nói thẳng: **câu hỏi pháp lý này chưa ngã ngũ.** Trọng số model có được bảo hộ bản quyền
không, license phi thương mại của *dữ liệu train* có lan sang *output* không — chưa có lời giải
thống nhất, khác nhau theo từng nước. Đây không phải tư vấn pháp lý.

**Một ngoại lệ đáng tránh thật:** CPML của XTTS-v2 (và viXTTS kế thừa) **nói thẳng là phi thương
mại** — đó là điều khoản rõ ràng, không phải suy diễn. Nếu kênh có doanh thu thì chọn checkpoint khác.

| Model | Ghi chú |
|---|---|
| XTTS-v2 / viXTTS | CPML nói rõ phi thương mại → tránh nếu có doanh thu |
| F5-TTS | code MIT; checkpoint gốc train trên bộ CC-BY-NC → mờ, chưa có tiền lệ rõ |
| Piper | trọng số phần lớn permissive → sạch nhất, nhưng chất lượng thấp hơn |

→ Ở Phase 0 chỉ cần **5 phút**: ghi tên checkpoint đã tải + license của nó vào
`data/LICENSE-NOTES.md`. Không phải để chặn, mà để 6 tháng sau kênh lớn lên thì bạn còn biết
mình đã dùng gì và đổi được. Đừng để nó cản việc bắt tay làm.

## Phase 0 — Môi trường

**Mục tiêu:** máy GPU sẵn sàng, checkpoint đã chốt license.

```bash
nvidia-smi                      # cần >= 8 GB VRAM cho inference, >= 16 GB cho fine-tune
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

- Python **3.10–3.12** cho stack TTS (nhiều package chưa theo kịp 3.13+).
- Cài: `torch` (bản CUDA khớp driver) · `ffmpeg` · `audio-separator[gpu]` hoặc `demucs` ·
  `faster-whisper` hoặc `transformers` (cho PhoWhisper) · repo model đã chốt.

**Xong khi:** `torch.cuda.is_available()` là `True`, và `data/LICENSE-NOTES.md` ghi một dòng:
checkpoint nào + license gì. Ghi để sau này tra lại được, không phải để chặn.

---

## Phase 1 — Tách giọng khỏi nhạc nền

**Mục tiêu:** từ video → file giọng sạch, không còn nhạc.

```bash
# 1. rút audio, chuẩn hoá về 24 kHz mono (khớp sample rate model)
ffmpeg -i video.mp4 -vn -ac 1 -ar 24000 -c:a pcm_s16le data/raw/ep01.wav

# 2. tách giọng — thử vài model, nghe rồi chọn
audio-separator data/raw/ep01.wav --model_filename UVR-MDX-NET-Voc_FT.onnx \
    --output_dir data/vocals
```

**Rủi ro lớn nhất của cả plan:** nhạc nền để lại artifact, và model sẽ **học cả artifact đó** —
giọng ra nghe rít hoặc có tiếng nền ma.

Cách giảm:
- Ưu tiên đoạn **không có nhạc** (mở đầu, kết, đoạn lặng nhạc) — lọc bằng Phase 2.
- Thử ít nhất 2 model tách (`UVR-MDX-NET-Voc_FT`, `Kim_Vocal_2`, `htdemucs`), **nghe bằng tai**
  rồi chọn, đừng tin điểm số.
- Nếu chủ kênh còn **file thu gốc trước khi mix** → dùng cái đó, bỏ hẳn Phase này.

**Xong khi:** nghe 3 đoạn ngẫu nhiên 10 giây, không nghe thấy nhạc và không nghe thấy tiếng rít.

---

## Phase 2 — Cắt câu, phiên âm, dựng dataset

**Mục tiêu:** `data/dataset/wavs/*.wav` + `metadata.csv` dạng `tên_file|transcript`.

1. **VAD cắt câu** — Silero VAD. Giữ đoạn **2–12 giây**, bỏ ngắn hơn và dài hơn.
2. **Phiên âm** — PhoWhisper-large (chuyên tiếng Việt, tốt hơn Whisper thường cho tiếng Việt)
   hoặc `faster-whisper large-v3` với `language="vi"`.
3. **Lọc rác** — bỏ đoạn có: SNR thấp · transcript rỗng hoặc 1 từ · tỉ lệ ký tự/giây bất thường
   (dấu hiệu phiên âm sai) · nhạc còn sót.
4. **Chuẩn hoá text** — số thành chữ, bỏ ký tự lạ, thống nhất dấu câu.

**Xong khi:** ≥ 30 phút audio sạch (mục tiêu 60 phút), và **đọc tay 20 dòng transcript ngẫu
nhiên** thấy khớp audio. Phiên âm sai → model học phát âm sai, đây là lỗi âm thầm và đắt nhất.

---

## Phase 3 — Đo độ phủ và bù chỗ mỏng

**Mục tiêu:** biết dataset thiếu âm gì, bù bằng vài câu có chủ đích thay vì đọc thêm hàng giờ.

```bash
python3 scripts/check-phonetic-coverage.py data/dataset/metadata.csv --min 5
```

Script này **chạy CPU, không cần GPU** — làm được trước khi sang máy GPU.

Đã đo trên text mẫu 9 phút: phủ 27/28 âm đầu, 6/6 thanh, mỏng ở thanh `ngã`/`hỏi` và âm đầu `p`.

**Xong khi:** đủ 6 thanh với mỗi thanh ≥ 200 lần · âm đầu phủ ≥ 26/28 · không quá 40 ô
âm-đầu×thanh mỏng. Còn thiếu thì viết vài câu nhắm đúng ô thiếu, thu bổ sung ~5 phút.

---

## Phase 4 — CỬA QUYẾT ĐỊNH: thử zero-shot trước

**Mục tiêu:** biết có cần train hay không. **Đừng bỏ qua phase này.**

Model 2024+ clone zero-shot chỉ cần **10–30 giây** mẫu, không train gì.

1. Chọn 1 đoạn **sạch nhất, 20–30 giây**, giọng đều, không nhạc.
2. Tổng hợp thử 5 đoạn từ `manga-narration/.../tts-lines/` — chọn cả đoạn ngắn và đoạn dài.
3. **Nghe và so với giọng thật.**

| Kết quả | Làm gì |
|---|---|
| Nghe ra là giọng mình, prosody ổn | **DỪNG. Không train.** Sang Phase 6 |
| Giống timbre nhưng nhịp máy móc | thử tinh chỉnh tham số trước, rồi mới fine-tune |
| Không ra giọng mình | Phase 5 |

Ghi kết quả vào `data/zero-shot-notes.md` kèm file nghe thử, để lần sau không thử lại từ đầu.

---

## Phase 5 — Fine-tune (chỉ khi Phase 4 không đạt)

**Mục tiêu:** checkpoint mang chất giọng chủ kênh.

- Bắt đầu từ checkpoint tiếng Việt đã chốt ở Phase 0.
- Dataset từ Phase 2, split train/val ~95/5.
- Learning rate thấp (fine-tune, không train từ đầu), theo mặc định của repo rồi giảm nếu overfit.
- **Lưu checkpoint theo từng mốc** và nghe thử ở mỗi mốc — loss giảm không có nghĩa giọng hay hơn.
- Dừng khi tai không thấy khá thêm, không chờ loss đáy.

Thời gian ước lượng: 1–3 giờ trên T4/A10 với 30–60 phút data. Colab free (T4) đủ nhưng
có giới hạn session — lưu checkpoint ra Drive thường xuyên.

**Xong khi:** nghe 10 đoạn, ≥ 8 đoạn không phân biệt được với giọng thật, và **không đoạn nào
có artifact nhạc nền**.

---

## Phase 6 — Nối vào dây chuyền manga-narration

**Mục tiêu:** thay `tts-edge.py` bằng giọng riêng, giữ nguyên phần còn lại.

Viết `manga-narration/scripts/tts-custom.py` **cùng giao diện** với `tts-edge.py`:

```
đọc  results/tts-lines/Sxx.txt  →  ghi  results/audio/Sxx.mp3
cờ:  --only Sxx   --force   --rate
```

Giữ đúng giao diện thì `build-video.py` và `retime-from-audio.py` không phải sửa dòng nào.

Sau đó **đo lại tốc độ đọc thật** của giọng mới:

```bash
python3 scripts/build-narration.py <results> --wps <đo được> --overhead <đo được>
```

Đã đo cho edge-tts giọng nam: `overhead 1,1s + 3,8 từ/giây`. Giọng mới gần như chắc chắn khác —
đo lại bằng 8 mẫu dài ngắn khác nhau rồi cập nhật hai con số đó.

**Xong khi:** chạy `pipeline.sh` một chapter từ đầu đến cuối, ra `video-track.mp4` khớp giọng mới.

---

## Thứ tự thực hiện

```
Phase 0 (GPU sẵn sàng + chốt license)
   └→ Phase 1 (tách nhạc)  ← rủi ro chất lượng lớn nhất
        └→ Phase 2 (dataset)
             └→ Phase 3 (đo phủ âm)   ← chạy được trên CPU, làm sớm
                  └→ Phase 4 (ZERO-SHOT)  ← CỬA QUYẾT ĐỊNH
                       ├─ đạt → Phase 6
                       └─ không đạt → Phase 5 → Phase 6
```

## Rủi ro, xếp theo mức độ

| Rủi ro | Ảnh hưởng | Giảm thế nào |
|---|---|---|
| Artifact nhạc nền lọt vào dataset | giọng ra rít, không sửa được sau khi train | ưu tiên đoạn không nhạc; thử nhiều model tách; nghe tay |
| Checkpoint gốc là non-commercial | chỉ thành vấn đề khi kênh lớn hoặc muốn bán lại giọng | ghi lại license ở Phase 0 để đổi được về sau; tránh XTTS-v2/viXTTS nếu có doanh thu |
| Phiên âm sai | model học phát âm sai, lỗi âm thầm | đọc tay 20 dòng ngẫu nhiên ở Phase 2 |
| Train mà lẽ ra không cần | mất 1 ngày + tiền GPU | **bắt buộc qua Phase 4 trước** |
| Colab ngắt session giữa train | mất tiến độ | lưu checkpoint ra Drive theo mốc |
| Tên riêng nước ngoài đọc sai | không model nào tự đúng | dùng `series/<BỘ>/tts-pronounce.tsv` đã có |

## Câu chưa có lời

- Chủ kênh có còn **file thu gốc trước khi mix nhạc** không? Có thì bỏ được Phase 1 — cải thiện
  chất lượng nhiều nhất trong toàn plan.
- Tổng thời lượng video có giọng là bao nhiêu? Dưới 30 phút thì zero-shot là lựa chọn duy nhất hợp lý.
- Kênh đã/sẽ có doanh thu, và có ý định bán/cho thuê lại giọng không? Chỉ ảnh hưởng việc chọn
  checkpoint, không ảnh hưởng việc bắt đầu.
- GPU dự kiến là gì (VRAM bao nhiêu)? Dưới 16 GB thì Phase 5 phải giảm batch size.
