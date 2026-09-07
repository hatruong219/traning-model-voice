# LICENSE NOTES

Kiểm ngày 2026-09-07.

| Model | License | Thương mại |
|---|---|---|
| `F5TTS_v1_Base` (mặc định) | code MIT, checkpoint train trên Emilia CC-BY-NC | không · **và không đọc được tiếng Việt** |
| `hynt/F5-TTS-Vietnamese-ViVoice` | CC-BY-NC-SA-4.0 | **không** |
| `erax-ai/EraX-Smile-Female-F5-V1.0` | NC (tự khai kế thừa Emilia) | **không** |

**Mọi checkpoint F5-TTS tiếng Việt tìm được đều phi thương mại**, vì đều fine-tune từ
`F5TTS_Base` vốn train trên bộ Emilia (CC-BY-NC). Fine-tune từ checkpoint NC thì kết quả
vẫn NC.

## Nếu kênh có doanh thu

- **Piper** — trọng số permissive, nhưng phải tự train giọng và chất lượng thấp hơn
- **Dịch vụ trả tiền** — Vbee, FPT.AI, Zalo AI, Viettel AI: có license rõ ràng
- **edge-tts** — miễn phí, không cần key, nhưng là endpoint không chính thức của Microsoft
  và không kèm giấy phép thương mại

Rủi ro thực tế tỉ lệ với quy mô: dùng riêng cho kênh mình, không phát hành model → thấp.
Muốn bán/cho thuê lại giọng → phải xử lý tử tế.
