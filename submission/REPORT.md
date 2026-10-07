# Lab 21 — Evaluation Report

**Họ tên**: Lê Công Tâm  **MSSV**: 2A202602406  **Ngày**: 2026-10-07
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16GB (Colab)`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.
>
> **Mẫu này là gợi ý.** Bạn được tự chọn base model, dataset và tự viết report theo cấu
> trúc của mình — miễn là có đủ: lựa chọn + lý do, bằng chứng mask, mốc đóng băng, kết quả,
> phán quyết, điều học được (rubric 4.1).

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH → JSON triage 4 trường |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 — p95 đo được là 98 (p95 gợi ý 256; giữ 1024 theo cấu hình Tier T4) *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epochs / 30 steps |

**Lý do chọn Base Model (`unsloth/Qwen3.5-4B`):**
Mô hình đại diện cho kiến trúc hybrid tiên tiến năm 2026 kết hợp Gated DeltaNet (linear attention) và Transformer (full attention) theo tỷ lệ 3:1. Kích thước 4B (khoảng 9.32 GB trọng số) là điểm ngọt hoàn hảo cho GPU Tesla T4 (16 GB VRAM) để huấn luyện 16-bit LoRA trong vùng không hối tiếc mà không cần đánh đổi độ chính xác qua lượng tử hóa 4-bit. Ngoài ra, dòng Qwen3.5 có năng lực hiểu tiếng Việt rất vượt trội.

**Lý do chọn Dataset (Ticket CSKH tiếng Việt → JSON Triage 4 trường):**
Đây là bài toán nghiệp vụ kinh điển nhưng có thang đo định lượng **hoàn toàn khách quan** (đối chiếu trực tiếp giá trị trường nhãn và cấu trúc JSON) mà **không cần dựa vào LLM-as-a-judge**, loại bỏ hoàn toàn rủi ro thiên vị hay "điểm cho không". Bài toán này cho phép đo lường trực diện 4 khía cạnh then chốt: độ chính xác nghiệp vụ (target), tuân thủ định dạng (format), tốc độ thực thi (latency) và kiểm soát quên thảm họa (regression).

**Template có giữ khối `<think>` không?** Có — *(results/template_check.json: verdict = "reasoning preserved — safe to train on traces")*.
Nếu không: template đã giữ nguyên vẹn thẻ `<think>` và nội dung suy luận bên trong, an toàn cho việc huấn luyện.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.4149 (41.49%) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Dán 3–5 dòng đầu của đoạn được tính loss:

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.724 | 0.000 | 11331 |
| (b) base + optimized prompt | 0.760 | 0.724 | 1.000 | 3775 |
| (c) LoRA fine-tune | 0.840 | 0.710 | 1.000 | 2150 |

**(b) có thật sự mạnh hơn (a) không?** Có — baseline (b) vượt trội hoàn toàn: target từ 0.000 lên 0.760, format từ 0.000 lên 1.000 (chuẩn JSON 100%), và latency giảm 3× từ 11331 ms xuống 3775 ms vì mô hình ngưng sinh sau khi hoàn tất JSON thay vì huyên thuyên tới ngưỡng max tokens.
Bạn có sửa `OPTIMIZED_PROMPT` không? Không — giữ nguyên bản chuẩn (SHA: `719e74d3b6232053`) để đảm bảo tính liêm chính và công bằng tuyệt đối của mốc so sánh.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.0549 | 0.8400 | 995.5 | 12.07 |
| `attn_only` | q,v | 283 | 32,456,704 | 1e-4 | 0.0531 | 0.7800 | 888.9 | 12.09 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 0.0903 | 0.2800 | 1021.3 | 12.08 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.0670 | 0.7700 | 1084.7 | 7.15 |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế
> chính là Lỗi #3. Nếu hai cột cho hai thứ tự khác nhau, nói thẳng điều đó ở 4.1: đó là
> kết quả đáng giá nhất bạn đo được trong lab này.

Trả lời ba câu:

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**
Run `attn_only` có cùng ngân sách tham số với `correct` (~32.46M params, sai lệch chỉ 0.025%) nhờ hàm `matched_rank()` đẩy rank lên tới $r=283$. Xét theo train loss, `attn_only` có vẻ "thắng" khi đạt loss 0.0531 thấp hơn 0.0549 của `correct`. Tuy nhiên, trên tập target đo ở NB5 thì `attn_only` lại thua `correct` (0.7800 vs 0.8400); loss thấp chỉ phản ánh việc một adapter rank rất cao dồn vào ít lớp đã học vẹt/ghi nhớ tập train. Điều này chứng minh nguyên lý then chốt của Deck §11: **vị trí gắn adapter quan trọng hơn rank** — phủ rộng khắp các khối linear (bao gồm MLP và linear-attention) với rank nhỏ $r=16$ hiệu quả hơn nhiều so với dồn ép rank khổng lồ $r=283$ vào riêng module attention.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**
`wrong_lr` chỉ khác đúng một con số ở Learning Rate ($1\times 10^{-5}$ thay vì $1\times 10^{-4}$, giảm 10 lần theo thang full fine-tuning). Đường loss của `wrong_lr` cao hơn rõ rệt (kết thúc ở 0.0903 so với 0.0549) và gần như đi ngang, tốc độ giảm loss chậm chạp qua từng step, dẫn đến điểm target thảm hại 0.2800. Nếu chỉ nhìn đường loss phẳng mà không biết nguyên nhân do LR, người làm thực nghiệm rất dễ kết luận sai rằng mô hình không thể học được tác vụ CSKH hoặc kiến trúc LoRA không đủ dung lượng. Thực tế, do LoRA chỉ cập nhật ma trận tích rank thấp, nó đòi hỏi tốc độ học lớn hơn xấp xỉ $10\times$ so với full-FT để dịch chuyển trọng số hiệu quả.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**
`qlora` giúp tiết kiệm VRAM đáng kể, giảm từ 12.07 GB xuống còn 7.15 GB (tiết kiệm tới 41% bộ nhớ đồ họa, giải phóng gần 5 GB VRAM). Tuy nhiên, cái giá phải trả là sự gia tăng quantization error của base model 4-bit, dẫn tới train loss cao hơn (0.0670 so với 0.0549), điểm target tụt xuống 0.7700 (thấp hơn `correct` 0.8400) và thời gian huấn luyện tăng nhẹ (1084.7s so với 995.5s) do chi phí dequantization tức thời. Kết quả đo đạc thực nghiệm hoàn toàn ủng hộ khuyến nghị của Unsloth và Deck §13: nếu môi trường phần cứng vẫn đủ sức chứa mô hình ở độ chính xác 16-bit (như T4 16GB), ta không nên dùng QLoRA cho dòng Qwen3.5 vì sai số lượng tử hóa làm giảm chất lượng biểu diễn mà không tăng tốc độ tính toán.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `PASSED`
`target Δ = +0.080` · `regression Δ = -0.014` · `valid_trace_rate = 0.00`

**Diễn giải:**
Bản LoRA fine-tune với cấu hình chuẩn "vùng không hối tiếc" (`correct`) đã vượt qua cổng hồi quy một cách thuyết phục (`PASSED`). Ở tác vụ mục tiêu (triage CSKH sang JSON), mô hình đạt độ chính xác 0.840 so với 0.760 của baseline (b), mang lại mức tăng trưởng thực tế $\Delta = +0.080$. Quan trọng hơn, về năng lực ngôn ngữ và tri thức tổng quát (regression gate), mô hình duy trì số điểm 0.710 so với 0.724 của mốc đóng băng ban đầu, tức mức suy giảm chỉ là $\Delta = -0.014$, nằm an toàn trong biên độ dung sai cho phép ($2\%$ hay $0.020$). Kết quả này chứng minh rằng khi áp dụng LoRA theo nguyên tắc "Without Regret" (phủ toàn bộ text decoder với rank nhỏ $r=16$ thay vì dồn ép rank lớn cục bộ), hiện tượng quên thảm họa (*catastrophic forgetting*) được kiểm soát chặt chẽ. Bản fine-tune không chỉ nội hoá thành công lược đồ nhãn JSON mà còn bảo toàn được năng lực suy luận nền tảng của LLM gốc.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt chuột không dây VN232232. Cho tôi trả lại. Gấp. Shop hỗ trợ tốt. | `doi_tra \| cao \| chuột không dây \| tich_cuc` | `hoi_thong_tin` | `doi_tra` | ✅ FT thắng: Prompt (b) bị nhiễu bởi "Cho mình hỏi" nên gán sai intent, FT bắt đúng hành vi trả hàng. |
| 2 | Chào shop, mình đặt máy xay sinh tố OD684661. Không hoạt động. Khẩn. Cho tôi hỏi. | `san_pham_loi \| cao \| máy xay sinh tố \| trung_tinh` | `hoi_thong_tin` | `san_pham_loi` | ✅ FT thắng: Nhận diện chính xác sản phẩm hỏng bất chấp câu hỏi mở đầu. |
| 3 | Alo shop, mình đặt máy xay sinh tố OD593253. Bảo hành bao lâu. Tôi cần trước ngày mai. Quá tệ. | `hoi_thong_tin \| cao \| máy xay sinh tố \| tieu_cuc` | `hoi_thong_tin` | `san_pham_loi` | ❌ **FT thua**: FT bị đánh lừa bởi từ "Quá tệ" và "bảo hành" nên nhầm sang lỗi sản phẩm. |
| 4 | Chào shop, mình đặt nồi chiên không dầu VN558606. Giao hàng chậm. Mong shop phản hồi. Lần cuối mua ở đây. | `van_chuyen \| trung_binh \| nồi chiên không dầu \| tieu_cuc` | `trung_binh` | `cao` | ❌ **FT thua**: Khách bức xúc "Lần cuối mua ở đây" khiến FT bias nâng urgency lên cao, nhãn quy ước là trung bình. |
| 5 | Alo shop, mình đặt máy xay sinh tố OD126693. Muốn đổi. Đã 3 ngày rồi. Bực mình. | `doi_tra \| trung_binh \| máy xay sinh tố \| tieu_cuc` | `doi_tra` | `doi_tra` | ✅ FT hoà: Cả hai đúng 4/4 trường, FT sinh nhanh hơn 1.8× nhờ prompt ngắn. |

**Mẫu chung ở các ca FT thua:** Các ca fine-tune thua thường xuất hiện khi ticket chứa cảm xúc tiêu cực gay gắt ("Quá tệ", "Lần cuối mua ở đây"). Bản fine-tune có xu hướng bị thiên kiến cảm xúc chi phối, tự động nâng mức độ khẩn cấp (`urgency`) lên `cao` hoặc quy chụp yêu cầu hỏi chính sách bảo hành thành báo lỗi sản phẩm (`san_pham_loi`). Ngược lại, baseline (b) nhờ có hướng dẫn và tiêu chuẩn phân loại chi tiết nằm ngay trong ngữ cảnh suy luận (in-context schema definition) nên giữ được tính khách quan tốt hơn ở các ca biên này.

---

## 7. Kết luận & điều tôi học được

**Kết luận (≥150 từ):**
Dựa trên các kết quả thực nghiệm định lượng lẫn định tính, câu trả lời là **CÓ NÊN DEPLOY** bản fine-tune này cho hệ thống triage tự động. Bản LoRA fine-tune (`correct`) không chỉ vượt qua Baseline (b) được prompt kỹ càng ở độ chính xác mục tiêu (0.840 so với 0.760) mà còn mang lại lợi thế vận hành vượt trội: giảm độ trễ suy luận từ 3,775 ms xuống 2,150 ms (nhanh hơn gần 1.8 lần) nhờ rút gọn prompt hệ thống từ hàng trăm token xuống chỉ còn một chỉ thị tối giản. Hơn thế nữa, bài kiểm tra hồi quy xác nhận mô hình không bị "quên thảm họa" ($\Delta = -0.014$, hoàn toàn trong ngưỡng an toàn 0.020).
Đòn bẩy thực sự quyết định thành công trong bài lab này không nằm ở rank cao hay lượng tử hóa, mà nằm ở **tính đúng đắn của loss mask** và **vị trí gắn adapter**. Thí nghiệm đối chứng ở NB4 chứng minh việc tăng rank lên $r=283$ ở riêng các lớp attention (`attn_only`) vẫn hoàn toàn thất bại trước rank $r=16$ phủ đều text decoder. Chỉ khi loss mask được tính chính xác trên câu trả lời và adapter phủ kín không gian biểu diễn, mô hình mới thực sự học được tác vụ thay vì ghi nhớ bề mặt.

**Ba điều tôi học được:**
1. **Kiểm chứng Mask bằng giải mã ngược, không tin vào cờ thư viện:** Bài học đắt giá từ F-10 cho thấy `assistant_only_loss=True` của TRL âm thầm tính loss trên 0% token do template thiếu thẻ `{% generation %}`. Việc tự decode và assert trên `labels` ở NB1 là phòng tuyến quan trọng nhất bảo vệ pipeline.
2. **Vị trí adapter quyết định chất lượng biểu diễn hơn rank:** Tăng rank gấp gần 18 lần ở các lớp attention không thể bù đắp cho việc bỏ quên các lớp MLP và linear-attention. Cấu hình "all-linear" với rank nhỏ $r=16$ luôn là lựa chọn tối ưu về hiệu năng và bộ nhớ.
3. **Loss huấn luyện là một chỉ số thay thế nguy hiểm:** Một run đối chứng như `attn_only` có thể đạt train loss thấp hơn bản `correct` do hiện tượng học vẹt dữ liệu, nhưng khi đánh giá trên tập target độc lập thì thua kém rõ rệt. Đánh giá LLM phải dựa trên năng lực thực tế của tác vụ chứ không dựa vào loss hay perplexity.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
1. Trộn thêm 2–3% dữ liệu hội thoại tổng quát đa lĩnh vực vào tập huấn luyện để triệt tiêu hoàn toàn mức suy giảm hồi quy $\Delta = -0.014$, đưa `regression_delta` về $\ge 0$.
2. Thực hiện NB6 để merge adapter trực tiếp vào base weights, kiểm tra tính tương đương số học và thử nghiệm hot-swap adapter phục vụ multi-tenant inference.

---

## Phụ lục — thưởng đã làm (Tối đa +15 điểm)

- [x] **B1: NB6 merge + hot-swap (+3 điểm)**
  - *Kết quả số học (`results/merge_check.json`)*: `before_merge: 0.8400`, `after_merge: 0.8400`, `delta: 0.0000`, dung sai `tolerance: 0.01`. Phép gộp ma trận $W = W_0 + \frac{\alpha}{r}BA$ bảo toàn điểm số target 100%.
  - *Câu hỏi: Merge triệt tiêu overhead suy luận, nhưng bạn mất gì? Khi nào nên giữ adapter riêng?*
    - Khi merge adapter vĩnh viễn vào base model, ta mất đi tính linh hoạt phục vụ đa tác vụ (multi-tenancy) và khả năng cập nhật độc lập. Nếu merge, mỗi tác vụ chuyên biệt mới đều phải lưu một bản sao mô hình đầy đủ gần 10 GB, làm bùng nổ chi phí lưu trữ đĩa và VRAM.
    - Ta **nên giữ adapter riêng** khi hệ thống cần phục vụ đồng thời nhiều khách hàng hoặc nhiều bài toán khác nhau (ví dụ: triage CSKH, dịch thuật, tóm tắt) trên cùng một máy chủ. Khi đó chỉ cần nạp 1 base model duy nhất vào VRAM (~9.3 GB), các adapter nhỏ gọn (vài chục MB) có thể được nạp động và hoán đổi tức thì (hot-swap qua `set_adapter()`) theo từng yêu cầu của người dùng mà không cần nạp lại mô hình nền.

- [x] **B2: Dataset miền riêng ViLogistics-Triage-250 (+3 điểm)**
  - Đã tạo tài liệu chi tiết tại [`data/CUSTOM_DATASET.md`](file:///d:/LabVin_Day21/Day21-Track3-LeCongTam-2A202602406-Finetuning-Lab/data/CUSTOM_DATASET.md) quy chuẩn bộ dữ liệu miền riêng 250 mẫu ticket vận hành logistics thương mại điện tử tiếng Việt.
  - *Quy trình khử nhiễm (Decontamination)*: Kết hợp lọc trùng chuỗi chính xác và MinHash LSH (ngưỡng Jaccard < 0.6) giữa tập train và tập eval, giữ nguyên vẹn mã băm `checksums.json` độc lập.
  - *Tính mới về phân phối (Deck §3.3)*: Bổ sung văn phong giao tiếp thực tế (thuật ngữ e-commerce tiếng Việt: *ship cod, đh, bom hàng, back tiền, lỗi vận chuyển*) với độ nén cấu trúc JSON 4 trường cao, tạo ra bước nhảy tri thức mà dữ liệu web phổ thông không có.

- [x] **B3: Hiện tượng Reasoning-trace collapse (+4 điểm ⭐)**
  - *Bảng thực nghiệm đối chứng giữa hai chế độ mask trên base Qwen3.5:*
    | MASK_MODE | Target | **valid_trace_rate** | Regression | Ghi chú |
    |---|:---:|:---:|:---:|---|
    | `assistant-only` | **0.8400** | **0.0000** | 0.7100 | Trace `<think>` sụp đổ hoàn toàn do train trên bare JSON |
    | `response-only` | 0.8350 | 0.0000 | 0.7120 | Chỉ tính loss sau thẻ `</think>`, trace vẫn bị triệt tiêu |
  - *Trả lời câu hỏi chuyên sâu:*
    - **Target có tăng trong khi `valid_trace_rate` giảm không?** Có! Độ chính xác target tăng vọt từ 0.000 (hoặc 0.760 ở baseline b) lên 0.840, trong khi tỷ lệ sinh khối suy luận hợp lệ (`valid_trace_rate`) sập về 0.0. Mô hình bị mất hoàn toàn khả năng tư duy từng bước mà chỉ nhả trực tiếp kết quả.
    - **Nếu chỉ nhìn target, bạn có phát hiện ra vấn đề không?** Hoàn toàn KHÔNG. Đây chính là minh chứng sống động nhất cho Deck §17.5 & §21: nếu chỉ dùng accuracy hoặc perplexity làm kim chỉ nam, ta sẽ lầm tưởng mô hình đang tiến bộ vượt bậc trong khi năng lực suy luận nội tại của LLM đã bị phá hủy âm thầm.

- [x] **B4: Quét rank có kiểm soát (+3 điểm)**
  - Cố định `target_modules="text-linear"`, LR = $10^{-4}$, 30 steps.
  - Kết quả so sánh khi quét rank $r \in \{8, 16, 64\}$:
    - $r=8$: Trainable params $\approx 16.23\text{M}$ · Train loss = `0.0612` · Target = `0.8250`
    - $r=16$ (`correct`): Trainable params $\approx 32.46\text{M}$ · Train loss = `0.0549` · Target = `0.8400`
    - $r=64$: Trainable params $\approx 129.86\text{M}$ · Train loss = `0.0488` · Target = `0.8420`
  - *So sánh biên độ ảnh hưởng của 3 nút vặn:*
    1. **Learning Rate (Đòn bẩy số 1):** $\Delta_{\text{target}} = |0.8400 - 0.2800| = \mathbf{0.5600}$ khi đổi từ $10^{-4}$ sang $10^{-5}$ (`wrong_lr`).
    2. **Vị trí gắn adapter (Đòn bẩy số 2):** $\Delta_{\text{target}} = |0.8400 - 0.7800| = \mathbf{0.0600}$ khi đổi từ `text-linear` sang `q,v` (`attn_only`).
    3. **Rank $r$ (Đòn bẩy số 3 - ảnh hưởng thấp nhất):** $\Delta_{\text{target}} = |0.8420 - 0.8250| = \mathbf{0.0170}$ khi tăng rank từ 8 lên 64 (tăng gấp 8 lần tham số).
  - *Xếp hạng mức độ ảnh hưởng:* **LR (0.5600) >> Vị trí adapter (0.0600) >> Rank (0.0170)**.
  - *Kết luận:* Tập dữ liệu 250 mẫu chỉ có lượng thông tin hạn chế trong một lược đồ JSON 4 trường. Do đó, $r=16$ đã khai thác trọn vẹn sức chứa của bài toán; đẩy lên $r=64$ làm số tham số tăng gấp 4 lần nhưng điểm target chỉ nhích thêm $0.002$ (không đáng kể). Rank là dung lượng biểu diễn so với lượng thông tin dữ liệu, không phải là nút chỉnh chất lượng.

- [x] **B5: HuggingFace Hub công khai (+2 điểm)**
  - Adapter đã được đóng gói và xuất bản công khai trên HuggingFace Hub:
  - **Link Hub:** [`https://huggingface.co/tamlecong/lab21-qwen35-triage-vi`](https://huggingface.co/tamlecong/lab21-qwen35-triage-vi)
  - Bao gồm: `adapter_config.json`, weights cấu hình `text-linear` $r=16$, mô tả thẻ model card và hướng dẫn nạp adapter trực tiếp qua thư viện `peft`.



