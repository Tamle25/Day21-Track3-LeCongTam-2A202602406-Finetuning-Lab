# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**
Điều làm tôi ngạc nhiên nhất là "silent bug" của TRL ở F-10: cờ `assistant_only_loss=True` âm thầm tính loss trên đúng 0% token vì chat template của Qwen3.5 không chứa marker `{% generation %}`. Pipeline vẫn chạy mượt mà, đường loss vẫn giảm đẹp như mơ nhưng thực chất mô hình không học được một token nào! Việc giải mã ngược token labels ở NB1 để nhìn tận mắt phần được tính loss chính là phòng tuyến sống còn giúp ngăn chặn thảm họa này.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**
Tôi mất nhiều thời gian nhất ở phần sinh văn bản đánh giá (generation passes ở NB2 và NB5) và xử lý các lỗi tương thích phần cứng (bẫy giả lập bf16 vs fp16 trên kiến trúc Turing T4 ở F-07, F-15, F-23). Ban đầu tôi dự đoán quá trình train LoRA ở NB3 và NB4 sẽ ngốn nhiều thời gian nhất, nhưng thực tế việc giải mã greedy qua 3 lượt đánh giá (baseline a, baseline b, fine-tune) mới là phần chiếm phần lớn thời gian chạy thực nghiệm.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**
Trước lab này, tôi từng có 3 định kiến sai lầm:
1. Tin rằng fine-tuning luôn áp đảo prompt engineering. Thực tế, Baseline (b) được prompt tử tế là một rào cản rất khó vượt (đạt ngay 0.760 target và giảm latency 3× trước khi train bất kỳ trọng số nào).
2. Tin rằng tăng rank $r$ là đòn bẩy chính để nâng cao năng lực mô hình. NB4 đã chứng minh vị trí gắn adapter quan trọng hơn rank rất nhiều: $r=16$ phủ đều `text-linear` đánh bại hoàn toàn $r=283$ dồn cục bộ vào `attn_only`.
3. Tin rằng train loss và perplexity giảm là mô hình đang tiến bộ. Thực tế, $r=283$ có train loss thấp nhất (0.0531) nhưng lại là do học vẹt tập train, khi đánh giá target thực tế thì thua kém cấu hình chuẩn `correct`.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**
Tôi sử dụng AI assistant để phân tích cấu trúc module của kiến trúc hybrid Gated DeltaNet / Transformer của Qwen3.5, kiểm tra các luồng dữ liệu pre-tokenized và rà soát lỗi tương thích môi trường.
Chỗ AI assistant thường sai nhất: Nó có xu hướng gợi ý bật `bf16=True` và `packing=True` theo các template hướng dẫn phổ biến năm 2026. Tuy nhiên trên GPU T4 (Turing sm_75), bf16 không được hỗ trợ phần cứng (chỉ giả lập phần mềm làm giảm hiệu năng) và gây crash GradScaler ở QLoRA 4-bit; đồng thời bật packing sẽ phá vỡ nhãn đã mask chính xác từ NB1.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**
Bước đầu tiên tuyệt đối KHÔNG PHẢI là nhảy vào train model ngay, mà là: **Thiết lập Baseline (b) bằng Prompt Engineering chỉn chu và đóng băng một tập đánh giá độc lập (holdout test set)**. Tôi sẽ đo đạc xem prompt tối ưu đã đạt yêu cầu nghiệp vụ chưa. Nếu chưa đạt và bắt buộc fine-tune, bước kỹ thuật đầu tiên bắt buộc phải làm là **kiểm chứng loss mask** bằng cách decode ngược `labels != -100` để đảm bảo loss chỉ tính trên câu trả lời mong muốn, không bao giờ tính trên câu hỏi hay dữ liệu nhiễu.
