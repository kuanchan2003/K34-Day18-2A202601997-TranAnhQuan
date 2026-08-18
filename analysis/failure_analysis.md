# Failure Analysis — Lab 18: Production RAG

**Sinh viên:** Trần Anh Quân

**Mã sinh viên:** 2A202601997

**Phạm vi:** M1–M5

## RAGAS Scores



| Metric | Naive Baseline | Production estimate | Δ |
|---|---:|---:|---:|
| Faithfulness | 0.8208 | 0.9125 | +0.0917 |
| Answer Relevancy | 0.7658 | 0.8726 | +0.1068 |
| Context Precision | 0.9250 | 0.9000 | -0.0250 |
| Context Recall | 0.9250 | 0.9500 | +0.0250 |

Context precision có thể giảm nhẹ vì parent retrieval đưa nhiều ngữ cảnh hơn vào bước sinh đáp án. Đổi lại, context recall và faithfulness được kỳ vọng tăng nhờ hybrid retrieval, reranking và prompt xử lý phiên bản.

## Bottom-5 Failures

### #1 — Tính phí tạm ứng quá hạn

- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Quá hạn 5 ngày; phí 2%/tháng là 300.000 VNĐ/tháng, pro-rata khoảng 50.000 VNĐ cho 5 ngày.
- **Got:** Bị tính phí quá hạn 300.000 VNĐ.
- **Worst metric:** Answer Relevancy — 0.61.
- **Error Tree:** Output sai một phần → Context đúng → Query retrieval đúng → Reasoning số học thiếu bước pro-rata.
- **Root cause:** Generator sao chép mức phí tháng, không quy đổi theo số ngày thực tế.
- **Suggested fix:** Tách các biến số, yêu cầu xuất công thức `15.000.000 × 2% × 5/30`, kiểm tra lại đơn vị trước khi trả lời.

### #2 — Xung đột phiên bản phép năm

- **Question:** Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?
- **Expected:** Bản v2024 quy định 1 ngày/3 năm; bản v2023 1 ngày/5 năm đã bị thay thế.
- **Got:** Cộng 1 ngày sau mỗi 5 năm.
- **Worst metric:** Faithfulness — 0.64.
- **Error Tree:** Output sai → Context có cả cũ và mới → Query đúng → Version resolution sai.
- **Root cause:** Chunk cũ có lexical match mạnh và không có bộ lọc `superseded`.
- **Suggested fix:** Trích `version`, `effective_date`, `status`; lọc bản hết hiệu lực trước RRF hoặc cộng boost cho bản hiện hành.

### #3 — Mua laptop 30 triệu

- **Question:** Nếu cần mua laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Director phê duyệt, CNTT xác nhận cấu hình, kèm ít nhất 3 báo giá.
- **Got:** Director phê duyệt và CNTT xác nhận cấu hình.
- **Worst metric:** Context Recall — 0.67.
- **Error Tree:** Output thiếu ý → Context thiếu quy định báo giá → Query multi-hop chưa tách → Retrieval failure.
- **Root cause:** Một query phải nối ba điều kiện ở các đoạn khác nhau.
- **Suggested fix:** Query decomposition thành `ngưỡng phê duyệt`, `thiết bị CNTT`, `số báo giá`, rerank từng nhánh rồi hợp nhất.

### #4 — Phân loại thông tin lương

- **Question:** Thông tin lương thuộc cấp độ phân loại dữ liệu nào?
- **Expected:** Bí mật, cấp 3; mã hóa khi truyền và need-to-know.
- **Got:** Dữ liệu bí mật, không được chia sẻ với đồng nghiệp.
- **Worst metric:** Context Precision — 0.70.
- **Error Tree:** Output đúng nhưng thiếu chi tiết → Có policy lương → Thiếu ưu tiên policy phân loại → Ranking failure.
- **Root cause:** Chunk chính sách lương lấn át chunk định nghĩa cấp độ dữ liệu.
- **Suggested fix:** Boost cụm từ “cấp độ phân loại”, giữ đa dạng nguồn trong top contexts.

### #5 — Senior 9 năm thâm niên

- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày và lương trong khoảng nào?
- **Expected:** 18 ngày; 20–35 triệu VNĐ/tháng.
- **Got:** 18 ngày và 20–35 triệu VNĐ/tháng.
- **Worst metric:** Context Precision — 0.74.
- **Error Tree:** Output đúng → Context đúng nhưng dư → Query multi-hop → Fusion đưa thêm dải lương không liên quan.
- **Root cause:** Hai sub-topic làm top-k chứa nhiều chunk lương khác cấp bậc.
- **Suggested fix:** Rerank từng sub-query, lấy tối đa một parent cho phép năm và một parent cho dải lương Senior.

## Case Study

**Chọn:** Xung đột chính sách phép năm v2023/v2024.

1. **Output đúng?** Không; dùng mốc 5 năm của bản cũ.
2. **Context đúng?** Có tài liệu liên quan nhưng chứa đồng thời hai phiên bản.
3. **Query rewrite OK?** Query đúng chủ đề nhưng chưa thêm ràng buộc “hiện hành/mới nhất”.
4. **Fix ở bước:** Enrichment metadata và filtering trước fusion; prompt chỉ là lớp bảo vệ cuối.

Nếu có thêm một giờ, ưu tiên xây dựng version resolver dựa trên `effective_date/status`, sau đó chạy ablation test Dense → Hybrid → Hybrid + rerank → Full để đo đóng góp thật của từng module.
