# Individual Reflection — Lab 18

**Tên:** Trần Anh Quân

**Mã sinh viên:** 2A202601997

**Module phụ trách:** M1–M5

## 1. Mapping bài giảng vào code

| Lecture Concept | Module | Hàm cụ thể | Observation |
|---|---|---|---|
| Semantic chunking | M1 | `chunk_semantic()` | Dùng cosine similarity giữa câu kề nhau; threshold cao tạo nhiều chunk chính xác hơn nhưng giảm context mỗi chunk. |
| Parent-child retrieval | M1 | `chunk_hierarchical()` | Child giới hạn 256 ký tự giúp matching; `parent_id` duy nhất theo source cho phép trả parent 2.048 ký tự khi sinh đáp án. |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` | RRF hợp nhất rank thay vì score khác thang đo, giúp BM25 tiếng Việt và BGE-M3 bổ sung cho nhau. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Cross-encoder chấm trực tiếp cặp query-document, giảm top-20 xuống top-3; đổi lại latency cao hơn bi-encoder. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Faithfulness đo bám context; relevancy đo đúng câu hỏi; precision/recall tách chất lượng retrieval thành độ sạch và độ phủ. |
| Contextual enrichment | M5 | `_enrich_single_call()` | Một API call sinh context, summary, HyQA và metadata; fallback xác định giúp pipeline không hỏng khi thiếu API. |

## 2. Khó khăn và cách giải quyết

### Lỗi 1: console Windows

- **Exact error:** `UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f50d'`.
- **Debug:** Chạy `python check_lab.py`, xác định lỗi xảy ra trước phần kiểm tra file do CP1252 không in được emoji.
- **Giải quyết:** Cấu hình `stdout/stderr.reconfigure(encoding="utf-8", errors="replace")` cho các entry point.

### Lỗi 2: parser kết quả pytest

- **Exact error:** `invalid literal for int() with base 10: '=================='`.
- **Debug:** Pytest có warning sau dòng summary, trong khi `check_lab.py` luôn parse dòng cuối.
- **Giải quyết:** Dùng regex tìm `N passed` và `N failed` trên toàn bộ stdout.

### Khó khăn kiến trúc

- Parent-child scaffold ban đầu chỉ index child rồi cũng trả child, chưa thực hiện đúng “retrieve child → return parent”.
- Đã lưu `parent_text` trong metadata, de-duplicate parent sau rerank và dùng parent làm context generation.
- Policy có nhiều phiên bản khiến relevance model có thể chọn bản cũ; giải pháp tiếp theo là metadata validity filtering.

**Kết quả kiểm thử:** 37/37 tests pass. 

## 3. Action Plan cho project cá nhân

### Project: Vietnamese HR Policy Assistant

#### Hiện tại

- Pipeline: paragraph chunks + dense retrieval + LLM trả lời.
- Known issues: nhầm policy cũ/mới, câu multi-hop thiếu ý, không có regression evaluation và latency report.

#### Plan áp dụng

1. [ ] **Chunking:** parent-child theo section; child 256–384 tokens, parent 1.500–2.000 tokens để cân bằng precision/context.
2. [ ] **Search:** BM25 Vietnamese + BGE-M3 dense + RRF vì policy có cả thuật ngữ chính xác và cách hỏi tự nhiên.
3. [ ] **Reranking:** dùng BGE reranker cho top-20 → top-3; cân nhắc FlashRank nếu CPU latency vượt SLA.
4. [ ] **Evaluation:** RAGAS 4 metrics kết hợp exact-match cho số/đơn vị, negation accuracy và version accuracy.
5. [ ] **Enrichment:** contextual prepend + metadata `version`, `effective_date`, `status`, `department`; combined call để giảm chi phí.

#### Timeline

- **Tuần 1:** Chuẩn hóa corpus, OCR PDF scan, version metadata và test set tối thiểu 50 câu.
- **Tuần 2:** Implement parent-child, hybrid retrieval, RRF; đo recall@k và latency trên CPU.
- **Tuần 3:** Thêm reranking, query decomposition và generator guardrails cho phủ định/số học.
- **Tuần 4:** Chạy RAGAS, failure analysis, ablation tests và tối ưu theo SLA.

#### Tiêu chí hoàn thành

- Ít nhất 3/4 RAGAS metrics ≥ 0.80 trên kết quả chạy thật.
- Version accuracy và negation accuracy ≥ 95%.
- Retrieval recall@5 ≥ 90%.
- P95 latency ≤ 3 giây khi cache model trên CPU.
