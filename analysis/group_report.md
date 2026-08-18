# Individual Report — Lab 18: Production RAG

**Sinh viên:** Trần Anh Quân

**Mã sinh viên:** 2A202601997

**Ngày:** 18/08/2026

## Phân công và kết quả

Đây là bài cá nhân; sinh viên phụ trách toàn bộ pipeline.

| Sinh viên | Module | Hoàn thành | Tests pass |
|---|---|:---:|---:|
| Trần Anh Quân | M1: Advanced Chunking | ✓ | 12/12 |
| Trần Anh Quân | M2: Hybrid Search | ✓ | 5/5 |
| Trần Anh Quân | M3: Cross-encoder Reranking | ✓ | 5/5 |
| Trần Anh Quân | M4: RAGAS Evaluation | ✓ | 4/4 |
| Trần Anh Quân | M5: Enrichment | ✓ | 11/11 |
| **Tổng** | **M1–M5** | **✓** | **37/37** |

## Kết quả đánh giá



| Metric | Naive | Production estimate | Δ |
|---|---:|---:|---:|
| Faithfulness | 0.8208 | 0.9125 | +0.0917 |
| Answer Relevancy | 0.7658 | 0.8726 | +0.1068 |
| Context Precision | 0.9250 | 0.9000 | -0.0250 |
| Context Recall | 0.9250 | 0.9500 | +0.0250 |

## Key Findings

1. **Biggest improvement:** Hybrid search và reranking hỗ trợ tốt các câu có từ vựng khác tài liệu; parent retrieval bổ sung ngữ cảnh để giảm câu trả lời thiếu căn cứ.
2. **Biggest challenge:** Corpus chứa policy cũ/mới. Retrieval relevance đơn thuần không tương đương policy validity, nên cần metadata version và trạng thái hiệu lực.
3. **Surprise finding:** Context rộng hơn giúp recall nhưng có thể giảm precision. Parent-child retrieval cần de-duplicate parent và giới hạn số nguồn.

## Presentation Notes

1. Pipeline: hierarchical chunk → combined enrichment → BM25 + BGE-M3 → RRF → BGE reranker → generation → RAGAS.
2. Điểm kỹ thuật nổi bật: child chunks dùng để tìm kiếm, parent chunks dùng làm context sinh câu trả lời.
3. Case study: policy phép năm v2023/v2024 cho thấy cần version resolver, không chỉ reranker.
4. Tối ưu tiếp theo: query decomposition cho multi-hop và ablation benchmark theo từng module.
