from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import glob
import hashlib
import os
import re
import sys
from dataclasses import dataclass, field

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\n\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────

_model_cache = None


def _get_model():
    """Load SentenceTransformer model một lần và cache lại để tránh load lại nhiều lần."""
    global _model_cache
    if _model_cache is None:
        from sentence_transformers import SentenceTransformer
        _model_cache = SentenceTransformer("all-MiniLM-L6-v2")
    return _model_cache


def _cosine_sim(a, b):
    from numpy import dot
    from numpy.linalg import norm
    return dot(a, b) / (norm(a) * norm(b) + 1e-9)


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    metadata = metadata or {}
    # Tách text thành câu/đoạn nhỏ
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n\n', text) if s.strip()]

    if not sentences:
        return []

    # Encode các câu bằng all-MiniLM-L6-v2
    model = _get_model()
    embeddings = model.encode(sentences)

    chunks = []
    current = [sentences[0]]

    # So cosine similarity giữa hai câu kề nhau
    for i in range(1, len(sentences)):
        sim = _cosine_sim(embeddings[i - 1], embeddings[i])
        if sim < threshold:
            # Dưới ngưỡng → bắt đầu chunk mới
            chunks.append(Chunk(text=" ".join(current),
                                metadata={**metadata, "strategy": "semantic"}))
            current = [sentences[i]]
        else:
            # Trên ngưỡng → gộp vào chunk hiện tại
            current.append(sentences[i])

    if current:
        chunks.append(Chunk(text=" ".join(current),
                            metadata={**metadata, "strategy": "semantic"}))

    return chunks


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    if parent_size <= 0 or child_size <= 0:
        raise ValueError("parent_size and child_size must be positive")
    if child_size >= parent_size:
        raise ValueError("child_size must be smaller than parent_size")

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    def split_oversized(value: str, limit: int) -> list[str]:
        """Split a long paragraph without silently violating the size contract."""
        pieces = []
        remaining = value.strip()
        while len(remaining) > limit:
            cut = remaining.rfind(" ", 0, limit + 1)
            if cut <= 0:
                cut = limit
            pieces.append(remaining[:cut].strip())
            remaining = remaining[cut:].strip()
        if remaining:
            pieces.append(remaining)
        return pieces

    parent_units = [piece for para in paragraphs for piece in split_oversized(para, parent_size)]
    source = str(metadata.get("source", "document"))
    document_key = hashlib.sha1(source.encode("utf-8")).hexdigest()[:10]

    # 1. Gộp paragraphs thành parent chunks (mỗi parent ≤ parent_size chars)
    parents = []
    current = ""
    for para in parent_units:
        separator = "\n\n" if current else ""
        if len(current) + len(separator) + len(para) > parent_size and current:
            pid = f"{document_key}_parent_{len(parents)}"
            parents.append(Chunk(text=current.strip(),
                                 metadata={**metadata, "chunk_type": "parent", "parent_id": pid}))
            current = ""
        current = f"{current}\n\n{para}" if current else para
    if current.strip():
        pid = f"{document_key}_parent_{len(parents)}"
        parents.append(Chunk(text=current.strip(),
                             metadata={**metadata, "chunk_type": "parent", "parent_id": pid}))

    # 2. Mỗi parent → split thành children (mỗi child ≤ child_size chars)
    children = []
    for parent in parents:
        pid = parent.metadata["parent_id"]
        parent_paras = [
            piece
            for paragraph in parent.text.split("\n\n")
            if paragraph.strip()
            for piece in split_oversized(paragraph, child_size)
        ]
        current_child = ""
        for para in parent_paras:
            separator = "\n\n" if current_child else ""
            if len(current_child) + len(separator) + len(para) > child_size and current_child:
                children.append(Chunk(text=current_child.strip(),
                                      metadata={**metadata, "chunk_type": "child"},
                                      parent_id=pid))
                current_child = ""
            current_child = f"{current_child}\n\n{para}" if current_child else para
        if current_child.strip():
            children.append(Chunk(text=current_child.strip(),
                                  metadata={**metadata, "chunk_type": "child"},
                                  parent_id=pid))

    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}
    lines = text.split("\n")
    chunks = []
    current_header = None
    current_lines = []
    in_code_block = False

    def flush():
        nonlocal current_lines, current_header
        if current_lines:
            content = "\n".join(current_lines).strip()
            if content:
                header_text = current_header or ""
                chunk_text = (header_text + "\n\n" + content).strip() if header_text else content
                chunks.append(Chunk(
                    text=chunk_text,
                    metadata={**metadata, "section": current_header or "", "strategy": "structure"}
                ))
        current_lines = []

    for line in lines:
        # Track code blocks để không cắt giữa code block
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            current_lines.append(line)
            continue

        header_match = re.match(r'^(#{1,3})\s+(.+)$', line)
        if header_match and not in_code_block:
            flush()
            current_header = line
        else:
            current_lines.append(line)

    flush()

    # Fallback: nếu không có header nào, trả về toàn bộ text làm 1 chunk
    if not chunks and text.strip():
        chunks.append(Chunk(text=text.strip(),
                            metadata={**metadata, "section": "", "strategy": "structure"}))

    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
