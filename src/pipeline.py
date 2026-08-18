from __future__ import annotations

"""Production RAG Pipeline — ghép M1 + M5 + M2 + M3 + M4."""

import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    # Step 1: Load & Chunk (M1)
    timings = {}
    t0 = time.perf_counter()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        parent_lookup = {parent.metadata["parent_id"]: parent.text for parent in parents}
        for child in children:
            all_chunks.append({
                "text": child.text,
                "metadata": {
                    **child.metadata,
                    "parent_id": child.parent_id,
                    "parent_text": parent_lookup[child.parent_id],
                },
            })
    timings["chunking"] = time.perf_counter() - t0
    print(f"  ✓ {len(all_chunks)} chunks from {len(docs)} documents ({timings['chunking']:.1f}s)", flush=True)

    # Step 2: Enrichment (M5)
    t0 = time.perf_counter()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    enriched = enrich_chunks(all_chunks)
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        timings["enrichment"] = time.perf_counter() - t0
        print(f"  ✓ Enriched {len(enriched)} chunks ({timings['enrichment']:.1f}s)", flush=True)
    else:
        timings["enrichment"] = time.perf_counter() - t0
        print("  ⚠️  Enrichment returned no chunks — using raw chunks", flush=True)

    # Step 3: Index (M2)
    t0 = time.perf_counter()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    timings["indexing"] = time.perf_counter() - t0
    print(f"  ✓ Indexed ({timings['indexing']:.1f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.perf_counter()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    # Load here so initialization time is measured instead of deferred to query 1.
    reranker._load_model()
    timings["reranker_loading"] = time.perf_counter() - t0
    print(f"  ✓ Reranker ready ({timings['reranker_loading']:.1f}s)", flush=True)

    search.pipeline_timings = timings

    return search, reranker


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    results = search.search(query)
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    selected = reranked if reranked else results[:3]
    # Hierarchical retrieval searches precise children but gives the generator
    # their larger parents. De-duplicate when several children share a parent.
    contexts = []
    seen = set()
    for result in selected:
        context = result.metadata.get("parent_text") or result.text
        if context not in seen:
            contexts.append(context)
            seen.add(context)

    from config import OPENAI_API_KEY
    if OPENAI_API_KEY and contexts:
        try:
            from openai import OpenAI
            client = OpenAI()
            context_str = "\n\n".join(contexts)
            resp = client.chat.completions.create(model="gpt-4o-mini", messages=[
                {"role": "system", "content": (
                    "Trả lời ngắn gọn bằng tiếng Việt và CHỈ dựa trên context. "
                    "Nếu có nhiều phiên bản chính sách, ưu tiên bản mới nhất/hiện hành và nêu rõ bản cũ đã bị thay thế. "
                    "Giữ chính xác phủ định, con số và đơn vị. Nếu context không đủ, nói 'Không tìm thấy.'"
                )},
                {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
            ])
            answer = resp.choices[0].message.content
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}", flush=True)
            answer = contexts[0]
    else:
        answer = contexts[0] if contexts else "Không tìm thấy thông tin."
    return answer, contexts


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    t0 = time.perf_counter()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    eval_seconds = time.perf_counter() - t0
    print(f"  ✓ RAGAS done ({eval_seconds:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []), bottom_n=5)
    timings = {**getattr(search, "pipeline_timings", {}), "evaluation": eval_seconds}
    timings["total_recorded"] = sum(timings.values())
    save_report(results, failures, extra={"latency_seconds": timings})
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
