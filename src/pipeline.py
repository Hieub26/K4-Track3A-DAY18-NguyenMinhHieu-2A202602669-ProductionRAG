from __future__ import annotations

"""Production RAG Pipeline — Ghép toàn bộ M1+M2+M3+M4+M5."""

import os, sys, time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K

_PARENTS: dict[tuple, str] = {}       # (source, parent_id) → parent text
LATENCY: dict[str, float] = {}        # bước → giây
_QUERY_TIMES: list[dict] = []         # latency từng query: search / rerank / generate

def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    # Step 1: Load & Chunk (M1)
    t0 = time.time()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    _PARENTS.clear()
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        # parent_id chỉ duy nhất trong 1 document → key theo (source, parent_id).
        for parent in parents:
            _PARENTS[(parent.metadata.get("source"), parent.metadata["parent_id"])] = parent.text
        for child in children:
            all_chunks.append({"text": child.text, "metadata": {**child.metadata, "parent_id": child.parent_id}})
    LATENCY["1. Chunking (M1)"] = time.time() - t0
    print(f"  ✓ {len(all_chunks)} chunks from {len(docs)} documents ({time.time()-t0:.1f}s)", flush=True)

    # Step 2: Enrichment (M5)
    t0 = time.time()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    enriched = enrich_chunks(all_chunks)
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        print(f"  ✓ Enriched {len(enriched)} chunks ({time.time()-t0:.1f}s)", flush=True)
    else:
        print("  ⚠️  M5 not implemented — using raw chunks", flush=True)
    LATENCY["2. Enrichment (M5)"] = time.time() - t0

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    LATENCY["3. Indexing BM25 + Dense (M2)"] = time.time() - t0
    print(f"  ✓ Indexed ({time.time()-t0:.1f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    reranker._load_model()  # load ngay để thời gian load không lẫn vào latency của query đầu tiên
    LATENCY["4. Load reranker (M3)"] = time.time() - t0
    print(f"  ✓ Reranker ready ({time.time()-t0:.1f}s)", flush=True)

    return search, reranker


SYSTEM_PROMPT = (
    "Bạn là trợ lý tra cứu chính sách nội bộ. Trả lời CHỈ dựa trên context, đúng trọng tâm câu hỏi, "
    "giữ nguyên các con số và đơn vị như trong context. "
    # Câu trả lời cụt ("CEO", "1.000.000 VNĐ") bị RAGAS chấm faithfulness = 0 dù đúng → buộc nêu căn cứ.
    "Trả lời bằng câu hoàn chỉnh, nhắc lại quy định/điều kiện trong context làm căn cứ (ngưỡng, thời hạn, đối tượng). "
    "Nếu câu hỏi có nhiều ý thì trả lời đủ từng ý; nếu cần tính toán thì áp dụng quy định để tính ra con số cụ thể. "
    "Nếu context có nhiều phiên bản của cùng một chính sách, trả lời theo phiên bản mới nhất (hiện hành) "
    "và nói rõ phiên bản cũ quy định gì nhưng đã bị thay thế. "
    "Nếu context không có thông tin → nói 'Không tìm thấy.'"
)


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    t0 = time.perf_counter()
    results = search.search(query)
    t1 = time.perf_counter()
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    # Rerank dư ra vài child vì nhiều child có thể chung 1 parent.
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K * 2)
    t2 = time.perf_counter()

    # Retrieve child (precision) → return parent (context): child 256 ký tự quá ngắn để LLM trả lời.
    ranked = reranked if reranked else results[:RERANK_TOP_K]
    contexts: list[str] = []
    for r in ranked:
        parent = _PARENTS.get((r.metadata.get("source"), r.metadata.get("parent_id")), r.text)
        if parent not in contexts:
            contexts.append(parent)
        if len(contexts) == RERANK_TOP_K:
            break

    from config import GEMINI_API_KEY
    if GEMINI_API_KEY and contexts:
        try:
            from src.llm import chat
            context_str = "\n\n---\n\n".join(contexts)
            answer = chat([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
            ]) or contexts[0]
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}", flush=True)
            answer = contexts[0]
    else:
        answer = contexts[0] if contexts else "Không tìm thấy thông tin."
    t3 = time.perf_counter()
    _QUERY_TIMES.append({"search": t1 - t0, "rerank": t2 - t1, "generate": t3 - t2})
    return answer, contexts


def save_latency_report(path: str = "reports/latency_report.json") -> None:
    """In + lưu bảng latency breakdown từng bước của pipeline."""
    import json
    if _QUERY_TIMES:
        n = len(_QUERY_TIMES)
        LATENCY["5. Hybrid search / query (M2)"] = sum(q["search"] for q in _QUERY_TIMES) / n
        LATENCY["6. Rerank / query (M3)"] = sum(q["rerank"] for q in _QUERY_TIMES) / n
        LATENCY["7. LLM generate / query (gồm chờ rate limit)"] = sum(q["generate"] for q in _QUERY_TIMES) / n
    print("\n" + "=" * 60)
    print("LATENCY BREAKDOWN")
    print("=" * 60)
    for step, seconds in LATENCY.items():
        print(f"  {step:<48} {seconds:>9.2f}s")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"unit": "seconds", "num_queries": len(_QUERY_TIMES), "steps": LATENCY},
                  f, ensure_ascii=False, indent=2)
    print(f"Latency report saved to {path}")


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

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    LATENCY["8. RAGAS evaluation (M4)"] = time.time() - t0
    print(f"  ✓ RAGAS done ({time.time()-t0:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)
    save_latency_report()
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
